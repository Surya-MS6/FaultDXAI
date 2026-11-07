import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import Sequence
from scipy.fft import fft, fftfreq
import csv

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure TensorFlow to use less memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# ============================================================================
# ATTENTION MECHANISM LAYER
# ============================================================================
class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer for 1D signals
    Learns which parts of the frequency spectrum are most important
    """

    def __init__(self, return_sequences=True, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Calculate attention scores
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=1)

        if self.return_sequences:
            output = x * a
        else:
            output = keras.backend.sum(x * a, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape
        else:
            return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'return_sequences': self.return_sequences})
        return config


# ============================================================================
# DATA GENERATOR  (fixed — no duplicate first-sample processing)
# ============================================================================
class FrequencyDomainDataGenerator(Sequence):
    """
    Data generator with frequency domain transformation and preprocessing.
    Converts time-domain signals to frequency domain via FFT,
    applies Z-score normalization, and cuts to the relevant frequency range.
    """

    def __init__(self, file_paths, labels, batch_size=32, shuffle=True,
                 sampling_rate=25000, max_freq=500):
        self.file_paths   = file_paths
        self.labels       = labels
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.sampling_rate = sampling_rate
        self.max_freq     = max_freq
        self.indices      = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def time_to_frequency(self, time_signal):
        fft_vals = fft(time_signal)
        fft_mag  = np.abs(fft_vals)
        freqs    = fftfreq(len(time_signal), 1 / self.sampling_rate)

        positive_freqs = freqs[:len(freqs) // 2]
        fft_mag        = fft_mag[:len(fft_mag) // 2]

        freq_mask   = positive_freqs <= self.max_freq
        fft_mag_cut = fft_mag[freq_mask]
        return fft_mag_cut

    def z_score_normalize(self, signal):
        mean = np.mean(signal)
        std  = np.std(signal)
        if std == 0:
            return signal - mean
        return (signal - mean) / std

    def __getitem__(self, index):
        indices     = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.file_paths[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices]
        actual_batch_size = len(batch_paths)

        # Determine frequency-domain length from the first signal
        first_signal = np.load(batch_paths[0])
        if first_signal.shape[0] != 4:
            first_signal = first_signal.T

        num_channels    = 4
        channel_0_freq  = self.time_to_frequency(first_signal[0])
        freq_length     = len(channel_0_freq)

        X = np.zeros((actual_batch_size, freq_length, num_channels), dtype=np.float32)

        # Process ALL samples (including the first — fixes the duplicate bug)
        for i, path in enumerate(batch_paths):
            signal = np.load(path)
            if signal.shape[0] != 4:
                signal = signal.T

            for ch in range(num_channels):
                freq_signal = self.time_to_frequency(signal[ch])
                freq_signal = self.z_score_normalize(freq_signal)
                X[i, :, ch] = freq_signal.astype(np.float32)

        y = np.array(batch_labels, dtype=np.int32)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================================
# MAIN CLASSIFIER CLASS
# ============================================================================
class AttentionFaultClassifier:
    """
    1D CNN + Attention fault classifier.
    Trains 10 separate models, each using a different total number of real
    training samples (10, 20, 30, …, 100), mixed with all synthetic data.
    """

    def __init__(self, data_path, real_data_path=None,
                 val_ratio=0.1, sampling_rate=25000, max_freq=500):
        """
        Args:
            data_path       : Path to the synthetic dataset folder
            real_data_path  : Path to the real dataset folder
            val_ratio       : Fraction of combined data used for validation
            sampling_rate   : Signal sampling rate in Hz
            max_freq        : Upper frequency cut-off in Hz
        """
        self.data_path      = data_path
        self.real_data_path = real_data_path
        self.val_ratio      = val_ratio
        self.sampling_rate  = sampling_rate
        self.max_freq       = max_freq

        self.fault_classes  = ['Normal', 'Misalignment', 'Looseness', 'Unbalance']
        self.class_to_idx   = {cls: idx for idx, cls in enumerate(self.fault_classes)}
        self.model          = None
        self.history        = None
        self.signal_length  = None
        self.num_channels   = 4

    # ------------------------------------------------------------------
    # DATA LOADING — SYNTHETIC
    # ------------------------------------------------------------------
    def get_synthetic_data(self):
        """
        Scan the synthetic dataset and return all file paths + labels.
        Expected folder structure:
            data_path / fault_class / test_folder / aug_folder / *.npy
        Also sets self.signal_length (frequency-domain size).
        """
        print("=" * 70)
        print("SCANNING SYNTHETIC DATASET")
        print("=" * 70)
        print(f"Dataset path: {self.data_path}")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Synthetic data path not found: {self.data_path}")

        print(f"Available folders: {os.listdir(self.data_path)}")

        all_paths  = []
        all_labels = []

        for fault_class in self.fault_classes:
            class_path = os.path.join(self.data_path, fault_class)
            if not os.path.exists(class_path):
                print(f"  Warning: {class_path} does not exist — skipping.")
                continue

            test_folders = sorted([f for f in os.listdir(class_path)
                                   if os.path.isdir(os.path.join(class_path, f))])
            print(f"\n{fault_class}: {len(test_folders)} test folders")

            for test_folder in test_folders:
                test_path   = os.path.join(class_path, test_folder)
                aug_folders = sorted([f for f in os.listdir(test_path)
                                      if os.path.isdir(os.path.join(test_path, f))])

                for aug_folder in aug_folders:
                    aug_path  = os.path.join(test_path, aug_folder)
                    npy_files = sorted([f for f in os.listdir(aug_path) if f.endswith('.npy')])
                    for npy_file in npy_files:
                        all_paths.append(os.path.join(aug_path, npy_file))
                        all_labels.append(self.class_to_idx[fault_class])

            count = sum(1 for l in all_labels if l == self.class_to_idx[fault_class])
            print(f"  Total {fault_class} samples: {count}")

        if not all_paths:
            raise ValueError("No synthetic data files found! Check dataset structure.")

        # Determine frequency-domain signal length
        sample_signal = np.load(all_paths[0])
        print(f"\nFirst sample shape (time domain): {sample_signal.shape}")

        time_length = sample_signal.shape[1] if sample_signal.shape[0] == 4 else sample_signal.shape[0]

        freq_bins        = fftfreq(time_length, 1 / self.sampling_rate)[:time_length // 2]
        freq_mask        = freq_bins <= self.max_freq
        self.signal_length = int(np.sum(freq_mask))

        print(f"\n" + "=" * 70)
        print("SIGNAL PROCESSING PARAMETERS")
        print("=" * 70)
        print(f"Time domain length      : {time_length} points")
        print(f"Sampling rate           : {self.sampling_rate} Hz")
        print(f"Max frequency           : {self.max_freq} Hz")
        print(f"Frequency domain (cut)  : {self.signal_length} points")
        print(f"Channels                : {self.num_channels}")
        print(f"Total synthetic samples : {len(all_paths)}")
        print("=" * 70)

        unique, counts = np.unique(all_labels, return_counts=True)
        print("\nSYNTHETIC CLASS DISTRIBUTION:")
        for idx, cnt in zip(unique, counts):
            print(f"  {self.fault_classes[idx]}: {cnt} ({cnt/len(all_labels)*100:.1f}%)")

        return all_paths, all_labels

    # ------------------------------------------------------------------
    # DATA LOADING — REAL (scan all once)
    # ------------------------------------------------------------------
    def get_all_real_data(self):
        """
        Scan and return ALL available real-data file paths + labels.
        Expected folder structure:
            real_data_path / fault_class / **/*.npy   (any depth)
        """
        if self.real_data_path is None:
            print("No real data path provided — skipping real data.")
            return [], []

        print("\n" + "=" * 70)
        print("SCANNING REAL DATASET")
        print("=" * 70)
        print(f"Real dataset path: {self.real_data_path}")

        if not os.path.exists(self.real_data_path):
            print(f"Warning: real data path not found: {self.real_data_path}")
            return [], []

        all_real_paths  = []
        all_real_labels = []

        for fault_class in self.fault_classes:
            class_path = os.path.join(self.real_data_path, fault_class)
            if not os.path.exists(class_path):
                print(f"  Warning: {class_path} does not exist — skipping.")
                continue

            class_files = []
            for root, _, files in os.walk(class_path):
                for f in files:
                    if f.endswith('.npy'):
                        class_files.append(os.path.join(root, f))

            print(f"  {fault_class}: {len(class_files)} files found")
            for fp in class_files:
                all_real_paths.append(fp)
                all_real_labels.append(self.class_to_idx[fault_class])

        print(f"\nTotal real samples available: {len(all_real_paths)}")
        print("=" * 70)
        return all_real_paths, all_real_labels

    # ------------------------------------------------------------------
    # REAL DATA SAMPLING  (balanced across 4 classes)
    # ------------------------------------------------------------------
    def sample_real_data_balanced(self, all_real_paths, all_real_labels, total_samples):
        """
        Randomly select `total_samples` real samples, balanced across classes.
        E.g. total_samples=40 → 10 per class.

        Args:
            all_real_paths  : Full list of real file paths
            all_real_labels : Corresponding labels
            total_samples   : Desired total count (split evenly across 4 classes)

        Returns:
            sampled_paths, sampled_labels
        """
        if not all_real_paths:
            return [], []

        num_classes       = len(self.fault_classes)
        samples_per_class = total_samples // num_classes

        print(f"\n" + "=" * 70)
        print(f"SAMPLING {total_samples} TOTAL REAL SAMPLES")
        print("=" * 70)
        print(f"Target per class: {samples_per_class}")

        sampled_paths  = []
        sampled_labels = []

        for fault_class in self.fault_classes:
            class_idx    = self.class_to_idx[fault_class]
            class_paths  = [p for p, l in zip(all_real_paths, all_real_labels) if l == class_idx]

            if len(class_paths) >= samples_per_class:
                chosen = list(np.random.choice(class_paths, samples_per_class, replace=False))
                print(f"  {fault_class}: selected {samples_per_class}/{len(class_paths)}")
            else:
                chosen = class_paths
                print(f"  {fault_class}: selected ALL {len(class_paths)} (fewer than {samples_per_class})")

            sampled_paths.extend(chosen)
            sampled_labels.extend([class_idx] * len(chosen))

        print(f"Total selected: {len(sampled_paths)}")
        print("=" * 70)
        return sampled_paths, sampled_labels

    # ------------------------------------------------------------------
    # COMBINE SYNTHETIC + REAL
    # ------------------------------------------------------------------
    def combine_and_split(self, synthetic_paths, synthetic_labels,
                          real_paths, real_labels):
        """
        Merge synthetic and real data then do a stratified 90/10 train/val split.

        Returns:
            paths_train, paths_val, labels_train, labels_val
        """
        print("\n" + "=" * 70)
        print("COMBINING SYNTHETIC AND REAL DATA")
        print("=" * 70)

        if real_paths:
            combined_paths  = np.concatenate([np.array(synthetic_paths), np.array(real_paths)])
            combined_labels = np.concatenate([np.array(synthetic_labels), np.array(real_labels)])
            print(f"Synthetic : {len(synthetic_paths)} samples")
            print(f"Real      : {len(real_paths)} samples")
        else:
            combined_paths  = np.array(synthetic_paths)
            combined_labels = np.array(synthetic_labels)
            print("Using ONLY synthetic data (no real data added).")

        print(f"Total     : {len(combined_paths)} samples")

        # Stratified split
        paths_train, paths_val, labels_train, labels_val = train_test_split(
            combined_paths, combined_labels,
            test_size=self.val_ratio,
            random_state=42,
            stratify=combined_labels
        )

        print(f"\nTrain split : {len(paths_train)} samples (90%)")
        print(f"Val split   : {len(paths_val)} samples (10%)")

        print("\nTRAIN CLASS DISTRIBUTION:")
        for fault_class in self.fault_classes:
            cnt = int(np.sum(labels_train == self.class_to_idx[fault_class]))
            print(f"  {fault_class}: {cnt}")

        print("\nVALIDATION CLASS DISTRIBUTION:")
        for fault_class in self.fault_classes:
            cnt = int(np.sum(labels_val == self.class_to_idx[fault_class]))
            print(f"  {fault_class}: {cnt}")

        print("=" * 70)
        return paths_train, paths_val, labels_train, labels_val

    # ------------------------------------------------------------------
    # MODEL DEFINITION
    # ------------------------------------------------------------------
    def build_model(self, input_shape, num_classes):
        """
        1D CNN + Attention architecture:
          Conv1D (32 filters, kernel 5, ReLU)
          → AttentionLayer
          → Dropout 0.3
          → MaxPooling1D (pool 2)
          → Flatten
          → Dense 128 (ReLU)
          → Dropout 0.3
          → Dense num_classes (Softmax)
        """
        print(f"\n" + "=" * 70)
        print("BUILDING 1D CNN + ATTENTION MODEL")
        print("=" * 70)
        print(f"Input shape : {input_shape}  (frequency_bins × channels)")
        print(f"Output      : {num_classes} classes (Softmax)")
        print("=" * 70)

        model = keras.Sequential([
            layers.Conv1D(
                filters=32,
                kernel_size=5,
                activation='relu',
                input_shape=input_shape,
                padding='same',
                name='conv1d_layer'
            ),
            AttentionLayer(return_sequences=True, name='attention_layer'),
            layers.Dropout(0.3, name='dropout_1'),
            layers.MaxPooling1D(pool_size=2, name='pooling_layer'),
            layers.Flatten(name='flatten_layer'),
            layers.Dense(128, activation='relu', name='fc_layer'),
            layers.Dropout(0.3, name='dropout_2'),
            layers.Dense(num_classes, activation='softmax', name='output_layer')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    def train_model(self, paths_train, labels_train,
                    paths_val, labels_val,
                    num_real_samples=0, epochs=100,
                    checkpoint_path='best_model_checkpoint.keras'):
        """
        Train the model with:
          - Early stopping (patience 10, min_delta 0.001)
          - ReduceLROnPlateau  (patience 5, factor 0.5)
          - ModelCheckpoint    (best val_accuracy)
        """
        print("\n" + "=" * 70)
        print(f"TRAINING — Synthetic + {num_real_samples} Real samples")
        print("=" * 70)
        print("Hyperparameters:")
        print("  Batch size         : 32")
        print("  Max epochs         : 100")
        print("  Initial LR         : 0.001")
        print("  Early stopping     : patience=10, min_delta=0.001")
        print("  LR reduction       : patience=5, factor=0.5")
        print("=" * 70)

        train_gen = FrequencyDomainDataGenerator(
            paths_train, labels_train,
            batch_size=32, shuffle=True,
            sampling_rate=self.sampling_rate,
            max_freq=self.max_freq
        )
        val_gen = FrequencyDomainDataGenerator(
            paths_val, labels_val,
            batch_size=32, shuffle=False,
            sampling_rate=self.sampling_rate,
            max_freq=self.max_freq
        )

        input_shape = (self.signal_length, self.num_channels)
        num_classes  = len(self.fault_classes)
        self.model   = self.build_model(input_shape, num_classes)

        print("\nModel Summary:")
        self.model.summary()

        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                min_delta=0.001,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]

        print("\nStarting training…")
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        total_epochs = len(self.history.history['accuracy'])
        best_val_acc = max(self.history.history['val_accuracy'])
        print(f"\nTraining complete — {total_epochs} epochs")
        print(f"  Final train accuracy : {self.history.history['accuracy'][-1]:.4f}")
        print(f"  Final val accuracy   : {self.history.history['val_accuracy'][-1]:.4f}")
        print(f"  Best  val accuracy   : {best_val_acc:.4f}")

        return self.history

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------
    def evaluate_model(self, paths_test, labels_test):
        """Run prediction on paths_test and return (accuracy, y_pred, cm)."""
        print("\n" + "=" * 70)
        print("EVALUATING MODEL")
        print("=" * 70)

        test_gen = FrequencyDomainDataGenerator(
            paths_test, labels_test,
            batch_size=32, shuffle=False,
            sampling_rate=self.sampling_rate,
            max_freq=self.max_freq
        )

        y_pred_probs = self.model.predict(test_gen, verbose=1)
        y_pred       = np.argmax(y_pred_probs, axis=1)
        y_true       = np.array(labels_test)

        accuracy = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=self.fault_classes, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)
        return accuracy, y_pred, cm

    # ------------------------------------------------------------------
    # SINGLE-RUN PLOT (training curves + confusion matrix)
    # ------------------------------------------------------------------
    def plot_training_results(self, cm, num_real_samples, output_dir):
        """Save training-curve + confusion-matrix figure for one run."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Accuracy curve
        axes[0].plot(self.history.history['accuracy'],     label='Train', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Val',   linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title(f'Accuracy — {num_real_samples} Real Samples', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Loss curve
        axes[1].plot(self.history.history['loss'],     label='Train', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Val',   linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title(f'Loss — {num_real_samples} Real Samples', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.fault_classes,
                    yticklabels=self.fault_classes,
                    ax=axes[2], cbar_kws={'label': 'Count'})
        axes[2].set_xlabel('Predicted', fontsize=12)
        axes[2].set_ylabel('Actual', fontsize=12)
        axes[2].set_title(f'Confusion Matrix — {num_real_samples} Real Samples',
                          fontsize=13, fontweight='bold')
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        fname = os.path.join(output_dir, f'training_results_real_{num_real_samples}.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"  Plot saved: {fname}")
        plt.close()

    # ------------------------------------------------------------------
    # SUMMARY PLOT (all runs combined)
    # ------------------------------------------------------------------
    def plot_summary(self, results, output_dir):
        """Two-panel summary: accuracy trend + epochs trained per run."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        num_samples = results['num_real_samples']
        train_accs  = results['train_accuracy']
        val_accs    = results['val_accuracy']

        axes[0].plot(num_samples, train_accs, 'o-', label='Training Accuracy',
                     linewidth=2, markersize=8, color='#2E86AB')
        axes[0].plot(num_samples, val_accs,   's-', label='Validation Accuracy',
                     linewidth=2, markersize=8, color='#A23B72')
        axes[0].set_xlabel('Number of Real Samples', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Performance vs Real Data Amount', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(num_samples)

        axes[1].bar(num_samples, results['epochs_trained'], color='#F18F01', alpha=0.7)
        axes[1].set_xlabel('Number of Real Samples', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Epochs Trained', fontsize=12, fontweight='bold')
        axes[1].set_title('Training Duration (Early Stopping)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_xticks(num_samples)

        plt.tight_layout()
        fname = os.path.join(output_dir, 'summary_results.png')
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"\nSummary plot saved: {fname}")
        plt.close()

    # ------------------------------------------------------------------
    # CSV EXPORT
    # ------------------------------------------------------------------
    def save_results_csv(self, results, output_dir):
        csv_file = os.path.join(output_dir, 'training_results.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Num_Real_Samples', 'Train_Accuracy', 'Val_Accuracy', 'Epochs_Trained'])
            for i in range(len(results['num_real_samples'])):
                writer.writerow([
                    results['num_real_samples'][i],
                    f"{results['train_accuracy'][i]:.4f}",
                    f"{results['val_accuracy'][i]:.4f}",
                    results['epochs_trained'][i]
                ])
        print(f"Results CSV saved: {csv_file}")

    # ------------------------------------------------------------------
    # MAIN LOOP — train with 10, 20, …, 100 real samples
    # ------------------------------------------------------------------
    def train_with_varying_real_samples(self, real_sample_range, output_dir='models_attention'):
        """
        Core experiment loop.
        For each value in real_sample_range:
          1. Sample that many real samples (balanced across 4 classes)
          2. Combine with all synthetic data
          3. Split 90 / 10 (train / val)
          4. Train CNN+Attention model
          5. Evaluate on validation set
          6. Save model as  model_real_<N>.keras
          7. Save training-curve + confusion-matrix plot

        At the end, saves a summary plot and CSV.

        Args:
            real_sample_range : iterable of ints, e.g. range(10, 101, 10)
            output_dir        : directory for saved models and plots
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print("ATTENTION MODEL — VARYING REAL DATA EXPERIMENT")
        print("=" * 70)
        print(f"Real sample counts : {list(real_sample_range)}")
        print(f"Output directory   : {output_dir}/")
        print("=" * 70)

        # ── Load synthetic data once ──────────────────────────────────
        synthetic_paths, synthetic_labels = self.get_synthetic_data()

        # ── Load ALL real data once ───────────────────────────────────
        all_real_paths, all_real_labels = self.get_all_real_data()

        if not all_real_paths:
            print("\nERROR: No real data found — aborting experiment.")
            return

        # ── Result storage ────────────────────────────────────────────
        results = {
            'num_real_samples': [],
            'train_accuracy':   [],
            'val_accuracy':     [],
            'epochs_trained':   []
        }

        # ── Iterate over each real-sample count ───────────────────────
        for num_real_samples in real_sample_range:
            print("\n" + "#" * 70)
            print(f"### TRAINING WITH {num_real_samples} TOTAL REAL SAMPLES ###")
            print("#" * 70)

            # Release previous model from GPU / RAM
            tf.keras.backend.clear_session()
            gc.collect()

            # 1. Sample balanced real data
            real_paths, real_labels = self.sample_real_data_balanced(
                all_real_paths, all_real_labels, num_real_samples
            )

            # 2. Combine + split
            paths_train, paths_val, labels_train, labels_val = self.combine_and_split(
                synthetic_paths, synthetic_labels,
                real_paths, real_labels
            )

            # 3. Train
            checkpoint_path = os.path.join(output_dir,
                                           f'checkpoint_real_{num_real_samples}.keras')
            self.train_model(
                paths_train, labels_train,
                paths_val,   labels_val,
                num_real_samples=num_real_samples,
                checkpoint_path=checkpoint_path
            )

            # 4. Record metrics
            train_acc     = self.history.history['accuracy'][-1]
            val_acc       = self.history.history['val_accuracy'][-1]
            epochs_trained = len(self.history.history['accuracy'])

            results['num_real_samples'].append(num_real_samples)
            results['train_accuracy'].append(train_acc)
            results['val_accuracy'].append(val_acc)
            results['epochs_trained'].append(epochs_trained)

            # 5. Evaluate
            accuracy, y_pred, cm = self.evaluate_model(paths_val, labels_val)

            # 6. Plot training curves + confusion matrix
            self.plot_training_results(cm, num_real_samples, output_dir)

            # 7. Save model
            model_path = os.path.join(output_dir, f'model_real_{num_real_samples}.keras')
            self.model.save(model_path)

            print(f"\n{'='*70}")
            print(f"Model saved : {model_path}")
            print(f"Train acc   : {train_acc:.4f}")
            print(f"Val acc     : {val_acc:.4f}")
            print(f"Epochs      : {epochs_trained}")
            print(f"{'='*70}")

        # ── End-of-experiment outputs ─────────────────────────────────
        self.plot_summary(results, output_dir)
        self.save_results_csv(results, output_dir)

        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":

    # ── Paths — UPDATE THESE ──────────────────────────────────────────
    SYNTHETIC_DATA_PATH = r"D:\Final Project\Fault_Detection"
    REAL_DATA_PATH      = r"D:\Final Project\New folder\Real Dataset"
    OUTPUT_DIR          = "models_attention_varying_real"

    # ── Experiment range: 10, 20, 30, …, 100 ─────────────────────────
    REAL_SAMPLE_RANGE = range(70, 101, 10)   # 10 models total

    SAMPLING_RATE = 25000   # Hz
    MAX_FREQ      = 500    # Hz

    print("=" * 70)
    print("1D CNN + ATTENTION — VARYING REAL DATA TRAINING")
    print("=" * 70)
    print("Architecture  : Conv1D → Attention → Dropout → MaxPool → FC → Output")
    print("Experiment    : Train 10 models with 10, 20, …, 100 real samples")
    print("Real sampling : Balanced across all 4 fault classes")
    print("Each model    : Saved separately in output directory")
    print("=" * 70)

    classifier = AttentionFaultClassifier(
        data_path      = SYNTHETIC_DATA_PATH,
        real_data_path = REAL_DATA_PATH,
        val_ratio      = 0.1,
        sampling_rate  = SAMPLING_RATE,
        max_freq       = MAX_FREQ
    )

    results = classifier.train_with_varying_real_samples(
        real_sample_range = REAL_SAMPLE_RANGE,
        output_dir        = OUTPUT_DIR
    )

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETE — FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Real Samples':>14} | {'Train Acc':>10} | {'Val Acc':>9} | {'Epochs':>7}")
    print("-" * 50)
    for i in range(len(results['num_real_samples'])):
        n  = results['num_real_samples'][i]
        ta = results['train_accuracy'][i]
        va = results['val_accuracy'][i]
        ep = results['epochs_trained'][i]
        print(f"{n:>14} | {ta:>10.4f} | {va:>9.4f} | {ep:>7}")

    print("\nOutputs saved in:", OUTPUT_DIR)
    print("  model_real_10.keras  …  model_real_100.keras")
    print("  training_results_real_10.png  …  training_results_real_100.png")
    print("  summary_results.png")
    print("  training_results.csv")

    print("\nSystem Information:")
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  GPU        : {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("=" * 70)