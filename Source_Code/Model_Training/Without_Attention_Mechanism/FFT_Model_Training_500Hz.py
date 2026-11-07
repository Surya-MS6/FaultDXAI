import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import Sequence
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq

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


class FrequencyDomainDataGenerator(Sequence):
    """
    Data generator with frequency domain transformation and preprocessing
    As per paper: converts time domain to frequency domain, applies Z-score normalization,
    and cuts signal to relevant frequency range
    """

    def __init__(self, file_paths, labels, batch_size=32, shuffle=True,
                 sampling_rate=25000, max_freq=500):
        """
        Args:
            file_paths: List of file paths
            labels: Corresponding labels
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            sampling_rate: Sampling rate of signals (Hz)
            max_freq: Maximum frequency to consider (Hz) - cuts signal to reduce computational cost
        """
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampling_rate = sampling_rate
        self.max_freq = max_freq
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def time_to_frequency(self, time_signal):
        """
        Convert time domain signal to frequency domain using FFT
        Returns magnitude spectrum
        """
        # Apply FFT
        fft_vals = fft(time_signal)
        fft_mag = np.abs(fft_vals)

        # Get frequency bins
        freqs = fftfreq(len(time_signal), 1 / self.sampling_rate)

        # Take only positive frequencies
        positive_freqs = freqs[:len(freqs) // 2]
        fft_mag = fft_mag[:len(fft_mag) // 2]

        # Cut to max_freq to reduce computational cost
        freq_mask = positive_freqs <= self.max_freq
        fft_mag_cut = fft_mag[freq_mask]

        return fft_mag_cut

    def z_score_normalize(self, signal):
        """
        Z-score normalization (standardization)
        As per paper: signal normalization using Z-score
        """
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return signal - mean
        return (signal - mean) / std

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.file_paths[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices]

        actual_batch_size = len(batch_paths)

        # Load first signal to determine dimensions
        first_signal = np.load(batch_paths[0])
        if first_signal.shape[0] != 4:
            first_signal = first_signal.T

        num_channels = 4

        # Process first signal to get frequency domain length
        channel_0_time = first_signal[0]
        channel_0_freq = self.time_to_frequency(channel_0_time)
        freq_length = len(channel_0_freq)

        # Pre-allocate arrays for all channels in frequency domain
        X = np.zeros((actual_batch_size, freq_length, num_channels), dtype=np.float32)

        # Process first sample - all channels
        for ch in range(num_channels):
            freq_signal = self.time_to_frequency(first_signal[ch])
            freq_signal = self.z_score_normalize(freq_signal)
            X[0, :, ch] = freq_signal.astype(np.float32)

        # Process remaining samples
        for i, path in enumerate(batch_paths[1:], 1):
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


class SyntheticFaultClassifier:
    def __init__(self, data_path, val_ratio=0.1, sampling_rate=25000, max_freq=500):
        """
        Initialize the synthetic fault classifier with frequency domain processing

        As per paper methodology:
        - 4 mechanical faults: Normal, Misalignment, Looseness, Unbalance
        - Frequency domain analysis for interpretability
        - Z-score normalization
        - Signal cutting to reduce computational cost

        Args:
            data_path: Path to the synthetic dataset folder
            val_ratio: Ratio of training data to use for validation (0.1 = 10%)
            sampling_rate: Sampling rate in Hz (default 25000 Hz)
            max_freq: Maximum frequency to consider (default 1000 Hz)
        """
        self.data_path = data_path
        self.val_ratio = val_ratio
        self.sampling_rate = sampling_rate
        self.max_freq = max_freq

        # 4 mechanical faults as per paper
        self.fault_classes = ['Normal', 'Misalignment', 'Looseness', 'Unbalance']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.fault_classes)}
        self.model = None
        self.history = None
        self.signal_length = None
        self.num_channels = 4

    def get_file_paths_and_labels(self):
        """
        Get all file paths and labels from synthetic augmented dataset
        Structure: fault_class/test_folder/augmented_folder/*.npy
        """
        print("=" * 70)
        print("SCANNING SYNTHETIC DATASET")
        print("=" * 70)
        print(f"Dataset path: {self.data_path}")

        if os.path.exists(self.data_path):
            available_folders = os.listdir(self.data_path)
            print(f"Available folders: {available_folders}")
        else:
            raise FileNotFoundError(f"Dataset path does not exist: {self.data_path}")

        all_paths = []
        all_labels = []

        for fault_class in self.fault_classes:
            class_path = os.path.join(self.data_path, fault_class)

            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist!")
                continue

            # Get all test folders
            test_folders = sorted([f for f in os.listdir(class_path)
                                   if os.path.isdir(os.path.join(class_path, f))])

            print(f"\n{fault_class}: Found {len(test_folders)} test folders")

            for test_folder in test_folders:
                test_path = os.path.join(class_path, test_folder)

                # Get all augmented folders
                aug_folders = sorted([f for f in os.listdir(test_path)
                                      if os.path.isdir(os.path.join(test_path, f))])

                for aug_folder in aug_folders:
                    aug_path = os.path.join(test_path, aug_folder)

                    # Load all .npy files
                    npy_files = sorted([f for f in os.listdir(aug_path) if f.endswith('.npy')])

                    for npy_file in npy_files:
                        file_path = os.path.join(aug_path, npy_file)
                        all_paths.append(file_path)
                        all_labels.append(self.class_to_idx[fault_class])

            count = len([l for l in all_labels if l == self.class_to_idx[fault_class]])
            print(f"  Total {fault_class} samples: {count}")

        if len(all_paths) == 0:
            raise ValueError("No data files found! Please check your dataset structure.")

        # Get signal shape from first file
        sample_signal = np.load(all_paths[0])
        print(f"\nFirst sample shape (time domain): {sample_signal.shape}")

        if sample_signal.shape[0] == 4:
            time_length = sample_signal.shape[1]
        elif sample_signal.shape[1] == 4:
            time_length = sample_signal.shape[0]
        else:
            raise ValueError(f"Unexpected signal shape: {sample_signal.shape}")

        # Calculate frequency domain length after FFT and cutting
        freq_length = len(fftfreq(time_length, 1 / self.sampling_rate)[:time_length // 2])
        freq_bins = fftfreq(time_length, 1 / self.sampling_rate)[:time_length // 2]
        freq_mask = freq_bins <= self.max_freq
        self.signal_length = np.sum(freq_mask)

        print(f"\n" + "=" * 70)
        print("SIGNAL PROCESSING PARAMETERS (as per paper)")
        print("=" * 70)
        print(f"Time domain length: {time_length} points")
        print(f"Sampling rate: {self.sampling_rate} Hz")
        print(f"Frequency domain (full): {freq_length} points")
        print(f"Max frequency considered: {self.max_freq} Hz")
        print(f"Frequency domain (cut): {self.signal_length} points")
        print(f"Number of channels: {self.num_channels}")
        print(f"Total samples: {len(all_paths)}")
        print("=" * 70)

        # Check class distribution
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"\nCLASS DISTRIBUTION:")
        for cls_idx, count in zip(unique, counts):
            cls_name = self.fault_classes[cls_idx]
            print(f"  {cls_name}: {count} samples ({count / len(all_labels) * 100:.1f}%)")

        return all_paths, all_labels

    def split_data_for_synthetic(self, file_paths, labels):
        """
        Split synthetic data into train and validation sets (90/10 split)
        As per paper: 90% training, 10% validation using cross-validation method
        """
        print("\n" + "=" * 70)
        print("DATA SPLITTING (as per paper)")
        print("=" * 70)

        file_paths = np.array(file_paths)
        labels = np.array(labels)

        # 90% training, 10% validation - stratified to maintain class distribution
        paths_train, paths_val, labels_train, labels_val = train_test_split(
            file_paths, labels, test_size=self.val_ratio,
            random_state=42, stratify=labels
        )

        print(f"Training set: {len(paths_train)} samples (90%)")
        print(f"Validation set: {len(paths_val)} samples (10%)")
        print("Note: Training group composed ONLY of synthetic signals")
        print("      Test set will be composed of REAL signals (not used here)")
        print("=" * 70)

        return paths_train, paths_val, labels_train, labels_val

    def build_model(self, input_shape, num_classes):
        """
        Build 1D CNN model according to paper specifications

        Architecture (as per paper):
        - 1D CNN layer (32 feature maps, kernel size 5, ReLU activation)
        - Dropout layer (regularization to slow down learning and avoid overfitting)
        - Pooling layer (consolidate features, keep only essential ones)
        - Flatten layer
        - Fully connected layer (buffer between features and output for interpretation)
        - Output layer (Softmax for multi-class classification)

        Objective: Use only 1 CNN layer to reduce complexity and computational cost
        """
        print(f"\n" + "=" * 70)
        print("BUILDING MODEL (as per paper architecture)")
        print("=" * 70)
        print(f"Input shape: {input_shape} (frequency_bins, channels)")
        print(f"Architecture:")
        print(f"  - 1D CNN: 32 filters, kernel size 5, ReLU")
        print(f"  - Dropout: 0.5 (slow down learning, avoid overfitting)")
        print(f"  - MaxPooling: pool_size 2 (consolidate features)")
        print(f"  - Flatten")
        print(f"  - Dense: 128 units, ReLU (buffer for interpretation)")
        print(f"  - Output: {num_classes} units, Softmax")
        print("=" * 70)

        model = keras.Sequential([
            # 1D CNN layer with 32 parallel feature maps, kernel size 5
            # High learning capacity, processes all channels simultaneously
            layers.Conv1D(
                filters=32,
                kernel_size=5,
                activation='relu',
                input_shape=input_shape,
                padding='same',
                name='conv1d_layer'
            ),

            # Dropout for regularization - slows down learning to avoid overfitting
            # As per paper: "dropout layer was used to slow down the learning process"
            layers.Dropout(0.5, name='dropout_layer'),

            # Pooling layer - consolidates learned features, keeps only essential ones
            # As per paper: "prevents network from learning features uncorrelated with condition"
            layers.MaxPooling1D(pool_size=2, name='pooling_layer'),

            # Flatten to 1D vector
            layers.Flatten(name='flatten_layer'),

            # Fully connected layer - buffer between features and output
            # As per paper: "provides buffer with intention of interpreting learned features"
            layers.Dense(128, activation='relu', name='fc_layer'),

            # Output layer with Softmax for multi-class classification
            layers.Dense(num_classes, activation='softmax', name='output_layer')
        ])

        # Compile with Adam optimizer and categorical crossentropy
        # As per paper: Adam optimizer, categorical cross entropy loss
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_model(self, paths_train, labels_train, paths_val, labels_val, epochs=100):
        """
        Train the model with early stopping (as per paper)

        As per paper:
        - Variable number of epochs based on Early Stopping
        - Patience: 8
        - Min delta: 0.001 for validation accuracy
        - Batch size: 32
        """
        print("\n" + "=" * 70)
        print("TRAINING MODEL (as per paper methodology)")
        print("=" * 70)
        print("Training on SYNTHETIC signals in FREQUENCY DOMAIN")
        print("Hyperparameters:")
        print("  - Batch size: 32")
        print("  - Max epochs: 100 (variable with Early Stopping)")
        print("  - Early Stopping patience: 8")
        print("  - Early Stopping min_delta: 0.001")
        print("  - Optimizer: Adam")
        print("  - Loss: Categorical Cross Entropy")
        print("=" * 70)

        # Create frequency domain data generators
        train_gen = FrequencyDomainDataGenerator(
            paths_train, labels_train,
            batch_size=32,
            shuffle=True,
            sampling_rate=self.sampling_rate,
            max_freq=self.max_freq
        )
        val_gen = FrequencyDomainDataGenerator(
            paths_val, labels_val,
            batch_size=32,
            shuffle=False,
            sampling_rate=self.sampling_rate,
            max_freq=self.max_freq
        )

        # Build model
        input_shape = (self.signal_length, self.num_channels)
        num_classes = len(self.fault_classes)
        self.model = self.build_model(input_shape, num_classes)

        print("\nModel Architecture:")
        self.model.summary()

        # Early stopping callback - as per paper specifications
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )

        print("\nStarting training...")
        # Train model
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=[early_stop],
            verbose=1
        )

        print(f"\nTraining completed!")
        print(f"Total epochs trained: {len(self.history.history['accuracy'])}")
        print(f"Final training accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {self.history.history['val_accuracy'][-1]:.4f}")

        return self.history

    def evaluate_model(self, paths_test, labels_test):
        """Evaluate model on test set"""
        print("\n" + "=" * 70)
        print("EVALUATING MODEL")
        print("=" * 70)

        # Create test generator
        test_gen = FrequencyDomainDataGenerator(
            paths_test, labels_test,
            batch_size=32,
            shuffle=False,
            sampling_rate=self.sampling_rate,
            max_freq=self.max_freq
        )

        # Predictions
        y_pred_probs = self.model.predict(test_gen, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_test = labels_test

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=self.fault_classes,
                                    digits=4))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        return accuracy, y_pred, cm

    def plot_results(self, cm):
        """Plot training history and confusion matrix"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot training & validation accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy\n(Frequency Domain - Synthetic Data)', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot training & validation loss
        axes[1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Model Loss\n(Frequency Domain - Synthetic Data)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.fault_classes,
                    yticklabels=self.fault_classes,
                    ax=axes[2], cbar_kws={'label': 'Count'})
        axes[2].set_xlabel('Predicted', fontsize=12)
        axes[2].set_ylabel('Actual', fontsize=12)
        axes[2].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('frequency_domain_training_results.png', dpi=300, bbox_inches='tight')
        print("\nPlots saved as 'frequency_domain_training_results.png'")
        plt.show()

    def run_multiple_trials(self, num_trials=10):
        """
        Run multiple training trials to assess stability (as per paper)

        As per paper: "To ensure that the model is stable and robust to variations
        in both network training and signal generation, the analyses were repeated 10 times.
        At each analysis, new synthetic signals were randomly generated and model trained."

        Standard deviation is calculated to measure variability.
        """
        print(f"\n" + "=" * 70)
        print(f"STABILITY ANALYSIS - {num_trials} TRIALS")
        print("=" * 70)
        print("As per paper methodology:")
        print("  - Assessing robustness to network stochasticity")
        print("  - Assessing robustness to signal generation variations")
        print("  - Each trial: new random generation + new training")
        print("  - Metric: Standard deviation of accuracy")
        print("=" * 70)

        # Get file paths once
        file_paths, labels = self.get_file_paths_and_labels()
        paths_train, paths_val, labels_train, labels_val = \
            self.split_data_for_synthetic(file_paths, labels)

        train_accuracies = []
        val_accuracies = []

        for trial in range(num_trials):
            print(f"\n" + "=" * 70)
            print(f"TRIAL {trial + 1}/{num_trials}")
            print("=" * 70)

            # Clear session to free memory
            tf.keras.backend.clear_session()

            # Reset random seeds for each trial (simulate new generation)
            np.random.seed(trial)
            tf.random.set_seed(trial)

            # Train model
            self.train_model(paths_train, labels_train, paths_val, labels_val)

            # Get final training accuracy
            train_acc = self.history.history['accuracy'][-1]
            train_accuracies.append(train_acc)

            # Evaluate on validation set
            val_gen = FrequencyDomainDataGenerator(
                paths_val, labels_val,
                batch_size=32,
                shuffle=False,
                sampling_rate=self.sampling_rate,
                max_freq=self.max_freq
            )
            val_loss, val_acc = self.model.evaluate(val_gen, verbose=0)
            val_accuracies.append(val_acc)

            print(f"\nTrial {trial + 1} Results:")
            print(f"  Training Accuracy: {train_acc:.4f} ({train_acc * 100:.2f}%)")
            print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")

            # Force garbage collection
            del self.model
            gc.collect()

        # Calculate statistics
        mean_train = np.mean(train_accuracies)
        std_train = np.std(train_accuracies)
        mean_val = np.mean(val_accuracies)
        std_val = np.std(val_accuracies)

        print(f"\n" + "=" * 70)
        print(f"STABILITY ANALYSIS RESULTS ({num_trials} trials)")
        print("=" * 70)
        print(f"\nTraining Accuracy:")
        print(f"  Mean ± Std: {mean_train:.4f} ± {std_train:.4f}")
        print(f"  Min: {np.min(train_accuracies):.4f}")
        print(f"  Max: {np.max(train_accuracies):.4f}")
        print(f"\nValidation Accuracy:")
        print(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
        print(f"  Min: {np.min(val_accuracies):.4f}")
        print(f"  Max: {np.max(val_accuracies):.4f}")
        print(f"\nVariability (Std Dev): {std_val:.4f}")
        print(f"Coefficient of Variation: {(std_val / mean_val) * 100:.2f}%")
        print("=" * 70)

        # Plot stability results
        self.plot_stability_results(train_accuracies, val_accuracies, num_trials)

        return train_accuracies, val_accuracies, mean_val, std_val

    def plot_stability_results(self, train_accs, val_accs, num_trials):
        """Plot stability analysis results"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        trials = range(1, num_trials + 1)

        # Plot 1: Training and Validation Accuracies per trial
        axes[0].plot(trials, train_accs, 'o-', label='Training Accuracy',
                     linewidth=2, markersize=8, color='#2E86AB')
        axes[0].plot(trials, val_accs, 's-', label='Validation Accuracy',
                     linewidth=2, markersize=8, color='#A23B72')
        axes[0].axhline(y=np.mean(train_accs), color='#2E86AB', linestyle='--',
                        alpha=0.5, linewidth=2, label=f'Mean Train: {np.mean(train_accs):.4f}')
        axes[0].axhline(y=np.mean(val_accs), color='#A23B72', linestyle='--',
                        alpha=0.5, linewidth=2, label=f'Mean Val: {np.mean(val_accs):.4f}')
        axes[0].fill_between(trials,
                             np.mean(val_accs) - np.std(val_accs),
                             np.mean(val_accs) + np.std(val_accs),
                             alpha=0.2, color='#A23B72')
        axes[0].set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Model Stability Analysis ({num_trials} Trials)\nFrequency Domain',
                          fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(trials)
        axes[0].set_ylim([min(min(train_accs), min(val_accs)) - 0.05, 1.0])

        # Plot 2: Box plot comparison
        data_to_plot = [train_accs, val_accs]
        bp = axes[1].boxplot(data_to_plot, labels=['Training', 'Validation'],
                             patch_artist=True, widths=0.6)
        colors = ['#2E86AB', '#A23B72']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)

        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title('Accuracy Distribution\n(Statistical Summary)',
                          fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        # Add statistics text
        stats_text = f"Training:    {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}\n"
        stats_text += f"Validation: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}\n"
        stats_text += f"Variability: {np.std(val_accs):.4f}"
        axes[1].text(0.5, 0.02, stats_text, transform=axes[1].transAxes,
                     fontsize=10, verticalalignment='bottom', ha='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.5))

        plt.tight_layout()
        plt.savefig('frequency_domain_stability_analysis.png', dpi=300, bbox_inches='tight')
        print("\nStability plots saved as 'frequency_domain_stability_analysis.png'")
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    SYNTHETIC_DATA_PATH = r"D:\Final Project\Fault_Detection"  # UPDATE THIS PATH
    NUM_TRIALS = 10  # 10 trials as per paper for stability analysis
    SAMPLING_RATE = 25000  # Hz - adjust based on your data
    MAX_FREQ = 500  # Hz - cut frequency to reduce computational cost

    print("=" * 70)
    print("1D CNN FAULT CLASSIFICATION - PAPER METHODOLOGY")
    print("=" * 70)
    print("Based on: Deep Learning for Fault Diagnosis")
    print("\nKey Features (as per paper):")
    print("  ✓ Frequency domain analysis (for interpretability)")
    print("  ✓ Z-score normalization")
    print("  ✓ Signal cutting to reduce computational cost")
    print("  ✓ 4 mechanical faults: Normal, Misalignment, Looseness, Unbalance")
    print("  ✓ Single CNN layer (reduce complexity)")
    print("  ✓ Training on synthetic data only")
    print("  ✓ 10 trials for stability assessment")
    print("=" * 70)

    # Initialize classifier
    classifier = SyntheticFaultClassifier(
        data_path=SYNTHETIC_DATA_PATH,
        val_ratio=0.1,  # 90% training, 10% validation
        sampling_rate=SAMPLING_RATE,
        max_freq=MAX_FREQ
    )

    if NUM_TRIALS == 1:
        # Single trial with detailed results
        print("\n" + "=" * 70)
        print("RUNNING SINGLE TRIAL (Quick Test)")
        print("=" * 70)

        # Get file paths and split
        file_paths, labels = classifier.get_file_paths_and_labels()
        paths_train, paths_val, labels_train, labels_val = \
            classifier.split_data_for_synthetic(file_paths, labels)

        # Train model
        classifier.train_model(paths_train, labels_train, paths_val, labels_val)

        # Evaluate on validation set
        accuracy, y_pred, cm = classifier.evaluate_model(paths_val, labels_val)
        classifier.plot_results(cm)

        # Save model
        classifier.model.save('frequency_domain_fault_classifier.keras')
        print("\n" + "=" * 70)
        print("Model saved as 'frequency_domain_fault_classifier.keras'")
        print("=" * 70)

    else:
        # Multiple trials for stability analysis (as per paper)
        print("\n" + "=" * 70)
        print("RUNNING STABILITY ANALYSIS (AS PER PAPER)")
        print("=" * 70)
        print("10 trials to assess:")
        print("  1. Robustness to network stochasticity")
        print("  2. Robustness to signal generation variations")
        print("=" * 70)

        train_accs, val_accs, mean_val, std_val = classifier.run_multiple_trials(NUM_TRIALS)

        # Train one final model and save it
        print("\n" + "=" * 70)
        print("TRAINING FINAL MODEL TO SAVE")
        print("=" * 70)

        file_paths, labels = classifier.get_file_paths_and_labels()
        paths_train, paths_val, labels_train, labels_val = \
            classifier.split_data_for_synthetic(file_paths, labels)
        classifier.train_model(paths_train, labels_train, paths_val, labels_val)

        # Evaluate final model
        accuracy, y_pred, cm = classifier.evaluate_model(paths_val, labels_val)
        classifier.plot_results(cm)

        classifier.model.save('frequency_domain_fault_classifier.keras')
        print("\n" + "=" * 70)
        print("Final model saved as 'frequency_domain_fault_classifier.keras'")
        print("=" * 70)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print("Methodology Applied:")
    print("  ✓ Frequency domain transformation (FFT)")
    print("  ✓ Z-score normalization")
    print(f"  ✓ Signal cut to {MAX_FREQ} Hz (reduced computational cost)")
    print("  ✓ Multi-channel processing (4 accelerometer channels)")
    print("  ✓ Simple architecture (1 CNN layer)")
    print("  ✓ Early stopping (patience=8, min_delta=0.001)")
    print("  ✓ Trained on SYNTHETIC data only")
    print("\nNext Steps:")
    print("  → Test model on REAL fault signals")
    print("  → Expected: Model generalizes from synthetic to real data")
    print("  → Interpretability: Frequency domain enables fault analysis")
    print("=" * 70)

    # Print system info
    print("\nSystem Information:")
    print(f"  TensorFlow version: {tf.__version__}")
    print(f"  GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print(f"  GPU device: {tf.config.list_physical_devices('GPU')[0].name}")
    print("=" * 70)