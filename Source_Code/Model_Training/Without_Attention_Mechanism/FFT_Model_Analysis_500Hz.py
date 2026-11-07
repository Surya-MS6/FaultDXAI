import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
from tensorflow.keras.utils import Sequence

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class RealDataGenerator(Sequence):
    """
    Data generator for REAL signals with frequency domain transformation
    Uses same preprocessing as training: FFT + Z-score + frequency cutting
    """

    def __init__(self, file_paths, labels, batch_size=32, shuffle=False,
                 sampling_rate=25000, max_freq=500):
        """
        Args:
            file_paths: List of file paths to REAL data
            labels: Corresponding labels
            batch_size: Batch size
            shuffle: Whether to shuffle (False for test data)
            sampling_rate: Sampling rate (Hz)
            max_freq: Maximum frequency to consider (Hz)
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
        """Convert time domain to frequency domain using FFT"""
        fft_vals = fft(time_signal)
        fft_mag = np.abs(fft_vals)
        freqs = fftfreq(len(time_signal), 1 / self.sampling_rate)

        # Take only positive frequencies
        positive_freqs = freqs[:len(freqs) // 2]
        fft_mag = fft_mag[:len(fft_mag) // 2]

        # Cut to max_freq
        freq_mask = positive_freqs <= self.max_freq
        fft_mag_cut = fft_mag[freq_mask]

        return fft_mag_cut

    def z_score_normalize(self, signal):
        """Z-score normalization"""
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

        # Load first signal to get dimensions
        first_signal = np.load(batch_paths[0])
        if first_signal.shape[0] != 4:
            first_signal = first_signal.T

        num_channels = 4
        channel_0_freq = self.time_to_frequency(first_signal[0])
        freq_length = len(channel_0_freq)

        # Pre-allocate
        X = np.zeros((actual_batch_size, freq_length, num_channels), dtype=np.float32)

        # Process all samples
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


class RealDataAnalyzer:
    """
    Analyzer for testing trained model on REAL data
    Evaluates generalization from synthetic to real signals
    """

    def __init__(self, model_path, real_data_path, sampling_rate=25000, max_freq=500):
        """
        Args:
            model_path: Path to trained model (.keras file)
            real_data_path: Path to REAL data folder
            sampling_rate: Sampling rate (Hz)
            max_freq: Maximum frequency to consider (Hz)
        """
        self.model_path = model_path
        self.real_data_path = real_data_path
        self.sampling_rate = sampling_rate
        self.max_freq = max_freq
        self.fault_classes = ['Normal', 'Misalignment', 'Looseness', 'Unbalance']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.fault_classes)}
        self.model = None
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    def load_model(self):
        """Load the trained model"""
        print("\n" + "=" * 70)
        print("LOADING TRAINED MODEL")
        print("=" * 70)
        print(f"Model path: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = keras.models.load_model(self.model_path)
        print("✓ Model loaded successfully!")
        print("\nModel Architecture:")
        self.model.summary()
        print("=" * 70)

    def get_real_data_paths(self):
        """
        Get file paths and labels for REAL data
        Expected structure: fault_class/test_folder/*.npy
        """
        print("\n" + "=" * 70)
        print("SCANNING REAL DATA")
        print("=" * 70)
        print(f"Real data path: {self.real_data_path}")

        if not os.path.exists(self.real_data_path):
            raise FileNotFoundError(f"Real data path does not exist: {self.real_data_path}")

        all_paths = []
        all_labels = []

        for fault_class in self.fault_classes:
            class_path = os.path.join(self.real_data_path, fault_class)

            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist, skipping...")
                continue

            # Get all test folders
            test_folders = sorted([f for f in os.listdir(class_path)
                                   if os.path.isdir(os.path.join(class_path, f))])

            print(f"\n{fault_class}: Found {len(test_folders)} test folders")

            for test_folder in test_folders:
                test_path = os.path.join(class_path, test_folder)

                # Load all .npy files directly from test folder
                npy_files = sorted([f for f in os.listdir(test_path) if f.endswith('.npy')])

                for npy_file in npy_files:
                    file_path = os.path.join(test_path, npy_file)
                    all_paths.append(file_path)
                    all_labels.append(self.class_to_idx[fault_class])

            count = len([l for l in all_labels if l == self.class_to_idx[fault_class]])
            print(f"  Total {fault_class} samples: {count}")

        if len(all_paths) == 0:
            raise ValueError("No real data files found! Please check your dataset structure.")

        # Check sample signal
        sample_signal = np.load(all_paths[0])
        print(f"\nFirst sample shape (time domain): {sample_signal.shape}")

        # Check class distribution
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"\nCLASS DISTRIBUTION (Real Data):")
        for cls_idx, count in zip(unique, counts):
            cls_name = self.fault_classes[cls_idx]
            print(f"  {cls_name}: {count} samples ({count / len(all_labels) * 100:.1f}%)")

        print(f"\nTotal REAL samples: {len(all_paths)}")
        print("=" * 70)

        return all_paths, all_labels

    def evaluate_on_real_data(self, file_paths, labels):
        """
        Evaluate model on REAL data
        This tests generalization from synthetic to real signals
        """
        print("\n" + "=" * 70)
        print("EVALUATING ON REAL DATA")
        print("=" * 70)
        print("Testing generalization: Synthetic Training → Real Testing")
        print("=" * 70)

        # Create generator for real data
        real_gen = RealDataGenerator(
            file_paths, labels,
            batch_size=32,
            shuffle=False,
            sampling_rate=self.sampling_rate,
            max_freq=self.max_freq
        )

        # Get predictions
        print("\nGenerating predictions...")
        y_pred_probs = self.model.predict(real_gen, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.array(labels)

        # Calculate overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n" + "=" * 70)
        print(f"OVERALL ACCURACY ON REAL DATA: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("=" * 70)

        # Classification report
        print("\nDETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_true, y_pred,
                                    target_names=self.fault_classes,
                                    digits=4))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nCONFUSION MATRIX:")
        print(cm)
        print("\nConfusion Matrix Breakdown:")
        for i, fault_class in enumerate(self.fault_classes):
            total = np.sum(cm[i, :])
            correct = cm[i, i]
            print(f"  {fault_class}: {correct}/{total} correct ({correct / total * 100:.2f}%)")

        return accuracy, y_pred, y_pred_probs, cm

    def analyze_per_class_confidence(self, y_true, y_pred_probs):
        """Analyze prediction confidence per class"""
        print("\n" + "=" * 70)
        print("PREDICTION CONFIDENCE ANALYSIS")
        print("=" * 70)

        for i, fault_class in enumerate(self.fault_classes):
            # Get samples of this class
            class_mask = y_true == i
            if np.sum(class_mask) == 0:
                continue

            class_probs = y_pred_probs[class_mask]

            # Max probability (confidence) for correct prediction
            max_probs = np.max(class_probs, axis=1)

            print(f"\n{fault_class}:")
            print(f"  Mean confidence: {np.mean(max_probs):.4f}")
            print(f"  Std confidence: {np.std(max_probs):.4f}")
            print(f"  Min confidence: {np.min(max_probs):.4f}")
            print(f"  Max confidence: {np.max(max_probs):.4f}")

            # Predictions for this class
            predicted_class = np.argmax(class_probs, axis=1)
            correct = np.sum(predicted_class == i)
            total = len(predicted_class)
            print(f"  Accuracy: {correct}/{total} ({correct / total * 100:.2f}%)")

    def analyze_misclassifications(self, file_paths, y_true, y_pred):
        """Analyze misclassified samples"""
        print("\n" + "=" * 70)
        print("MISCLASSIFICATION ANALYSIS")
        print("=" * 70)

        misclassified_indices = np.where(y_true != y_pred)[0]
        num_misclassified = len(misclassified_indices)

        print(f"\nTotal misclassified: {num_misclassified} / {len(y_true)}")
        print(f"Misclassification rate: {num_misclassified / len(y_true) * 100:.2f}%")

        if num_misclassified > 0:
            print("\nMisclassified samples breakdown:")
            for idx in misclassified_indices[:10]:  # Show first 10
                true_label = self.fault_classes[y_true[idx]]
                pred_label = self.fault_classes[y_pred[idx]]
                file_name = os.path.basename(file_paths[idx])
                print(f"  {file_name}: True={true_label}, Predicted={pred_label}")

            if num_misclassified > 10:
                print(f"  ... and {num_misclassified - 10} more")

    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.fault_classes,
                    yticklabels=self.fault_classes,
                    ax=ax, cbar_kws={'label': 'Number of Samples'},
                    linewidths=0.5, linecolor='gray')

        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Confusion Matrix - Real Data Evaluation\nSynthetic Training → Real Testing',
                     fontsize=16, fontweight='bold', pad=20)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('01_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 01_confusion_matrix.png")
        plt.close()

    def plot_normalized_confusion_matrix(self, cm):
        """Plot normalized confusion matrix (recall)"""
        fig, ax = plt.subplots(figsize=(10, 8))

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=self.fault_classes,
                    yticklabels=self.fault_classes,
                    ax=ax, cbar_kws={'label': 'Recall (True Positive Rate)'},
                    vmin=0, vmax=1, linewidths=0.5, linecolor='gray')

        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Normalized Confusion Matrix (Recall per Class)\nReal Data Evaluation',
                     fontsize=16, fontweight='bold', pad=20)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('02_normalized_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 02_normalized_confusion_matrix.png")
        plt.close()

    def plot_per_class_accuracy(self, cm):
        """Plot per-class accuracy bar chart"""
        fig, ax = plt.subplots(figsize=(12, 7))

        per_class_acc = np.diag(cm) / np.sum(cm, axis=1) * 100

        bars = ax.bar(self.fault_classes, per_class_acc, color=self.colors,
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, acc in zip(bars, per_class_acc):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_xlabel('Fault Class', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Per-Class Accuracy on Real Data\nModel Generalization Performance',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim([0, 110])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('03_per_class_accuracy.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 03_per_class_accuracy.png")
        plt.close()

    def plot_confidence_distribution(self, y_true, y_pred_probs):
        """Plot confidence distribution per class"""
        fig, ax = plt.subplots(figsize=(14, 7))

        for i, fault_class in enumerate(self.fault_classes):
            class_mask = y_true == i
            if np.sum(class_mask) == 0:
                continue

            class_probs = y_pred_probs[class_mask]
            max_probs = np.max(class_probs, axis=1)

            ax.hist(max_probs, bins=30, alpha=0.6, label=fault_class,
                    color=self.colors[i], edgecolor='black', linewidth=1)

        ax.set_xlabel('Prediction Confidence', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Frequency (Number of Samples)', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Prediction Confidence Distribution by Class\nReal Data Evaluation',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig('04_confidence_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 04_confidence_distribution.png")
        plt.close()

    def plot_metrics_comparison(self, y_true, y_pred):
        """Plot precision, recall, F1-score comparison"""
        from sklearn.metrics import precision_recall_fscore_support

        fig, ax = plt.subplots(figsize=(14, 7))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        x = np.arange(len(self.fault_classes))
        width = 0.25

        bars1 = ax.bar(x - width, precision, width, label='Precision',
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x, recall, width, label='Recall',
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score',
                       alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('Fault Class', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Score', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Classification Metrics Comparison per Class\nReal Data Evaluation',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.fault_classes, rotation=45, ha='right')
        ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
        ax.set_ylim([0, 1.15])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig('05_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 05_metrics_comparison.png")
        plt.close()

    def plot_confidence_boxplot(self, y_true, y_pred_probs):
        """Plot confidence box plot"""
        fig, ax = plt.subplots(figsize=(12, 7))

        confidence_data = []
        labels_list = []

        for i, fault_class in enumerate(self.fault_classes):
            class_mask = y_true == i
            if np.sum(class_mask) > 0:
                class_probs = y_pred_probs[class_mask]
                max_probs = np.max(class_probs, axis=1)
                confidence_data.append(max_probs)
                labels_list.append(fault_class)

        bp = ax.boxplot(confidence_data, tick_labels=labels_list, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='blue', linewidth=2, linestyle='--'))

        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Prediction Confidence', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_xlabel('Fault Class', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Confidence Distribution Box Plot by Class\nReal Data Evaluation',
                     fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Median'),
            Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=11, frameon=True, shadow=True)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('06_confidence_boxplot.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 06_confidence_boxplot.png")
        plt.close()

    def plot_class_distribution(self, y_true):
        """Plot class distribution pie and bar chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        unique, counts = np.unique(y_true, return_counts=True)
        class_names = [self.fault_classes[i] for i in unique]

        # Pie chart
        wedges, texts, autotexts = ax1.pie(counts, labels=class_names, autopct='%1.1f%%',
                                           colors=self.colors, startangle=90,
                                           textprops={'fontsize': 12, 'fontweight': 'bold'},
                                           explode=[0.05] * len(counts))

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)

        ax1.set_title('Class Distribution in Real Test Data\n(Pie Chart)',
                      fontsize=14, fontweight='bold', pad=20)

        # Bar chart
        bars = ax2.bar(class_names, counts, color=self.colors, alpha=0.8,
                       edgecolor='black', linewidth=1.5)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                     f'{count}\n({count / sum(counts) * 100:.1f}%)',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2.set_ylabel('Number of Samples', fontsize=13, fontweight='bold', labelpad=10)
        ax2.set_xlabel('Fault Class', fontsize=13, fontweight='bold', labelpad=10)
        ax2.set_title('Class Distribution in Real Test Data\n(Bar Chart)',
                      fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_axisbelow(True)

        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('07_class_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 07_class_distribution.png")
        plt.close()

    def plot_roc_curves(self, y_true, y_pred_probs):
        """Plot ROC curves for each class"""
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        fig, ax = plt.subplots(figsize=(12, 8))

        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=range(len(self.fault_classes)))

        for i, (fault_class, color) in enumerate(zip(self.fault_classes, self.colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=color, lw=2.5, alpha=0.8,
                    label=f'{fault_class} (AUC = {roc_auc:.4f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('ROC Curves for Multi-Class Classification\nReal Data Evaluation',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig('08_roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 08_roc_curves.png")
        plt.close()

    def plot_prediction_heatmap(self, y_true, y_pred_probs):
        """Plot prediction probability heatmap"""
        fig, ax = plt.subplots(figsize=(14, 10))

        # Sort by true label for better visualization
        sort_idx = np.argsort(y_true)
        y_pred_probs_sorted = y_pred_probs[sort_idx]

        # Limit to first 200 samples for readability
        max_samples = min(200, len(y_pred_probs_sorted))

        im = ax.imshow(y_pred_probs_sorted[:max_samples].T, aspect='auto',
                       cmap='YlOrRd', interpolation='nearest')

        ax.set_yticks(range(len(self.fault_classes)))
        ax.set_yticklabels(self.fault_classes, fontsize=12)
        ax.set_xlabel('Sample Index (sorted by true label)', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel('Predicted Class Probability', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_title(f'Prediction Probability Heatmap\nFirst {max_samples} Samples (Real Data)',
                     fontsize=15, fontweight='bold', pad=20)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig('09_prediction_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 09_prediction_heatmap.png")
        plt.close()

    def plot_confidence_statistics(self, y_true, y_pred_probs):
        """Plot confidence statistics comparison"""
        fig, ax = plt.subplots(figsize=(14, 7))

        stats_data = {'Mean': [], 'Median': [], 'Std': []}

        for i in range(len(self.fault_classes)):
            class_mask = y_true == i
            if np.sum(class_mask) > 0:
                class_probs = y_pred_probs[class_mask]
                max_probs = np.max(class_probs, axis=1)

                stats_data['Mean'].append(np.mean(max_probs))
                stats_data['Median'].append(np.median(max_probs))
                stats_data['Std'].append(np.std(max_probs))

        x = np.arange(len(self.fault_classes))
        width = 0.25

        bars1 = ax.bar(x - width, stats_data['Mean'], width, label='Mean',
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x, stats_data['Median'], width, label='Median',
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        bars3 = ax.bar(x + width, stats_data['Std'], width, label='Std Dev',
                       alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('Fault Class', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Confidence Value', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title('Confidence Statistics by Class\nReal Data Evaluation',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.fault_classes, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
        ax.set_ylim([0, max(max(stats_data['Mean']), max(stats_data['Median'])) * 1.2])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig('10_confidence_statistics.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 10_confidence_statistics.png")
        plt.close()

    def plot_performance_summary(self, cm, y_true, y_pred_probs, overall_acc):
        """Plot comprehensive performance summary"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Overall metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, np.argmax(y_pred_probs, axis=1), average=None
        )

        metrics_df = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Accuracy': np.diag(cm) / np.sum(cm, axis=1)
        }

        x = np.arange(len(self.fault_classes))
        width = 0.2

        for idx, (metric_name, values) in enumerate(metrics_df.items()):
            offset = (idx - 1.5) * width
            bars = ax1.bar(x + offset, values, width, label=metric_name,
                           alpha=0.8, edgecolor='black', linewidth=1)

        ax1.set_xlabel('Fault Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('All Classification Metrics Comparison', fontsize=13, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.fault_classes, rotation=45, ha='right')
        ax1.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
        ax1.set_ylim([0, 1.15])
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_axisbelow(True)

        # 2. Sample count per class
        unique, counts = np.unique(y_true, return_counts=True)
        bars = ax2.bar([self.fault_classes[i] for i in unique], counts,
                       color=self.colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + max(counts) * 0.01,
                     f'{count}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Fault Class', fontsize=12, fontweight='bold')
        ax2.set_title('Test Set Distribution', fontsize=13, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_axisbelow(True)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. Correct vs Incorrect predictions
        correct_counts = np.diag(cm)
        incorrect_counts = np.sum(cm, axis=1) - correct_counts

        x = np.arange(len(self.fault_classes))
        width = 0.35

        bars1 = ax3.bar(x - width / 2, correct_counts, width, label='Correct',
                        color='green', alpha=0.7, edgecolor='black', linewidth=1.5)
        bars2 = ax3.bar(x + width / 2, incorrect_counts, width, label='Incorrect',
                        color='red', alpha=0.7, edgecolor='black', linewidth=1.5)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                             f'{int(height)}',
                             ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax3.set_xlabel('Fault Class', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
        ax3.set_title('Correct vs Incorrect Predictions', fontsize=13, fontweight='bold', pad=15)
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.fault_classes, rotation=45, ha='right')
        ax3.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax3.set_axisbelow(True)

        # 4. Summary text
        ax4.axis('off')

        # Calculate additional metrics
        total_samples = len(y_true)
        mean_confidence = np.mean(np.max(y_pred_probs, axis=1))
        std_confidence = np.std(np.max(y_pred_probs, axis=1))

        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        summary_text = f"""
        PERFORMANCE SUMMARY - REAL DATA EVALUATION
        {'=' * 55}

        Overall Metrics:
        • Overall Accuracy:        {overall_acc:.4f} ({overall_acc * 100:.2f}%)
        • Macro Precision:         {macro_precision:.4f}
        • Macro Recall:            {macro_recall:.4f}
        • Macro F1-Score:          {macro_f1:.4f}

        Confidence Statistics:
        • Mean Confidence:         {mean_confidence:.4f}
        • Std Confidence:          {std_confidence:.4f}

        Dataset Information:
        • Total Test Samples:      {total_samples}
        • Number of Classes:       {len(self.fault_classes)}
        • Correct Predictions:     {np.sum(np.diag(cm))}
        • Incorrect Predictions:   {total_samples - np.sum(np.diag(cm))}

        Per-Class Performance:
        """

        for i, fault in enumerate(self.fault_classes):
            acc = np.diag(cm)[i] / np.sum(cm[i, :])
            summary_text += f"\n        • {fault:15s}: {acc:.4f} ({acc * 100:.2f}%)"

        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        plt.savefig('11_performance_summary.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: 11_performance_summary.png")
        plt.close()

    def generate_all_plots(self, cm, y_true, y_pred, y_pred_probs, overall_acc):
        """Generate all individual plots"""
        print("\n" + "=" * 70)
        print("GENERATING INDIVIDUAL PLOTS")
        print("=" * 70)

        self.plot_confusion_matrix(cm)
        self.plot_normalized_confusion_matrix(cm)
        self.plot_per_class_accuracy(cm)
        self.plot_confidence_distribution(y_true, y_pred_probs)
        self.plot_metrics_comparison(y_true, y_pred)
        self.plot_confidence_boxplot(y_true, y_pred_probs)
        self.plot_class_distribution(y_true)
        self.plot_roc_curves(y_true, y_pred_probs)
        self.plot_prediction_heatmap(y_true, y_pred_probs)
        self.plot_confidence_statistics(y_true, y_pred_probs)
        self.plot_performance_summary(cm, y_true, y_pred_probs, overall_acc)

        print("\n" + "=" * 70)
        print("✓ ALL PLOTS GENERATED SUCCESSFULLY")
        print("=" * 70)
        print("\nGenerated files:")
        print("  01_confusion_matrix.png")
        print("  02_normalized_confusion_matrix.png")
        print("  03_per_class_accuracy.png")
        print("  04_confidence_distribution.png")
        print("  05_metrics_comparison.png")
        print("  06_confidence_boxplot.png")
        print("  07_class_distribution.png")
        print("  08_roc_curves.png")
        print("  09_prediction_heatmap.png")
        print("  10_confidence_statistics.png")
        print("  11_performance_summary.png")
        print("=" * 70)

    def compare_synthetic_vs_real(self, synthetic_acc, real_acc):
        """Compare performance on synthetic vs real data"""
        print("\n" + "=" * 70)
        print("SYNTHETIC vs REAL DATA COMPARISON")
        print("=" * 70)
        print(f"Synthetic Data Accuracy: {synthetic_acc:.4f} ({synthetic_acc * 100:.2f}%)")
        print(f"Real Data Accuracy:      {real_acc:.4f} ({real_acc * 100:.2f}%)")
        print(f"Performance Drop:        {(synthetic_acc - real_acc):.4f} ({(synthetic_acc - real_acc) * 100:.2f}%)")

        if real_acc >= 0.95:
            print("\n✓ EXCELLENT: Model generalizes very well to real data!")
        elif real_acc >= 0.90:
            print("\n✓ GOOD: Model shows good generalization to real data")
        elif real_acc >= 0.80:
            print("\n⚠ MODERATE: Some generalization, but room for improvement")
        else:
            print("\n✗ POOR: Model struggles to generalize to real data")

        print("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = 'frequency_domain_fault_classifier.keras'
    REAL_DATA_PATH = r"D:\Final Project\New folder\Real Dataset"  # UPDATE THIS PATH
    SAMPLING_RATE = 25000  # Hz
    MAX_FREQ = 500  # Hz
    SYNTHETIC_ACCURACY = 0.9997  # From training output

    print("=" * 70)
    print("REAL DATA TESTING AND ANALYSIS")
    print("=" * 70)
    print("Testing model generalization: Synthetic → Real")
    print("\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Real data: {REAL_DATA_PATH}")
    print(f"  Sampling rate: {SAMPLING_RATE} Hz")
    print(f"  Max frequency: {MAX_FREQ} Hz")
    print("=" * 70)

    # Initialize analyzer
    analyzer = RealDataAnalyzer(
        model_path=MODEL_PATH,
        real_data_path=REAL_DATA_PATH,
        sampling_rate=SAMPLING_RATE,
        max_freq=MAX_FREQ
    )

    # Load trained model
    analyzer.load_model()

    # Get real data
    file_paths, labels = analyzer.get_real_data_paths()

    # Evaluate on real data
    accuracy, y_pred, y_pred_probs, cm = analyzer.evaluate_on_real_data(
        file_paths, labels
    )

    # Additional analyses
    analyzer.analyze_per_class_confidence(np.array(labels), y_pred_probs)
    analyzer.analyze_misclassifications(file_paths, np.array(labels), y_pred)

    # Generate all individual plots
    analyzer.generate_all_plots(cm, np.array(labels), y_pred, y_pred_probs, accuracy)

    # Compare with synthetic results
    analyzer.compare_synthetic_vs_real(SYNTHETIC_ACCURACY, accuracy)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("Key Findings:")
    print(f"  ✓ Tested on {len(file_paths)} real samples")
    print(f"  ✓ Overall accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  ✓ 11 comprehensive visualizations generated")
    print("\nInterpretation:")
    print("  - High accuracy indicates good generalization")
    print("  - Confusion matrix shows class-specific performance")
    print("  - Confidence analysis reveals prediction certainty")
    print("  - ROC curves demonstrate classification capability")
    print("=" * 70)