import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
from tensorflow.keras.utils import Sequence
import pandas as pd
from scipy.interpolate import make_interp_spline

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ============================================================================
# ATTENTION LAYER — must be registered so keras can load saved models
# ============================================================================
class AttentionLayer(layers.Layer):
    """
    Same AttentionLayer used during training.
    Must be provided as a custom object when loading .keras files.
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
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=1)
        if self.return_sequences:
            return x * a
        return keras.backend.sum(x * a, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'return_sequences': self.return_sequences})
        return config


# ============================================================================
# DATA GENERATOR  (max_freq=1000 to match attention training)
# ============================================================================
class FrequencyDomainDataGenerator(Sequence):
    """
    Identical preprocessing to training:
    FFT → positive frequencies → cut at max_freq → Z-score normalise.
    """

    def __init__(self, file_paths, labels, batch_size=32, shuffle=False,
                 sampling_rate=25000, max_freq=500):
        self.file_paths    = file_paths
        self.labels        = labels
        self.batch_size    = batch_size
        self.shuffle       = shuffle
        self.sampling_rate = sampling_rate
        self.max_freq      = max_freq
        self.indices       = np.arange(len(self.file_paths))
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
        return fft_mag[freq_mask]

    def z_score_normalize(self, signal):
        mean = np.mean(signal)
        std  = np.std(signal)
        if std == 0:
            return signal - mean
        return (signal - mean) / std

    def __getitem__(self, index):
        indices      = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths  = [self.file_paths[i] for i in indices]
        batch_labels = [self.labels[i]     for i in indices]
        actual_bs    = len(batch_paths)

        first_signal = np.load(batch_paths[0])
        if first_signal.shape[0] != 4:
            first_signal = first_signal.T

        freq_length = len(self.time_to_frequency(first_signal[0]))
        X = np.zeros((actual_bs, freq_length, 4), dtype=np.float32)

        for i, path in enumerate(batch_paths):
            signal = np.load(path)
            if signal.shape[0] != 4:
                signal = signal.T
            for ch in range(4):
                fs = self.time_to_frequency(signal[ch])
                fs = self.z_score_normalize(fs)
                X[i, :, ch] = fs.astype(np.float32)

        return X, np.array(batch_labels, dtype=np.int32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ============================================================================
# MAIN ANALYSER CLASS
# ============================================================================
class AttentionModelAnalyzer:
    """
    Loads every attention model (model_real_10.keras … model_real_100.keras),
    evaluates each on the full real test dataset, and produces:

    Plots
    -----
    01_overall_accuracy_vs_samples.png
    02_accuracy_improvement_analysis.png
    03_per_class_accuracy_trends.png
    04_confusion_matrices_grid.png
    05_per_class_detailed_metrics.png
    06_metrics_comparison_bars.png
    07_recommendation_summary.png

    CSV
    ---
    attention_results_summary.csv
    """

    def __init__(self, models_dir, real_data_path,
                 sampling_rate=25000, max_freq=500):
        """
        Args:
            models_dir      : Folder that contains model_real_10.keras, etc.
            real_data_path  : Root of the real test dataset
            sampling_rate   : Must match training (default 25000 Hz)
            max_freq        : Must match training (default 1000 Hz)
        """
        self.models_dir    = models_dir
        self.real_data_path = real_data_path
        self.sampling_rate = sampling_rate
        self.max_freq      = max_freq

        self.fault_classes = ['Normal', 'Misalignment', 'Looseness', 'Unbalance']
        self.class_to_idx  = {cls: idx for idx, cls in enumerate(self.fault_classes)}
        self.colors        = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        # Filled by evaluate_all_models()
        self.all_results         = {}
        self.optimal_sample_count = None
        self.optimal_accuracy     = None

        # Custom objects required to load models with AttentionLayer
        self._custom_objects = {'AttentionLayer': AttentionLayer}

    # ------------------------------------------------------------------
    # SCANNING
    # ------------------------------------------------------------------
    def get_model_paths(self, sample_counts):
        """Return dict {count: path} for every model that exists on disk."""
        model_paths = {}

        print("\n" + "=" * 70)
        print("SCANNING FOR ATTENTION MODELS")
        print("=" * 70)

        for count in sample_counts:
            name = f'model_real_{count}.keras'
            path = os.path.join(self.models_dir, name)
            if os.path.exists(path):
                model_paths[count] = path
                print(f"  ✓ Found : {name}")
            else:
                print(f"  ✗ Missing: {name}")

        print(f"\nTotal models found: {len(model_paths)}")
        print("=" * 70)
        return model_paths

    def get_real_data_paths(self):
        """
        Walk real_data_path/fault_class/**/*.npy and return
        (all_paths, all_labels).
        """
        print("\n" + "=" * 70)
        print("SCANNING REAL TEST DATA")
        print("=" * 70)
        print(f"Path: {self.real_data_path}")

        if not os.path.exists(self.real_data_path):
            raise FileNotFoundError(f"Real data path not found: {self.real_data_path}")

        all_paths  = []
        all_labels = []

        for fault_class in self.fault_classes:
            class_path = os.path.join(self.real_data_path, fault_class)
            if not os.path.exists(class_path):
                print(f"  Warning: {class_path} does not exist — skipping.")
                continue

            for root, _, files in os.walk(class_path):
                for f in files:
                    if f.endswith('.npy'):
                        all_paths.append(os.path.join(root, f))
                        all_labels.append(self.class_to_idx[fault_class])

            count = sum(1 for l in all_labels if l == self.class_to_idx[fault_class])
            print(f"  {fault_class}: {count} samples")

        if not all_paths:
            raise ValueError("No real test data found! Check dataset structure.")

        print(f"\nTotal test samples: {len(all_paths)}")

        unique, counts = np.unique(all_labels, return_counts=True)
        print("\nClass distribution:")
        for idx, cnt in zip(unique, counts):
            print(f"  {self.fault_classes[idx]}: {cnt} ({cnt/len(all_labels)*100:.1f}%)")

        print("=" * 70)
        return all_paths, all_labels

    # ------------------------------------------------------------------
    # SINGLE-MODEL EVALUATION
    # ------------------------------------------------------------------
    def evaluate_single_model(self, model_path, file_paths, labels, num_real_samples):
        """Load one model, run inference, return metrics dict."""
        print(f"\n  Evaluating model_real_{num_real_samples} ...", end="  ")

        # AttentionLayer must be passed as custom object
        model = keras.models.load_model(
            model_path, custom_objects=self._custom_objects
        )

        gen = FrequencyDomainDataGenerator(
            file_paths, labels,
            batch_size=32, shuffle=False,
            sampling_rate=self.sampling_rate,
            max_freq=self.max_freq
        )

        y_pred_probs = model.predict(gen, verbose=0)
        y_pred       = np.argmax(y_pred_probs, axis=1)
        y_true       = np.array(labels)

        accuracy      = accuracy_score(y_true, y_pred)
        cm            = confusion_matrix(y_true, y_pred)
        per_class_acc = np.diag(cm) / np.sum(cm, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        max_probs        = np.max(y_pred_probs, axis=1)
        mean_confidence  = float(np.mean(max_probs))

        print(f"Accuracy: {accuracy:.4f}")

        # Free GPU memory before next model
        del model
        tf.keras.backend.clear_session()

        return {
            'num_real_samples' : num_real_samples,
            'accuracy'         : accuracy,
            'per_class_accuracy': per_class_acc,
            'precision'        : precision,
            'recall'           : recall,
            'f1_score'         : f1,
            'mean_confidence'  : mean_confidence,
            'confusion_matrix' : cm,
            'y_pred'           : y_pred,
            'y_pred_probs'     : y_pred_probs,
        }

    # ------------------------------------------------------------------
    # EVALUATE ALL MODELS
    # ------------------------------------------------------------------
    def evaluate_all_models(self, sample_counts):
        """Iterate over every sample count, evaluate, store results."""
        print("\n" + "#" * 70)
        print("EVALUATING ALL ATTENTION MODELS ON REAL DATA")
        print("#" * 70)

        model_paths = self.get_model_paths(sample_counts)
        if not model_paths:
            raise ValueError("No models found — check models_dir.")

        file_paths, labels = self.get_real_data_paths()

        for count in sorted(model_paths.keys()):
            self.all_results[count] = self.evaluate_single_model(
                model_paths[count], file_paths, labels, count
            )

        print("\n" + "#" * 70)
        print("✓  ALL MODELS EVALUATED")
        print("#" * 70)
        return self.all_results

    # ------------------------------------------------------------------
    # FIND OPTIMAL SAMPLE COUNT
    # ------------------------------------------------------------------
    def find_optimal_samples(self, threshold=0.001):
        """
        Identify the point where adding more real samples yields
        less than `threshold` improvement in accuracy (diminishing returns).
        """
        sample_counts = sorted(self.all_results.keys())
        accuracies    = [self.all_results[c]['accuracy'] for c in sample_counts]

        improvements = [accuracies[i] - accuracies[i-1]
                        for i in range(1, len(accuracies))]

        optimal_idx = len(sample_counts) - 1        # default: last
        for i, imp in enumerate(improvements):
            if imp < threshold:
                optimal_idx = i
                break

        self.optimal_sample_count = sample_counts[optimal_idx]
        self.optimal_accuracy     = accuracies[optimal_idx]

        print("\n" + "=" * 70)
        print("OPTIMAL SAMPLE COUNT — DIMINISHING RETURNS ANALYSIS")
        print("=" * 70)
        print(f"Threshold              : {threshold}")
        print(f"Optimal sample count   : {self.optimal_sample_count}")
        print(f"Accuracy at optimal    : {self.optimal_accuracy:.4f} "
              f"({self.optimal_accuracy*100:.2f}%)")
        print(f"Maximum accuracy       : {max(accuracies):.4f} "
              f"({max(accuracies)*100:.2f}%)")
        print(f"Accuracy difference    : {(max(accuracies) - self.optimal_accuracy):.4f}")
        print(f"Sample efficiency      : {self.optimal_sample_count}/{sample_counts[-1]} "
              f"({self.optimal_sample_count/sample_counts[-1]*100:.0f}%)")
        print("=" * 70)
        return self.optimal_sample_count, self.optimal_accuracy

    # ══════════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════════

    def plot_01_overall_accuracy_vs_samples(self):
        """Smooth accuracy curve with optimal point highlighted."""
        fig, ax = plt.subplots(figsize=(16, 8))

        counts     = sorted(self.all_results.keys())
        accuracies = [self.all_results[c]['accuracy'] * 100 for c in counts]

        # Smooth spline
        if len(counts) > 3:
            x_smooth = np.linspace(min(counts), max(counts), 300)
            spl      = make_interp_spline(counts, accuracies, k=2)
            ax.plot(x_smooth, spl(x_smooth), '-', linewidth=3,
                    color='#2E86AB', alpha=0.5, label='Trend')

        ax.plot(counts, accuracies, 'o', markersize=14, color='#2E86AB',
                markeredgecolor='white', markeredgewidth=2,
                label='Measured Accuracy', zorder=5)

        # Optimal star
        if self.optimal_sample_count:
            opt_acc = self.optimal_accuracy * 100
            ax.plot(self.optimal_sample_count, opt_acc, '*', markersize=30,
                    color='gold', markeredgecolor='red', markeredgewidth=2,
                    label='Optimal Point', zorder=10)
            ax.annotate(
                f'OPTIMAL\n{self.optimal_sample_count} samples\n{opt_acc:.2f}%',
                xy=(self.optimal_sample_count, opt_acc),
                xytext=(35, 35), textcoords='offset points',
                fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='gold',
                          alpha=0.85, edgecolor='red', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2,
                                connectionstyle='arc3,rad=0.3')
            )

        # Value labels for non-optimal points
        for x, y in zip(counts, accuracies):
            if x != self.optimal_sample_count:
                ax.text(x, y + 0.5, f'{y:.2f}%', ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='#2E86AB')

        ax.set_xlabel('Number of Real Training Samples', fontsize=16,
                      fontweight='bold', labelpad=15)
        ax.set_ylabel('Test Accuracy (%)', fontsize=16,
                      fontweight='bold', labelpad=15)
        ax.set_title(
            'Attention Model — Overall Accuracy vs Real Training Samples\n'
            'Identifying the Optimal Training Data Amount',
            fontsize=18, fontweight='bold', pad=25
        )
        ax.set_xticks(counts)
        ax.set_ylim([min(accuracies) - 2, 102])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=13, loc='lower right', frameon=True,
                  shadow=True, fancybox=True)

        plt.tight_layout()
        plt.savefig('01_overall_accuracy_vs_samples.png', dpi=300,
                    bbox_inches='tight')
        print("✓ Saved: 01_overall_accuracy_vs_samples.png")
        plt.close()

    def plot_02_accuracy_improvement_analysis(self):
        """Incremental gain bars (left) + cumulative gain line (right)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        counts     = sorted(self.all_results.keys())
        accuracies = [self.all_results[c]['accuracy'] * 100 for c in counts]

        # ── Left: incremental improvement ────────────────────────────
        improvements = [0] + [accuracies[i] - accuracies[i-1]
                               for i in range(1, len(accuracies))]
        bar_colors = ['green' if v > 0 else 'red' for v in improvements]

        bars = ax1.bar(counts, improvements, color=bar_colors,
                       alpha=0.7, edgecolor='black', linewidth=2, width=5)

        for bar, imp, cnt in zip(bars, improvements, counts):
            if cnt != counts[0]:
                h = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2.,
                         h + 0.05 if h >= 0 else h - 0.05,
                         f'{imp:+.2f}%',
                         ha='center',
                         va='bottom' if h >= 0 else 'top',
                         fontsize=11, fontweight='bold')

        ax1.axhline(y=0.1, color='orange', linestyle='--', linewidth=2,
                    label='Diminishing Returns Threshold (0.1%)', alpha=0.8)
        ax1.axhline(y=0, color='black', linewidth=1)
        ax1.set_xlabel('Number of Real Training Samples', fontsize=14,
                       fontweight='bold')
        ax1.set_ylabel('Accuracy Improvement (%)', fontsize=14,
                       fontweight='bold')
        ax1.set_title('Incremental Accuracy Gain\n(vs Previous Step)',
                      fontsize=14, fontweight='bold', pad=15)
        ax1.set_xticks(counts)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.legend(fontsize=11)

        # ── Right: cumulative improvement ────────────────────────────
        baseline      = accuracies[0]
        cum_gains     = [acc - baseline for acc in accuracies]

        ax2.plot(counts, cum_gains, 'o-', linewidth=3, markersize=10,
                 color='#2E86AB', markeredgecolor='white', markeredgewidth=2)

        for x, y in zip(counts, cum_gains):
            ax2.text(x, y + 0.1, f'{y:.2f}%', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

        if self.optimal_sample_count:
            ax2.axvline(x=self.optimal_sample_count, color='red',
                        linestyle='--', linewidth=2, alpha=0.7,
                        label='Optimal Point')

        ax2.set_xlabel('Number of Real Training Samples', fontsize=14,
                       fontweight='bold')
        ax2.set_ylabel('Cumulative Accuracy Gain (%)', fontsize=14,
                       fontweight='bold')
        ax2.set_title(f'Total Improvement Over Baseline\n'
                      f'(Baseline = {counts[0]} samples → {baseline:.2f}%)',
                      fontsize=14, fontweight='bold', pad=15)
        ax2.set_xticks(counts)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=11)

        plt.tight_layout()
        plt.savefig('02_accuracy_improvement_analysis.png', dpi=300,
                    bbox_inches='tight')
        print("✓ Saved: 02_accuracy_improvement_analysis.png")
        plt.close()

    def plot_03_per_class_accuracy_trends(self):
        """One line per fault class showing how accuracy changes with samples."""
        fig, ax = plt.subplots(figsize=(16, 8))

        counts       = sorted(self.all_results.keys())
        class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

        for i, fault_class in enumerate(self.fault_classes):
            vals = [self.all_results[c]['per_class_accuracy'][i] * 100
                    for c in counts]
            ax.plot(counts, vals, 'o-', linewidth=3, markersize=10,
                    label=fault_class, color=class_colors[i],
                    markeredgecolor='white', markeredgewidth=2)
            # Label at the last point
            ax.text(counts[-1] + 1, vals[-1], f'{vals[-1]:.1f}%',
                    va='center', fontsize=11, fontweight='bold',
                    color=class_colors[i])

        if self.optimal_sample_count:
            ax.axvline(x=self.optimal_sample_count, color='red',
                       linestyle='--', linewidth=2, alpha=0.5,
                       label='Optimal Point')

        ax.set_xlabel('Number of Real Training Samples', fontsize=16,
                      fontweight='bold', labelpad=15)
        ax.set_ylabel('Class Accuracy (%)', fontsize=16,
                      fontweight='bold', labelpad=15)
        ax.set_title(
            'Per-Class Accuracy Trends — Attention Model\n'
            'How Each Fault Type Benefits from More Real Training Data',
            fontsize=18, fontweight='bold', pad=25
        )
        ax.set_xticks(counts)
        ax.set_ylim([0, 108])
        ax.legend(fontsize=13, loc='lower right', frameon=True,
                  shadow=True, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig('03_per_class_accuracy_trends.png', dpi=300,
                    bbox_inches='tight')
        print("✓ Saved: 03_per_class_accuracy_trends.png")
        plt.close()

    def plot_04_confusion_matrices_grid(self):
        """Grid of confusion matrices — one per model, optimal highlighted."""
        counts   = sorted(self.all_results.keys())
        n_models = len(counts)
        n_cols   = 3
        n_rows   = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        axes      = axes.flatten()

        for idx, count in enumerate(counts):
            cm  = self.all_results[count]['confusion_matrix']
            acc = self.all_results[count]['accuracy']

            is_optimal = (count == self.optimal_sample_count)
            cmap       = 'YlOrRd' if is_optimal else 'Blues'
            lw         = 3       if is_optimal else 0.5
            lc         = 'gold'  if is_optimal else 'gray'

            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                        xticklabels=self.fault_classes,
                        yticklabels=self.fault_classes,
                        ax=axes[idx], cbar=False,
                        linewidths=lw, linecolor=lc)

            axes[idx].set_xlabel('Predicted', fontsize=10, fontweight='bold')
            axes[idx].set_ylabel('True',      fontsize=10, fontweight='bold')

            prefix = '⭐ OPTIMAL: ' if is_optimal else ''
            axes[idx].set_title(
                f'{prefix}{count} Real Samples\nAcc: {acc:.4f} ({acc*100:.2f}%)',
                fontsize=12, fontweight='bold', pad=10
            )
            plt.setp(axes[idx].get_xticklabels(), rotation=45,
                     ha='right', fontsize=9)
            plt.setp(axes[idx].get_yticklabels(), rotation=0, fontsize=9)

        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(
            'Confusion Matrices — Attention Model (All Configurations)\n'
            'Gold border = optimal model',
            fontsize=16, fontweight='bold', y=0.995
        )
        plt.tight_layout()
        plt.savefig('04_confusion_matrices_grid.png', dpi=300,
                    bbox_inches='tight')
        print("✓ Saved: 04_confusion_matrices_grid.png")
        plt.close()

    def plot_05_per_class_detailed_metrics(self):
        """2×2 grid — one subplot per fault class, showing all 4 metrics."""
        counts = sorted(self.all_results.keys())
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()

        markers = ['o', 's', '^', 'D']
        labels  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

        for ci, fault_class in enumerate(self.fault_classes):
            ax = axes[ci]

            vals = {
                'Accuracy' : [self.all_results[c]['per_class_accuracy'][ci] * 100 for c in counts],
                'Precision': [self.all_results[c]['precision'][ci] * 100 for c in counts],
                'Recall'   : [self.all_results[c]['recall'][ci] * 100 for c in counts],
                'F1-Score' : [self.all_results[c]['f1_score'][ci] * 100 for c in counts],
            }

            for (metric, data), color, marker in zip(
                vals.items(), self.colors, markers
            ):
                ax.plot(counts, data, f'{marker}-', linewidth=2.5, markersize=8,
                        label=metric, color=color,
                        markeredgecolor='white', markeredgewidth=1.5)

            if self.optimal_sample_count:
                ax.axvline(x=self.optimal_sample_count, color='red',
                           linestyle='--', linewidth=2, alpha=0.5,
                           label='Optimal Point')

            ax.set_xlabel('Real Training Samples', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score (%)',              fontsize=12, fontweight='bold')
            ax.set_title(f'{fault_class} — Detailed Metrics',
                         fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks(counts)
            ax.set_ylim([0, 108])
            ax.legend(fontsize=10, frameon=True, shadow=True, ncol=2)
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.suptitle(
            'Per-Class Detailed Performance Metrics — Attention Model\n'
            'Dashed red line = optimal sample count',
            fontsize=16, fontweight='bold', y=0.995
        )
        plt.tight_layout()
        plt.savefig('05_per_class_detailed_metrics.png', dpi=300,
                    bbox_inches='tight')
        print("✓ Saved: 05_per_class_detailed_metrics.png")
        plt.close()

    def plot_06_metrics_comparison_bars(self):
        """Three bar charts: Macro Precision, Recall, F1 — optimal in gold."""
        counts = sorted(self.all_results.keys())
        bar_colors = ['gold' if c == self.optimal_sample_count
                      else '#2E86AB' for c in counts]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        x_pos = range(len(counts))

        metrics = [
            ('Macro Precision (%)', [self.all_results[c]['precision'].mean() * 100 for c in counts]),
            ('Macro Recall (%)',    [self.all_results[c]['recall'].mean()    * 100 for c in counts]),
            ('Macro F1-Score (%)',  [self.all_results[c]['f1_score'].mean()  * 100 for c in counts]),
        ]

        for ax, (ylabel, vals) in zip(axes, metrics):
            bars = ax.bar(x_pos, vals, color=bar_colors,
                          alpha=0.8, edgecolor='black', linewidth=2)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_title(ylabel.replace(' (%)', ' Comparison'),
                         fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(counts)
            ax.set_xlabel('Real Training Samples', fontsize=12,
                          fontweight='bold')
            ax.set_ylim([0, 108])
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        plt.suptitle(
            'Macro Metrics Comparison — Attention Model\n'
            'Gold bar = optimal model',
            fontsize=16, fontweight='bold', y=0.98
        )
        plt.tight_layout()
        plt.savefig('06_metrics_comparison_bars.png', dpi=300,
                    bbox_inches='tight')
        print("✓ Saved: 06_metrics_comparison_bars.png")
        plt.close()

    def plot_07_recommendation_summary(self):
        """Text card summarising the optimal-sample recommendation."""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')

        counts     = sorted(self.all_results.keys())
        accuracies = [self.all_results[c]['accuracy'] * 100 for c in counts]

        min_acc    = min(accuracies)
        max_acc    = max(accuracies)
        total_gain = max_acc - min_acc

        opt_acc     = self.optimal_accuracy * 100 if self.optimal_accuracy else max_acc
        acc_lost    = max_acc - opt_acc

        card = f"""
        ╔═══════════════════════════════════════════════════════════════╗
        ║                                                               ║
        ║      🎯 OPTIMAL TRAINING DATA RECOMMENDATION (ATTENTION) 🎯   ║
        ║                                                               ║
        ╚═══════════════════════════════════════════════════════════════╝

        ┌───────────────────────────────────────────────────────────────┐
        │                        KEY FINDINGS                           │
        └───────────────────────────────────────────────────────────────┘

        ✅ RECOMMENDED NUMBER OF REAL SAMPLES : {self.optimal_sample_count}

        📊 Expected Test Accuracy : {opt_acc:.2f}%

        💰 Sample Efficiency : {self.optimal_sample_count}/{counts[-1]} samples used
                              ({self.optimal_sample_count/counts[-1]*100:.0f}% of maximum)

        📉 Accuracy Trade-off : Only {acc_lost:.2f}% less than maximum

        ┌───────────────────────────────────────────────────────────────┐
        │                     PERFORMANCE RANGE                         │
        └───────────────────────────────────────────────────────────────┘

        Minimum  ({counts[0]:>3} samples)  →  {min_acc:.2f}%
        Optimal  ({self.optimal_sample_count:>3} samples)  →  {opt_acc:.2f}%
        Maximum  ({counts[-1]:>3} samples)  →  {max_acc:.2f}%

        Total Performance Gain :  {total_gain:.2f}%

        ┌───────────────────────────────────────────────────────────────┐
        │                      RECOMMENDATION                           │
        └───────────────────────────────────────────────────────────────┘
        """

        if acc_lost < 0.5:
            card += f"""
        ✅ STRONG: Use {self.optimal_sample_count} real samples.
           Negligible accuracy loss (<0.5%) — optimal efficiency.
        """
        elif acc_lost < 1.0:
            card += f"""
        ✅ RECOMMENDED: {self.optimal_sample_count} samples gives good efficiency.
           Small trade-off ({acc_lost:.2f}%) with significant data savings.
        """
        else:
            card += f"""
        ⚠️  CONSIDER: {self.optimal_sample_count} samples is efficient but
           loses {acc_lost:.2f}%.  Use more if max accuracy is critical.
        """

        card += """
        ═══════════════════════════════════════════════════════════════
        """

        ax.text(0.5, 0.5, card,
                transform=ax.transAxes,
                fontsize=13, va='center', ha='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='lightblue',
                          alpha=0.8, edgecolor='navy', linewidth=3))

        plt.tight_layout()
        plt.savefig('07_recommendation_summary.png', dpi=300,
                    bbox_inches='tight')
        print("✓ Saved: 07_recommendation_summary.png")
        plt.close()

    # ------------------------------------------------------------------
    # ORCHESTRATOR
    # ------------------------------------------------------------------
    def generate_all_plots(self):
        print("\n" + "=" * 70)
        print("GENERATING ALL VISUALISATIONS")
        print("=" * 70)

        self.plot_01_overall_accuracy_vs_samples()
        self.plot_02_accuracy_improvement_analysis()
        self.plot_03_per_class_accuracy_trends()
        self.plot_04_confusion_matrices_grid()
        self.plot_05_per_class_detailed_metrics()
        self.plot_06_metrics_comparison_bars()
        self.plot_07_recommendation_summary()

        print("\n" + "=" * 70)
        print("✓ ALL PLOTS GENERATED")
        print("=" * 70)
        print("Files saved:")
        for i in range(1, 8):
            names = [
                "01_overall_accuracy_vs_samples.png",
                "02_accuracy_improvement_analysis.png",
                "03_per_class_accuracy_trends.png",
                "04_confusion_matrices_grid.png",
                "05_per_class_detailed_metrics.png",
                "06_metrics_comparison_bars.png",
                "07_recommendation_summary.png",
            ]
            print(f"  {names[i-1]}")
        print("=" * 70)

    # ------------------------------------------------------------------
    # CSV + CONSOLE SUMMARY
    # ------------------------------------------------------------------
    def save_results_csv(self, filename='attention_results_summary.csv'):
        counts = sorted(self.all_results.keys())

        rows = []
        for c in counts:
            r = self.all_results[c]
            rows.append({
                'Real_Samples'    : c,
                'Accuracy_%'      : round(r['accuracy'] * 100, 2),
                'Normal_%'        : round(r['per_class_accuracy'][0] * 100, 2),
                'Misalignment_%'  : round(r['per_class_accuracy'][1] * 100, 2),
                'Looseness_%'     : round(r['per_class_accuracy'][2] * 100, 2),
                'Unbalance_%'     : round(r['per_class_accuracy'][3] * 100, 2),
                'Macro_Precision' : round(r['precision'].mean() * 100, 2),
                'Macro_Recall'    : round(r['recall'].mean() * 100, 2),
                'Macro_F1'        : round(r['f1_score'].mean() * 100, 2),
                'Mean_Confidence' : round(r['mean_confidence'], 4),
                'Is_Optimal'      : 'YES ⭐' if c == self.optimal_sample_count else 'No',
            })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY TABLE")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)
        print(f"\n✓ CSV saved: {filename}")

    def print_final_summary(self):
        counts     = sorted(self.all_results.keys())
        accuracies = [self.all_results[c]['accuracy'] for c in counts]

        best_count  = counts[int(np.argmax(accuracies))]
        worst_count = counts[int(np.argmin(accuracies))]

        print("\n" + "#" * 70)
        print("ANALYSIS COMPLETE — FINAL SUMMARY")
        print("#" * 70)
        print(f"\n{'Samples':>10} | {'Accuracy':>10} | {'Macro F1':>10} | {'Confidence':>12}")
        print("-" * 50)
        for c in counts:
            r   = self.all_results[c]
            tag = ' ⭐' if c == self.optimal_sample_count else \
                  ' 🏆' if c == best_count else ''
            print(f"{c:>10} | {r['accuracy']*100:>9.2f}% | "
                  f"{r['f1_score'].mean()*100:>9.2f}% | "
                  f"{r['mean_confidence']:>11.4f}{tag}")

        print(f"\n🎯 OPTIMAL : {self.optimal_sample_count} samples "
              f"→ {self.optimal_accuracy*100:.2f}% accuracy")
        print(f"🏆 BEST    : {best_count} samples "
              f"→ {self.all_results[best_count]['accuracy']*100:.2f}% accuracy")
        print("#" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":

    # ── Paths — UPDATE THESE ──────────────────────────────────────────
    MODELS_DIR      = r"C:\Users\HP\PycharmProjects\PythonProject\Attention Mechanism Varying Real Data\models_attention_varying_real"   # output dir from training
    REAL_DATA_PATH  = r"D:\Final Project\New folder\Real Dataset"     # UPDATE
    SAMPLE_COUNTS   = list(range(10, 101, 10))          # [10, 20, ..., 100]

    SAMPLING_RATE   = 25000   # Hz  — must match training
    MAX_FREQ        = 500    # Hz  — must match training

    print("=" * 70)
    print("ATTENTION MODEL ANALYSIS — VARYING REAL DATA")
    print("=" * 70)
    print(f"  Models dir   : {MODELS_DIR}")
    print(f"  Real data    : {REAL_DATA_PATH}")
    print(f"  Sample counts: {SAMPLE_COUNTS}")
    print(f"  Sampling rate: {SAMPLING_RATE} Hz")
    print(f"  Max frequency: {MAX_FREQ} Hz")
    print("=" * 70)

    # Initialise
    analyzer = AttentionModelAnalyzer(
        models_dir     = MODELS_DIR,
        real_data_path = REAL_DATA_PATH,
        sampling_rate  = SAMPLING_RATE,
        max_freq       = MAX_FREQ
    )

    # 1. Evaluate all models
    analyzer.evaluate_all_models(SAMPLE_COUNTS)

    # 2. Find optimal sample count
    analyzer.find_optimal_samples(threshold=0.001)

    # 3. Generate all 7 plots
    analyzer.generate_all_plots()

    # 4. Save CSV
    analyzer.save_results_csv('attention_results_summary.csv')

    # 5. Print console summary table
    analyzer.print_final_summary()

    print("\n✅ Done.  Check the 7 PNG files and attention_results_summary.csv")