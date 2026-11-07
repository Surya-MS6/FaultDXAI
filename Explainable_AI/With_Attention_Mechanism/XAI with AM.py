"""
GRAD-CAM 1.XAI without AM Visualization — 1D CNN WITH Attention Mechanism
Shows which frequency regions the model focuses on for each fault class

Requirements: pip install numpy scipy matplotlib tensorflow
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import fft, fftfreq
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)

# ============================================================================
# ✅ CONFIGURATION
# ============================================================================
MODEL_PATH = r"C:\Users\HP\PycharmProjects\PythonProject\XAI AM\checkpoint_real_30.keras"   # or improved_attention_classifier.keras

# ✅ PUT YOUR CHOSEN .npy FILE PATHS HERE
CUSTOM_SAMPLES = {
        "Normal":       r"D:\Final Project\New folder\Real Dataset\Normal\Test 17_Normal Condition\20220506-142317.npy",
        "Misalignment": r"D:\Final Project\New folder\Real Dataset\Misalignment\Test 13_Misalignment\20220506-124407.npy",
        "Looseness":    r"D:\Final Project\New folder\Real Dataset\Looseness\Test 08_Looseness\20220429-101809.npy",
        "Unbalance":    r"D:\Final Project\New folder\Real Dataset\Unbalance\Test 18_Unbalance\20220506-143510.npy",
    }

SAMPLING_RATE = 25000
MAX_FREQ      = 500
FAULT_NAMES   = ['Normal', 'Misalignment', 'Looseness', 'Unbalance']
COLORS = {
    'Normal':       '#00FF99',
    'Misalignment': '#00BFFF',
    'Looseness':    '#FF69B4',
    'Unbalance':    '#FFD700',
}

# ============================================================================
# STYLE
# ============================================================================
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'figure.facecolor':   '#0A0A0F',
    'axes.facecolor':     '#12141E',
    'axes.labelcolor':    '#F0F0F0',
    'axes.titlecolor':    '#FFFFFF',
    'axes.edgecolor':     '#444455',
    'xtick.color':        '#C0C0C0',
    'ytick.color':        '#C0C0C0',
    'text.color':         '#F0F0F0',
    'grid.color':         '#2E3050',
    'grid.alpha':          0.6,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   True,
    'axes.spines.bottom': True,
})

# ============================================================================
# CUSTOM ATTENTION LAYER
# ============================================================================
class AttentionLayer(keras.layers.Layer):
    def __init__(self, return_sequences=True, **kwargs):
        super().__init__(**kwargs)
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
        super().build(input_shape)

    def call(self, x):
        e = tf.math.tanh(tf.linalg.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        if self.return_sequences:
            return x * a
        else:
            return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({'return_sequences': self.return_sequences})
        return config


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================
def time_to_freq(signal_1ch):
    N        = len(signal_1ch)
    fft_vals = np.abs(fft(signal_1ch))[:N // 2]
    freqs    = fftfreq(N, 1 / SAMPLING_RATE)[:N // 2]
    mask     = freqs <= MAX_FREQ
    return freqs[mask], fft_vals[mask]

def z_score(sig):
    std = np.std(sig)
    return (sig - np.mean(sig)) / (std if std != 0 else 1)

def prepare_input(signal_4ch):
    channels = []
    for ch in range(4):
        _, fft_mag = time_to_freq(signal_4ch[ch])
        channels.append(z_score(fft_mag))
    X = np.stack(channels, axis=-1)
    return X[np.newaxis, :, :].astype(np.float32)


# ============================================================================
# GRAD-CAM FOR 1D CNN
# ============================================================================
def compute_gradcam(model, X, class_idx):
    """
    Compute Grad-CAM heatmap for 1D signal.
    """
    conv_layer = model.get_layer('conv1d_layer')
    grad_model = keras.Model(inputs=model.inputs,
                             outputs=[conv_layer.output, model.outputs[0]])

    inputs = tf.cast(X, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs, training=False)
        loss = predictions[:, class_idx]

    # Gradient of loss w.r.t. INPUT, then project onto conv output shape
    input_grads = tape.gradient(loss, inputs)  # (1, freq_bins, 4)

    if input_grads is None:
        print("  [WARNING] Grad-CAM grads are None. Falling back to conv activation.")
        # Fallback: just use raw conv activation magnitude as heatmap
        heatmap = tf.reduce_mean(tf.abs(conv_outputs[0]), axis=-1).numpy()
    else:
        # Use input gradients averaged over channels as importance signal
        heatmap = tf.reduce_mean(tf.abs(input_grads[0]), axis=-1).numpy()  # (freq_bins,)

    # Also blend with conv activation for richer signal
    conv_act = tf.reduce_mean(tf.abs(conv_outputs[0]), axis=-1).numpy()  # (freq_bins,)

    # Resize conv_act to match heatmap length if needed
    if conv_act.shape[0] != heatmap.shape[0]:
        conv_act = np.interp(
            np.linspace(0, 1, heatmap.shape[0]),
            np.linspace(0, 1, conv_act.shape[0]),
            conv_act
        )

    # Blend: 60% gradient signal + 40% activation
    heatmap = 0.6 * heatmap + 0.4 * conv_act

    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


# ============================================================================
# LOAD MODEL & DATA
# ============================================================================
def load_model_and_data():
    print("=" * 60)
    print("LOADING MODEL (WITH ATTENTION)")
    print("=" * 60)
    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    print(f"  ✓ {MODEL_PATH}")
    print(f"  Input:  {model.input_shape}")
    print(f"  Output: {model.output_shape}")

    print("\n" + "=" * 60)
    print("LOADING CUSTOM FILES")
    print("=" * 60)
    samples = {}
    for fault, path in CUSTOM_SAMPLES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        data = np.load(path)
        if data.shape[0] != 4:
            data = data.T
        samples[fault] = data
        print(f"  [{fault}] {os.path.basename(path)}  shape={data.shape}")
    print("=" * 60)
    print("\nMODEL LAYERS:")
    for layer in model.layers:
        print(f"  {layer.name:45s}  {type(layer).__name__}")
    return model, samples

# ============================================================================
# PLOT 1 — Individual Grad-CAM per fault (2x2 grid)
# ============================================================================
def plot_gradcam_individual(model, samples):
    fig = plt.figure(figsize=(22, 14), facecolor='#0A0A0F')
    fig.suptitle('GRAD-CAM 1.XAI without AM — 1D CNN WITH Attention\n'
                 'Highlighted regions = Frequencies the model focused on for classification',
                 fontsize=16, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(2, 2, hspace=0.52, wspace=0.35, top=0.92, bottom=0.06)

    for idx, fault in enumerate(FAULT_NAMES):
        ax    = fig.add_subplot(gs[idx // 2, idx % 2])
        color = COLORS[fault]
        sig   = samples[fault]
        X     = prepare_input(sig)

        freqs, fft_mag = time_to_freq(sig[0])
        fft_norm       = z_score(fft_mag)

        # Prediction
        probs      = model.predict(X, verbose=0)[0]
        pred_idx   = np.argmax(probs)
        pred_class = FAULT_NAMES[pred_idx]
        correct    = '✓' if pred_class == fault else '✗'
        confidence = probs[pred_idx] * 100

        # Grad-CAM for TRUE class
        heatmap = compute_gradcam(model, X, FAULT_NAMES.index(fault))

        # Plot FFT signal
        ax.plot(freqs, fft_norm, color=color, linewidth=1.5,
                alpha=0.7, label='FFT Signal', zorder=2)

        # Overlay Grad-CAM heatmap as filled area
        ax2 = ax.twinx()
        ax2.fill_between(freqs, heatmap, alpha=0.45, color='red', label='Grad-CAM')
        ax2.plot(freqs, heatmap, color='red', linewidth=1.0, linestyle='--', alpha=0.8)
        ax2.set_ylabel('Grad-CAM Importance', fontsize=8, color='red')
        ax2.tick_params(labelsize=7, colors='red')
        ax2.set_ylim(0, 1.3)

        # Mark peak importance frequency
        peak_freq = freqs[np.argmax(heatmap)]
        ax.axvline(peak_freq, color='yellow', linewidth=1.5,
                   linestyle=':', alpha=0.9, zorder=3)
        ax.annotate(f'Peak: {peak_freq:.1f} Hz',
                    xy=(peak_freq, fft_norm.max() * 0.85),
                    xytext=(peak_freq + 20, fft_norm.max() * 0.92),
                    fontsize=8, color='yellow', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=1.0))

        # Top-3 important frequency bands
        sorted_idx = np.argsort(heatmap)[::-1]
        top3_freqs = freqs[sorted_idx[:3]]
        for tf_val in top3_freqs:
            ax.axvspan(tf_val - 5, tf_val + 5, alpha=0.15, color='yellow', zorder=1)

        ax.set_title(f'{fault}  →  Pred: {pred_class} {correct} ({confidence:.1f}%)\n'
                     f'Red area = model attention | Peak focus: {peak_freq:.1f} Hz',
                     fontsize=10, color=color, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=8)
        ax.set_ylabel('FFT Magnitude (z-scored)', fontsize=8)
        ax.grid(True, alpha=0.4)
        ax.tick_params(labelsize=7)

    fig.text(0.5, 0.01,
             'Grad-CAM: Gradient-weighted Class Activation Mapping  |  '
             'Higher red = more important frequency for that fault class prediction',
             ha='center', fontsize=9, color='#94A3B8',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1E2433', alpha=0.8))

    plt.savefig('xai_gradcam_individual_attention.png', dpi=150,
                bbox_inches='tight', facecolor='#0A0A0F')
    plt.close()
    print("✓ Saved: xai_gradcam_individual_attention.png")


# ============================================================================
# PLOT 2 — All 4 Grad-CAM heatmaps overlaid for comparison
# ============================================================================
def plot_gradcam_comparison(model, samples):
    fig, axes = plt.subplots(2, 1, figsize=(22, 12), facecolor='#0A0A0F')
    fig.suptitle('GRAD-CAM COMPARISON — All 4 Fault Classes (WITH Attention)\n'
                 'Which frequency bands does the model use to distinguish each fault?',
                 fontsize=16, fontweight='bold', color='white', y=0.99)

    freqs_ref = None
    heatmaps  = {}
    fft_norms = {}
    preds     = {}

    for fault in FAULT_NAMES:
        sig            = samples[fault]
        X              = prepare_input(sig)
        freqs, fft_mag = time_to_freq(sig[0])
        if freqs_ref is None:
            freqs_ref = freqs
        fft_norms[fault] = z_score(fft_mag)
        heatmaps[fault]  = compute_gradcam(model, X, FAULT_NAMES.index(fault))
        probs            = model.predict(X, verbose=0)[0]
        pred_idx         = np.argmax(probs)
        preds[fault]     = (FAULT_NAMES[pred_idx],
                            '✓' if FAULT_NAMES[pred_idx] == fault else '✗',
                            probs[pred_idx] * 100)

    # Top panel: FFT signals
    ax0 = axes[0]
    ax0.set_facecolor('#12141E')
    for fault in FAULT_NAMES:
        ax0.plot(freqs_ref, fft_norms[fault], color=COLORS[fault],
                 linewidth=1.5, alpha=0.85,
                 label=f'{fault} → {preds[fault][0]} {preds[fault][1]} ({preds[fault][2]:.1f}%)')
    ax0.set_title('FFT Signals — All 4 Fault Classes (Real Data)',
                  fontsize=12, color='white', fontweight='bold')
    ax0.set_xlabel('Frequency (Hz)', fontsize=9)
    ax0.set_ylabel('Z-scored Magnitude', fontsize=9)
    ax0.legend(fontsize=9, facecolor='#12141E', edgecolor='#444', labelcolor='white')
    ax0.grid(True, alpha=0.4)
    ax0.tick_params(labelsize=8)

    # Bottom panel: Grad-CAM heatmaps
    ax1 = axes[1]
    ax1.set_facecolor('#12141E')
    for fault in FAULT_NAMES:
        ax1.plot(freqs_ref, heatmaps[fault], color=COLORS[fault],
                 linewidth=2.0, alpha=0.85, label=fault)
        ax1.fill_between(freqs_ref, heatmaps[fault],
                         alpha=0.15, color=COLORS[fault])
        # Mark each peak
        pk = freqs_ref[np.argmax(heatmaps[fault])]
        ax1.axvline(pk, color=COLORS[fault], linewidth=0.8,
                    linestyle=':', alpha=0.7)
        ax1.annotate(f'{pk:.0f}Hz', xy=(pk, heatmaps[fault].max()),
                     xytext=(pk + 8, heatmaps[fault].max() * 0.95),
                     fontsize=7, color=COLORS[fault])

    ax1.set_title('Grad-CAM Importance — Which Frequencies Each Fault Class Relies On',
                  fontsize=12, color='white', fontweight='bold')
    ax1.set_xlabel('Frequency (Hz)', fontsize=9)
    ax1.set_ylabel('Grad-CAM Importance (normalised)', fontsize=9)
    ax1.legend(fontsize=9, facecolor='#12141E', edgecolor='#444', labelcolor='white')
    ax1.grid(True, alpha=0.4)
    ax1.tick_params(labelsize=8)
    ax1.set_ylim(0, 1.15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.text(0.5, 0.01,
             'Each curve shows which frequency range the model focuses on to classify that fault  |  '
             'Different peaks = model learned distinct frequency signatures per fault',
             ha='center', fontsize=9, color='#94A3B8',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1E2433', alpha=0.8))

    plt.savefig('xai_gradcam_comparison_attention.png', dpi=150,
                bbox_inches='tight', facecolor='#0A0A0F')
    plt.close()
    print("✓ Saved: xai_gradcam_comparison_attention.png")


# ============================================================================
# PLOT 3 — Heatmap grid: all classes × all channels
# ============================================================================
def plot_gradcam_channels(model, samples):
    fig = plt.figure(figsize=(24, 14), facecolor='#0A0A0F')
    fig.suptitle('GRAD-CAM PER CHANNEL — 1D CNN WITH Attention\n'
                 'How model importance varies across all 4 sensor channels',
                 fontsize=16, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(4, 4, hspace=0.55, wspace=0.30,
                           top=0.92, bottom=0.06)

    for row, fault in enumerate(FAULT_NAMES):
        color = COLORS[fault]
        sig   = samples[fault]

        for ch in range(4):
            ax = fig.add_subplot(gs[row, ch])

            # Build input using only this channel (others zeroed)
            channels = []
            for c in range(4):
                _, fft_mag = time_to_freq(sig[c])
                channels.append(z_score(fft_mag))
            freqs, _ = time_to_freq(sig[ch])

            X_full   = np.stack(channels, axis=-1)[np.newaxis].astype(np.float32)
            heatmap  = compute_gradcam(model, X_full, FAULT_NAMES.index(fault))
            fft_norm = channels[ch]

            ax.plot(freqs, fft_norm, color=color, linewidth=1.0,
                    alpha=0.6, label='FFT')
            ax2 = ax.twinx()
            ax2.fill_between(freqs, heatmap, alpha=0.4, color='red')
            ax2.plot(freqs, heatmap, color='red', linewidth=0.8, linestyle='--')
            ax2.set_ylim(0, 1.3)
            ax2.tick_params(labelsize=6, colors='red')

            pk = freqs[np.argmax(heatmap)]
            ax.axvline(pk, color='yellow', linewidth=1.0, linestyle=':', alpha=0.8)

            title = f'{fault} — Ch{ch+1}' if ch == 0 else f'Ch{ch+1}'
            if ch == 0:
                ax.set_ylabel(fault, fontsize=9, color=color, fontweight='bold')
            ax.set_title(f'Ch{ch+1} | Peak: {pk:.0f}Hz',
                         fontsize=8, color=color)
            ax.set_xlabel('Freq (Hz)', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

    fig.text(0.5, 0.01,
             'Each cell: FFT signal (colored) + Grad-CAM heatmap (red) for that fault × channel combination  |  '
             'Yellow line = peak importance frequency',
             ha='center', fontsize=9, color='#94A3B8',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1E2433', alpha=0.8))

    plt.savefig('xai_gradcam_channels_attention.png', dpi=150,
                bbox_inches='tight', facecolor='#0A0A0F')
    plt.close()
    print("✓ Saved: xai_gradcam_channels_attention.png")


# ============================================================================
# PLOT 4 — Summary importance bar chart
# ============================================================================
def plot_importance_summary(model, samples):
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor='#0A0A0F')
    fig.suptitle('FREQUENCY IMPORTANCE SUMMARY — 1D CNN WITH Attention\n'
                 'Top important frequency bands per fault class (Grad-CAM)',
                 fontsize=16, fontweight='bold', color='white', y=0.99)

    # Define frequency bands
    bands = {
        '0–50 Hz':    (0,   50),
        '50–100 Hz':  (50,  100),
        '100–200 Hz': (100, 200),
        '200–300 Hz': (200, 300),
        '300–400 Hz': (300, 400),
        '400–500 Hz': (400, 500),
    }

    for idx, fault in enumerate(FAULT_NAMES):
        ax    = axes[idx]
        ax.set_facecolor('#12141E')
        color = COLORS[fault]
        sig   = samples[fault]
        X     = prepare_input(sig)

        freqs, _ = time_to_freq(sig[0])
        heatmap  = compute_gradcam(model, X, FAULT_NAMES.index(fault))

        # Average Grad-CAM importance per band
        band_names   = list(bands.keys())
        band_scores  = []
        for bname, (flow, fhigh) in bands.items():
            mask  = (freqs >= flow) & (freqs < fhigh)
            score = np.mean(heatmap[mask]) if np.any(mask) else 0
            band_scores.append(score)

        # Normalize
        band_scores = np.array(band_scores)
        if band_scores.max() > 0:
            band_scores = band_scores / band_scores.max()

        bar_colors = [color if s == band_scores.max() else '#334155'
                      for s in band_scores]
        bars = ax.barh(band_names, band_scores, color=bar_colors,
                       alpha=0.85, edgecolor='white', linewidth=0.5)

        for bar, score in zip(bars, band_scores):
            ax.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{score:.2f}', va='center', fontsize=8,
                    color='white', fontweight='bold')

        probs      = model.predict(X, verbose=0)[0]
        pred_idx   = np.argmax(probs)
        pred_class = FAULT_NAMES[pred_idx]
        correct    = '✓' if pred_class == fault else '✗'

        ax.set_title(f'{fault}\nPred: {pred_class} {correct} ({probs[pred_idx]*100:.1f}%)',
                     fontsize=10, color=color, fontweight='bold')
        ax.set_xlabel('Normalised Importance', fontsize=8)
        ax.set_xlim(0, 1.25)
        ax.grid(True, alpha=0.4, axis='x')
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.text(0.5, 0.01,
             'Bar height = average Grad-CAM importance in that frequency band  |  '
             'Brightest bar = most important band for that fault',
             ha='center', fontsize=9, color='#94A3B8',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1E2433', alpha=0.8))

    plt.savefig('xai_gradcam_summary_attention.png', dpi=150,
                bbox_inches='tight', facecolor='#0A0A0F')
    plt.close()
    print("✓ Saved: xai_gradcam_summary_attention.png")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("GRAD-CAM 1.XAI without AM — 1D CNN WITH ATTENTION")
    print("=" * 60)

    model, samples = load_model_and_data()

    print("\nGenerating Grad-CAM plots...")
    plot_gradcam_individual(model, samples)
    plot_gradcam_comparison(model, samples)
    plot_gradcam_channels(model, samples)
    plot_importance_summary(model, samples)

    print("\n" + "=" * 60)
    print("ALL 4 1.XAI without AM PLOTS SAVED")
    print("=" * 60)
    print("  xai_gradcam_individual_attention.png  ← per fault heatmap")
    print("  xai_gradcam_comparison_attention.png  ← all faults overlaid")
    print("  xai_gradcam_channels_attention.png    ← per channel analysis")
    print("  xai_gradcam_summary_attention.png     ← frequency band importance")
    print("=" * 60)