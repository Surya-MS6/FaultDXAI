"""
GRAD-CAM 1.XAI without AM Visualization — 1D CNN WITHOUT Attention Mechanism
===============================================================
Explains MODEL DECISIONS by showing which frequency regions
influenced the prediction for each fault class.

Dataset structure:
    ROOT/
    ├── Normal/Test XX_Normal/*.npy
    ├── Misalignment/Test XX_Misalignment/*.npy
    ├── Looseness/Test XX_Looseness/*.npy
    └── Unbalance/Test XX_Unbalance/*.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import fft, fftfreq
from scipy.ndimage import maximum_filter
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
# ✅ Change to your chosen model file
# e.g. r"models_varying_real\model_real_40.keras"
#      r"models_varying_real\model_real_100.keras"
MODEL_PATH = r"C:\Users\HP\PycharmProjects\PythonProject\XAI\model_real_20.keras"

# ✅ CHANGE THESE TO YOUR CHOSEN FILES
CUSTOM_SAMPLES = {
        "Normal": r"D:\Final Project\New folder\Real Dataset\Normal\Test 17_Normal Condition\20220506-142317.npy",
        "Misalignment": r"D:\Final Project\New folder\Real Dataset\Misalignment\Test 13_Misalignment\20220506-124407.npy",
        "Looseness": r"D:\Final Project\New folder\Real Dataset\Looseness\Test 08_Looseness\20220429-101809.npy",
        "Unbalance": r"D:\Final Project\New folder\Real Dataset\Unbalance\Test 18_Unbalance\20220506-143510.npy",
    }

SAMPLING_RATE = 25000
MAX_FREQ      = 500
FAULT_NAMES   = ['Normal', 'Misalignment', 'Looseness', 'Unbalance']

# ============================================================================
# COLORS & STYLE
# ============================================================================
COLORS = {
    'Normal':       '#00FF99',
    'Misalignment': '#00BFFF',
    'Looseness':    '#FF69B4',
    'Unbalance':    '#FFD700',
}
STAGE_COLOR = '#00FFFF'

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
# LOAD MODEL — No custom objects needed
# ============================================================================
def load_model(model_path):
    print("=" * 60)
    print("LOADING MODEL (WITHOUT ATTENTION)")
    print("=" * 60)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Check MODEL_PATH at the top of this script."
        )
    model = keras.models.load_model(model_path)
    print(f"  ✓ Loaded: {model_path}")
    print(f"  Input:  {model.input_shape}")
    print(f"  Output: {model.output_shape}")
    print("\nLayer names:")
    for i, layer in enumerate(model.layers):
        print(f"  [{i}] {layer.name:25s}  type: {type(layer).__name__}")
    print("=" * 60)
    return model


# ============================================================================
# LOAD CUSTOM FILES
# ============================================================================
def load_custom_samples():
    print("=" * 60)
    print("LOADING CUSTOM SELECTED FILES")
    print("=" * 60)
    samples = {}
    for fault, path in CUSTOM_SAMPLES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File not found: {path}\nCheck CUSTOM_SAMPLES paths."
            )
        data = np.load(path)
        if data.shape[0] != 4:
            data = data.T
        samples[fault] = data
        print(f"  [{fault}] {os.path.basename(path)}  shape={data.shape}")
    print("=" * 60)
    return samples


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
    """(4, N) → (1, freq_bins, 4)"""
    channels = []
    for ch in range(4):
        _, fft_mag = time_to_freq(signal_4ch[ch])
        channels.append(z_score(fft_mag))
    X = np.stack(channels, axis=-1)
    return X[np.newaxis, :, :].astype(np.float32)


# ============================================================================
# GRAD-CAM FOR 1D CNN (no attention)
# ============================================================================
def compute_gradcam(model, X, class_idx, conv_layer_name='conv1d_layer'):
    conv_layer = model.get_layer(conv_layer_name)
    grad_model = keras.Model(inputs=model.inputs,
                             outputs=[conv_layer.output, model.outputs[0]])

    inputs = tf.cast(X, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        conv_outputs, predictions = grad_model(inputs, training=False)
        loss = predictions[:, class_idx]

    input_grads = tape.gradient(loss, inputs)

    if input_grads is None:
        heatmap = tf.reduce_mean(tf.abs(conv_outputs[0]), axis=-1).numpy()
    else:
        heatmap = tf.reduce_mean(tf.abs(input_grads[0]), axis=-1).numpy()

    conv_act = tf.reduce_mean(tf.abs(conv_outputs[0]), axis=-1).numpy()
    if conv_act.shape[0] != heatmap.shape[0]:
        conv_act = np.interp(
            np.linspace(0, 1, heatmap.shape[0]),
            np.linspace(0, 1, conv_act.shape[0]),
            conv_act
        )

    heatmap = 0.6 * heatmap + 0.4 * conv_act
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap

def get_predictions(model, samples):
    print("=" * 60)
    print("RUNNING PREDICTIONS")
    print("=" * 60)
    predictions = {}
    for fault in FAULT_NAMES:
        X          = prepare_input(samples[fault])
        probs      = model.predict(X, verbose=0)[0]
        pred_idx   = np.argmax(probs)
        pred_class = FAULT_NAMES[pred_idx]
        correct    = '✓' if pred_class == fault else '✗'
        predictions[fault] = {
            'probs':      probs,
            'pred_class': pred_class,
            'pred_idx':   pred_idx,
            'correct':    correct,
            'confidence': probs[pred_idx] * 100,
            'X':          prepare_input(samples[fault]),
        }
        print(f"  [{fault:>12}] → {pred_class:<14} {correct}  "
              f"Confidence: {probs[pred_idx]*100:.1f}%")
    print("=" * 60)
    return predictions


# ============================================================================
# PLOT 1 — GRAD-CAM PER FAULT CLASS (main result)
# ============================================================================
def plot_gradcam_main(model, samples, predictions):
    fig = plt.figure(figsize=(22, 16), facecolor='#0A0A0F')
    fig.suptitle('GRAD-CAM 1.XAI without AM — Frequency Regions Driving Model Decision\n'
                 '(1D CNN Without Attention)',
                 fontsize=18, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(4, 1, hspace=0.55, top=0.93, bottom=0.05)

    for row, fault in enumerate(FAULT_NAMES):
        color = COLORS[fault]
        p     = predictions[fault]
        sig   = samples[fault]
        freqs, fft_mag = time_to_freq(sig[0])
        fft_norm       = z_score(fft_mag)
        heatmap        = compute_gradcam(model, p['X'], p['pred_idx'])

        ax  = fig.add_subplot(gs[row])
        ax2 = ax.twinx()

        ax.fill_between(freqs, fft_norm, alpha=0.3, color=color)
        ax.plot(freqs, fft_norm, color=color, linewidth=1.5,
                label='FFT Magnitude (z-scored)')

        ax2.fill_between(freqs, heatmap, alpha=0.5, color='red')
        ax2.plot(freqs, heatmap, color='red', linewidth=1.2,
                 linestyle='--', label='Grad-CAM Importance')
        ax2.set_ylabel('Grad-CAM Score', fontsize=8, color='red')
        ax2.tick_params(labelsize=7, colors='red')
        ax2.set_ylim(0, 1.3)

        top_freq = freqs[np.argmax(heatmap)]
        ax.axvline(top_freq, color='yellow', linewidth=1.5,
                   linestyle=':', alpha=0.9)
        ax.annotate(f'Key freq: {top_freq:.1f} Hz',
                    xy=(top_freq, fft_norm.max() * 0.85),
                    xytext=(top_freq + 20, fft_norm.max() * 0.92),
                    fontsize=8, color='yellow', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=1))

        threshold = 0.7
        important = freqs[heatmap >= threshold]
        if len(important) > 0:
            ax.axvspan(important[0], important[-1], alpha=0.08,
                       color='yellow', label=f'High importance (≥{threshold})')

        ax.set_title(
            f'{fault}  |  Predicted: {p["pred_class"]} {p["correct"]}  '
            f'Confidence: {p["confidence"]:.1f}%  |  '
            f'Model focused on: {top_freq:.1f} Hz',
            fontsize=10, color=color, fontweight='bold'
        )
        ax.set_xlabel('Frequency (Hz)', fontsize=8)
        ax.set_ylabel('FFT Magnitude (z-scored)', fontsize=8)
        ax.grid(True, alpha=0.4)
        ax.tick_params(labelsize=7)
        ax.set_xlim(freqs[0], freqs[-1])

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=7, facecolor='#12141E',
                  edgecolor='#444', labelcolor='white', loc='upper right')

    plt.savefig('xai_gradcam_per_class.png', dpi=150,
                bbox_inches='tight', facecolor='#0A0A0F')
    plt.close()
    print("✓ Saved: xai_gradcam_per_class.png")


# ============================================================================
# PLOT 2 — GRAD-CAM HEATMAP COMPARISON
# ============================================================================
def plot_gradcam_comparison(model, samples, predictions):
    fig = plt.figure(figsize=(22, 12), facecolor='#0A0A0F')
    fig.suptitle('GRAD-CAM 1.XAI without AM — Importance Heatmap Comparison Across Fault Classes\n'
                 '(1D CNN Without Attention)',
                 fontsize=17, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(2, 1, hspace=0.45, top=0.92, bottom=0.06)

    all_heatmaps = []
    all_freqs    = None

    for fault in FAULT_NAMES:
        p   = predictions[fault]
        sig = samples[fault]
        freqs, _ = time_to_freq(sig[0])
        if all_freqs is None:
            all_freqs = freqs
        heatmap = compute_gradcam(model, p['X'], p['pred_idx'])
        all_heatmaps.append(heatmap)

    heatmap_matrix = np.array(all_heatmaps)

    ax0 = fig.add_subplot(gs[0])
    im  = ax0.imshow(
        heatmap_matrix, aspect='auto', cmap='hot',
        extent=[all_freqs[0], all_freqs[-1], 3.5, -0.5],
        vmin=0, vmax=1, interpolation='nearest'
    )
    ax0.set_yticks([0, 1, 2, 3])
    ax0.set_yticklabels(
        [f'{fn}  →  {predictions[fn]["pred_class"]} '
         f'{predictions[fn]["correct"]}  ({predictions[fn]["confidence"]:.1f}%)'
         for fn in FAULT_NAMES],
        fontsize=9
    )
    ax0.set_xlabel('Frequency (Hz)', fontsize=9)
    ax0.set_title('Grad-CAM Importance Heatmap — Brighter = More Important Frequency',
                  fontsize=11, color='white', fontweight='bold')
    ax0.tick_params(labelsize=8)
    plt.colorbar(im, ax=ax0, fraction=0.02, pad=0.01, label='Importance Score')

    ax1 = fig.add_subplot(gs[1])
    for fault, heatmap in zip(FAULT_NAMES, all_heatmaps):
        p = predictions[fault]
        ax1.plot(all_freqs, heatmap, color=COLORS[fault], linewidth=2.0,
                 label=f'{fault} → {p["pred_class"]} {p["correct"]}',
                 alpha=0.85)
    ax1.set_title('Grad-CAM Score Overlay — All 4 Fault Classes',
                  fontsize=11, color='white', fontweight='bold')
    ax1.set_xlabel('Frequency (Hz)', fontsize=9)
    ax1.set_ylabel('Grad-CAM Importance Score', fontsize=9)
    ax1.legend(fontsize=9, facecolor='#12141E', edgecolor='#444', labelcolor='white')
    ax1.grid(True, alpha=0.4)
    ax1.tick_params(labelsize=8)
    ax1.set_xlim(all_freqs[0], all_freqs[-1])

    plt.savefig('xai_gradcam_comparison.png', dpi=150,
                bbox_inches='tight', facecolor='#0A0A0F')
    plt.close()
    print("✓ Saved: xai_gradcam_comparison.png")


# ============================================================================
# PLOT 3 — GRAD-CAM vs DROPOUT EFFECT (unique to no-attention model)
# ============================================================================
def plot_gradcam_vs_dropout(model, samples, predictions):
    """
    Shows Grad-CAM alongside Conv1D activation — unique insight for no-attention model.
    Demonstrates what features survived dropout and contributed to the decision.
    """
    fig = plt.figure(figsize=(22, 14), facecolor='#0A0A0F')
    fig.suptitle('GRAD-CAM vs Conv1D Activations — Feature Contribution Analysis\n'
                 '(1D CNN Without Attention)',
                 fontsize=17, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(2, 2, hspace=0.52, wspace=0.35, top=0.92, bottom=0.06)

    # Build conv output model
    conv_model = keras.Model(
        inputs  = model.inputs,
        outputs = model.get_layer('conv1d_layer').output
    )

    for idx, fault in enumerate(FAULT_NAMES):
        ax    = fig.add_subplot(gs[idx // 2, idx % 2])
        color = COLORS[fault]
        p     = predictions[fault]
        sig   = samples[fault]
        freqs, fft_mag = time_to_freq(sig[0])
        fft_norm       = z_score(fft_mag)

        # Grad-CAM
        gradcam = compute_gradcam(model, p['X'], p['pred_idx'])

        # Mean conv activation across filters
        conv_out  = conv_model.predict(p['X'], verbose=0)[0]  # (freq_bins, 32)
        mean_conv = np.mean(conv_out, axis=-1)                # (freq_bins,)
        if mean_conv.max() > 0:
            mean_conv = mean_conv / mean_conv.max()

        ax2 = ax.twinx()

        # FFT background
        ax.fill_between(freqs, fft_norm, alpha=0.15, color=color)
        ax.plot(freqs, fft_norm, color=color, linewidth=1.0, alpha=0.5,
                label='FFT Signal')

        # Grad-CAM
        ax.plot(freqs, gradcam, color='red', linewidth=1.8,
                label='Grad-CAM')

        # Conv activation
        ax2.plot(freqs, mean_conv, color='cyan', linewidth=1.5,
                 linestyle='--', label='Conv Activation (norm)', alpha=0.85)
        ax2.set_ylabel('Conv Activation', fontsize=7, color='cyan')
        ax2.tick_params(labelsize=6, colors='cyan')

        ax.set_title(
            f'{fault}  →  {p["pred_class"]} {p["correct"]}  ({p["confidence"]:.1f}%)',
            fontsize=9, color=color, fontweight='bold'
        )
        ax.set_xlabel('Frequency (Hz)', fontsize=8)
        ax.set_ylabel('Grad-CAM Score', fontsize=8)
        ax.grid(True, alpha=0.4)
        ax.tick_params(labelsize=7)
        ax.set_xlim(freqs[0], freqs[-1])

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=7, facecolor='#12141E',
                  edgecolor='#444', labelcolor='white')

    plt.savefig('xai_gradcam_vs_conv.png', dpi=150,
                bbox_inches='tight', facecolor='#0A0A0F')
    plt.close()
    print("✓ Saved: xai_gradcam_vs_conv.png")


# ============================================================================
# PLOT 4 — TOP IMPORTANT FREQUENCY BANDS PER CLASS
# ============================================================================
def plot_important_bands(model, samples, predictions):
    fig = plt.figure(figsize=(22, 10), facecolor='#0A0A0F')
    fig.suptitle('GRAD-CAM 1.XAI without AM — Top Important Frequency Bands Per Fault Class\n'
                 '(1D CNN Without Attention)',
                 fontsize=17, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(1, 4, hspace=0.4, wspace=0.4,
                           top=0.88, bottom=0.12)

    for idx, fault in enumerate(FAULT_NAMES):
        ax    = fig.add_subplot(gs[idx])
        color = COLORS[fault]
        p     = predictions[fault]
        sig   = samples[fault]
        freqs, _ = time_to_freq(sig[0])
        heatmap  = compute_gradcam(model, p['X'], p['pred_idx'])

        smoothed   = np.convolve(heatmap, np.ones(10)/10, mode='same')
        top5_idx   = np.argsort(smoothed)[::-1][:5]
        top5_freqs = freqs[top5_idx]
        top5_scores = heatmap[top5_idx]

        bars = ax.barh(
            [f'{f:.1f} Hz' for f in top5_freqs],
            top5_scores,
            color=color, alpha=0.85, edgecolor='white', linewidth=0.8
        )
        for bar, score in zip(bars, top5_scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontsize=8,
                    color='white', fontweight='bold')

        ax.set_title(f'{fault}\n→ {p["pred_class"]} {p["correct"]} '
                     f'({p["confidence"]:.1f}%)',
                     fontsize=9, color=color, fontweight='bold')
        ax.set_xlabel('Grad-CAM Score', fontsize=8)
        ax.set_ylabel('Frequency Band', fontsize=8)
        ax.set_xlim(0, 1.2)
        ax.grid(True, alpha=0.4, axis='x')
        ax.tick_params(labelsize=8)
        ax.invert_yaxis()

    plt.savefig('xai_important_bands.png', dpi=150,
                bbox_inches='tight', facecolor='#0A0A0F')
    plt.close()
    print("✓ Saved: xai_important_bands.png")


# ============================================================================
# PLOT 5 — SUMMARY 1.XAI without AM OVERVIEW
# ============================================================================
def plot_xai_summary(model, samples, predictions):
    fig = plt.figure(figsize=(24, 16), facecolor='#0A0A0F')
    fig.suptitle('1.XAI without AM SUMMARY — 1D CNN (No Attention)  |  Real Data Predictions + Grad-CAM',
                 fontsize=18, fontweight='bold', color='white', y=0.99)

    gs = gridspec.GridSpec(4, 3, hspace=0.6, wspace=0.35,
                           top=0.94, bottom=0.05,
                           width_ratios=[2, 2, 1])

    bar_colors = list(COLORS.values())

    for row, fault in enumerate(FAULT_NAMES):
        color = COLORS[fault]
        p     = predictions[fault]
        sig   = samples[fault]
        freqs, fft_mag = time_to_freq(sig[0])
        fft_norm       = z_score(fft_mag)
        heatmap        = compute_gradcam(model, p['X'], p['pred_idx'])

        # Col 0: FFT signal
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.fill_between(freqs, fft_norm, alpha=0.3, color=color)
        ax0.plot(freqs, fft_norm, color=color, linewidth=1.3)
        ax0.set_title(f'{fault} — FFT Input', fontsize=9, color=color, fontweight='bold')
        ax0.set_xlabel('Frequency (Hz)', fontsize=7)
        ax0.set_ylabel('Magnitude', fontsize=7)
        ax0.grid(True, alpha=0.4); ax0.tick_params(labelsize=6)

        # Col 1: FFT + Grad-CAM overlay
        ax1  = fig.add_subplot(gs[row, 1])
        ax1b = ax1.twinx()
        ax1.fill_between(freqs, fft_norm, alpha=0.2, color=color)
        ax1.plot(freqs, fft_norm, color=color, linewidth=1.2, alpha=0.7)
        ax1b.fill_between(freqs, heatmap, alpha=0.55, color='red')
        ax1b.plot(freqs, heatmap, color='red', linewidth=1.2, linestyle='--')
        ax1b.set_ylim(0, 1.3)
        ax1b.tick_params(labelsize=6, colors='red')
        top_f = freqs[np.argmax(heatmap)]
        ax1.axvline(top_f, color='yellow', linewidth=1.5, linestyle=':')
        ax1.annotate(f'{top_f:.1f} Hz', xy=(top_f, fft_norm.max() * 0.8),
                     xytext=(top_f + 15, fft_norm.max() * 0.88),
                     fontsize=7, color='yellow',
                     arrowprops=dict(arrowstyle='->', color='yellow', lw=0.8))
        ax1.set_title(f'Grad-CAM — Key freq: {top_f:.1f} Hz\n'
                      f'Pred: {p["pred_class"]} {p["correct"]} ({p["confidence"]:.1f}%)',
                      fontsize=9, color='white', fontweight='bold')
        ax1.set_xlabel('Frequency (Hz)', fontsize=7)
        ax1.set_ylabel('FFT Magnitude', fontsize=7)
        ax1.grid(True, alpha=0.4); ax1.tick_params(labelsize=6)

        # Col 2: Softmax probabilities
        ax2  = fig.add_subplot(gs[row, 2])
        bars = ax2.bar(range(4), p['probs'], color=bar_colors,
                       alpha=0.85, edgecolor='white', linewidth=0.6)
        bars[FAULT_NAMES.index(fault)].set_edgecolor('white')
        bars[FAULT_NAMES.index(fault)].set_linewidth(2.5)
        bars[p['pred_idx']].set_edgecolor('yellow')
        bars[p['pred_idx']].set_linewidth(2.5)
        for bar, pr in zip(bars, p['probs']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{pr*100:.0f}%', ha='center', va='bottom',
                     fontsize=7, color='white', fontweight='bold')
        ax2.set_title('Softmax Output', fontsize=9, color='white', fontweight='bold')
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(['N', 'M', 'L', 'U'], fontsize=7)
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel('Probability', fontsize=7)
        ax2.grid(True, alpha=0.4, axis='y'); ax2.tick_params(labelsize=7)

    fig.text(0.5, 0.01,
             'N=Normal  M=Misalignment  L=Looseness  U=Unbalance  |  '
             'Yellow border=True class  |  Yellow dashed=Key frequency  |  '
             'Red fill=Grad-CAM importance',
             ha='center', fontsize=8, color='#94A3B8',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1E2433', alpha=0.8))

    plt.savefig('xai_summary.png', dpi=150,
                bbox_inches='tight', facecolor='#0A0A0F')
    plt.close()
    print("✓ Saved: xai_summary.png")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("GRAD-CAM 1.XAI without AM — 1D CNN WITHOUT ATTENTION")
    print("=" * 60)

    # 1. Load model
    model = load_model(MODEL_PATH)

    # 2. Load custom files
    samples = load_custom_samples()

    # 3. Get predictions
    predictions = get_predictions(model, samples)

    # 4. Generate all 1.XAI without AM plots
    print("\nGenerating 1.XAI without AM plots...")
    plot_gradcam_main(model, samples, predictions)
    plot_gradcam_comparison(model, samples, predictions)
    plot_gradcam_vs_dropout(model, samples, predictions)
    plot_important_bands(model, samples, predictions)
    plot_xai_summary(model, samples, predictions)

    print("\n" + "=" * 60)
    print("ALL 5 1.XAI without AM PLOTS SAVED")
    print("=" * 60)
    print("  xai_gradcam_per_class.png       ← main Grad-CAM result")
    print("  xai_gradcam_comparison.png      ← heatmap across all classes")
    print("  xai_gradcam_vs_conv.png         ← Grad-CAM vs Conv activations")
    print("  xai_important_bands.png         ← top 5 frequency bands per fault")
    print("  xai_summary.png                 ← best single slide for presentation")
    print("=" * 60)
    print("\nPREDICTION SUMMARY:")
    correct = 0
    for fault in FAULT_NAMES:
        p = predictions[fault]
        print(f"  {fault:<14} → {p['pred_class']:<14} {p['correct']}  "
              f"({p['confidence']:.1f}%)")
        if p['correct'] == '✓':
            correct += 1
    print(f"\n  Accuracy on 4 samples: {correct}/4 ({correct*25}%)")
    print("=" * 60)