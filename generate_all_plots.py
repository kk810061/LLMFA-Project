import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Style configuration for clean, presentation-ready academic charts
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

output_dirs = [
    r'c:\Users\kriti\Downloads\ZipZap-main',
    r'C:\Users\kriti\.gemini\antigravity\brain\b130051c-0097-4ec9-b6f3-5e96c3def010'
]

# 1. Load predictions
npz_path = r'c:\Users\kriti\Downloads\ZipZap-main\Model\eval_predictions.npz'
if os.path.exists(npz_path):
    data = np.load(npz_path)
    fpr = data['fpr']
    tpr = data['tpr']
    y_true = data['y_true']
    y_scores = data['y_scores']
    roc_auc = auc(fpr, tpr)
    print(f"Loaded npz data: AUC = {roc_auc:.4f}, N = {len(y_true)}")
else:
    print("npz not ready yet, using parsed ground truth.")
    y_true = None

# --- CHART 1: ROC Curve ---
if y_true is not None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    ax.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'ZipZap BERT (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#888888', lw=1.2, linestyle='--', label='Random Classifier (AUC = 0.5000)')

    # Highlight operating point at threshold 0.5
    y_pred_05 = (y_scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_05).ravel()
    fpr_05 = fp / (fp + tn)
    tpr_05 = tp / (tp + fn)
    ax.scatter([fpr_05], [tpr_05], color='#d62728', s=80, zorder=5, label=f'Operating Point (thresh=0.5)\nFPR={fpr_05*100:.2f}%, TPR={tpr_05*100:.1f}%')

    # Highlight operating point at threshold 0.1
    y_pred_01 = (y_scores >= 0.1).astype(int)
    tn1, fp1, fn1, tp1 = confusion_matrix(y_true, y_pred_01).ravel()
    fpr_01 = fp1 / (fp1 + tn1)
    tpr_01 = tp1 / (tp1 + fn1)
    ax.scatter([fpr_01], [tpr_01], color='#2ca02c', s=80, zorder=5, label=f'High-Recall (thresh=0.1)\nFPR={fpr_01*100:.2f}%, TPR={tpr_01*100:.1f}%')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('ROC Curve: Ethereum Fraud Detection (Stage 1)\nZipZap Reproduction on $D_S$ (1M pool)', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    for out_dir in output_dirs:
        fig.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=300)
    plt.close(fig)
    print("Saved roc_curve.png")

# --- CHART 2 & 3: Confusion Matrices ---
def plot_cm(cm, thresh, filename, title_suffix):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ['Benign (0)', 'Phishing (1)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'Confusion Matrix (Threshold = {thresh})\n{title_suffix}',
           ylabel='Actual Class (Ground Truth)',
           xlabel='Predicted Class')
    
    ax.set_title(ax.get_title(), fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Actual Class (Ground Truth)', fontsize=11, fontweight='bold', labelpad=8)

    # Annotate cells
    thresh_color = cm.max() / 2.
    labels = [
        [f"True Negative\n{cm[0, 0]:,}\n({cm[0, 0]/(cm[0, 0]+cm[0, 1])*100:.2f}%)",
         f"False Positive\n{cm[0, 1]:,}\n({cm[0, 1]/(cm[0, 0]+cm[0, 1])*100:.2f}%)"],
        [f"False Negative\n{cm[1, 0]:,}\n({cm[1, 0]/(cm[1, 0]+cm[1, 1])*100:.2f}%)",
         f"True Positive\n{cm[1, 1]:,}\n({cm[1, 1]/(cm[1, 0]+cm[1, 1])*100:.2f}%)"]
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh_color else "black",
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    for out_dir in output_dirs:
        fig.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close(fig)
    print(f"Saved {filename}")

if y_true is not None:
    cm_05 = confusion_matrix(y_true, (y_scores >= 0.5).astype(int))
    plot_cm(cm_05, 0.5, 'confusion_matrix_threshold_0.5.png', 'Operational Default (High Precision: 84.15%)')

    cm_01 = confusion_matrix(y_true, (y_scores >= 0.1).astype(int))
    plot_cm(cm_01, 0.1, 'confusion_matrix_threshold_0.1.png', 'High Recall Configuration (Recall: 72.46%)')

# --- CHART 4: Threshold Tradeoff Curve ---
thresholds_arr = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
precisions = [0.3645, 0.5086, 0.6069, 0.6739, 0.7175, 0.7574, 0.7850, 0.8096, 0.8260, 0.8415, 0.8575, 0.8889, 0.9016, 0.9217, 0.9349, 0.9531, 0.9558, 0.9710, 0.9853]
recalls    = [0.7736, 0.7246, 0.6842, 0.6475, 0.6157, 0.5924, 0.5630, 0.5361, 0.5055, 0.4810, 0.4639, 0.4406, 0.4149, 0.3892, 0.3513, 0.3231, 0.2913, 0.2460, 0.1640]
f1_scores  = [0.4955, 0.5977, 0.6433, 0.6604, 0.6627, 0.6648, 0.6557, 0.6451, 0.6272, 0.6121, 0.6021, 0.5892, 0.5683, 0.5473, 0.5107, 0.4826, 0.4465, 0.3926, 0.2812]

fig, ax = plt.subplots(figsize=(7, 4.8), dpi=300)
ax.plot(thresholds_arr, precisions, marker='o', markersize=4, color='#1f77b4', lw=2, label='Precision (Phishing)')
ax.plot(thresholds_arr, recalls, marker='s', markersize=4, color='#d62728', lw=2, label='Recall (Phishing)')
ax.plot(thresholds_arr, f1_scores, marker='^', markersize=4, color='#2ca02c', lw=2, label='F1-Score')

ax.axvline(0.5, color='#888888', linestyle='--', alpha=0.7, label='Standard Threshold (0.5)')
ax.set_xlabel('Classification Threshold', fontsize=11, fontweight='bold', labelpad=8)
ax.set_ylabel('Score', fontsize=11, fontweight='bold', labelpad=8)
ax.set_title('Precision, Recall & F1 vs. Classification Threshold\nTrade-off Under Severe Imbalance (115:1 Benign:Phish)', fontsize=12, fontweight='bold', pad=12)
ax.set_xlim([0.02, 0.98])
ax.set_ylim([0.1, 1.02])
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='center right', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=9.5)
plt.tight_layout()

for out_dir in output_dirs:
    fig.savefig(os.path.join(out_dir, 'threshold_tradeoff.png'), dpi=300)
plt.close(fig)
print("Saved threshold_tradeoff.png")

# --- CHART 5: Pre-training Loss Curve ---
loss_json = r'c:\Users\kriti\Downloads\ZipZap-main\Model\pretrain_loss_data.json'
if os.path.exists(loss_json):
    with open(loss_json, 'r') as f:
        loss_pts = json.load(f)
    iters = [p['iter'] for p in loss_pts]
    losses = [p['loss'] for p in loss_pts]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(iters, losses, color='#6a1b9a', lw=2, label='Masked LM Pre-training Loss')
    ax.set_xlabel('Pre-training Iterations (Steps)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title('Pre-training Convergence Curve (5 Epochs, 72,402 Steps)\nZipZap Transformer on Kaggle GPU', fontsize=12, fontweight='bold', pad=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Annotate start and finish
    ax.annotate(f'Initial Loss: {losses[0]:.2f}', xy=(iters[0], losses[0]), xytext=(iters[0]+3000, losses[0]+0.2),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1), fontsize=9, fontweight='bold')
    ax.annotate(f'Final Loss: {losses[-1]:.2f}', xy=(iters[-1], losses[-1]), xytext=(iters[-1]-15000, losses[-1]+0.8),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1), fontsize=9, fontweight='bold')
    
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=9.5)
    plt.tight_layout()

    for out_dir in output_dirs:
        fig.savefig(os.path.join(out_dir, 'pretrain_loss_curve.png'), dpi=300)
    plt.close(fig)
    print("Saved pretrain_loss_curve.png")
