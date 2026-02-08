import torch
import torchaudio as T
import numpy as np
import os
import sys
from tqdm import tqdm
import torchaudio.transforms as TT
import torch.nn.functional as Fk
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from training_files.params import AttrDict, params as base_params
from training_files.model import WaveGrad
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import json

# -----------------------------
# SETTINGS
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sr_o=12000
hop = 300
win = hop * 4
n_fft = 2**((win-1).bit_length())
f_max = sr_o / 2.0
mel= TT.MelSpectrogram(sample_rate=sr_o, n_fft=n_fft, win_length=win, hop_length=hop, f_min=20.0, f_max=f_max, power=1.0, normalized=False, center=False).to(device)

class_mapping = {"n": 0, "b": 1, "i": 2, "o": 3}
class_names = ["normal", "ball", "inner", "outer"]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def load_model(model_dir, device, noise_schedule=None):
    print(f"🔄 Loading model from {model_dir} ...")
    checkpoint = torch.load(model_dir, map_location=device)
    model = WaveGrad(AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("✅ Model loaded successfully.")
    if noise_schedule is not None:
        model.params.noise_schedule = noise_schedule.tolist()
    return model

def predict_audio(spectrogram, model):
    beta = np.array(model.params.noise_schedule)
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)

    if len(spectrogram.shape) == 2:
        spectrogram = spectrogram.unsqueeze(0)
    spectrogram = spectrogram.to(device)

    audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

    with torch.no_grad():
        for n in range(len(alpha)-1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = (1 - alpha[n]) / (1 - alpha_cum[n])**0.5
            audio = c1 * (audio - c2 * model(audio, spectrogram, noise_scale[n]).squeeze(1))
            if n > 0:
                noise = torch.randn_like(audio)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)
    return audio.cpu()

def log_mel_mse(x, y):
    """Compute log-mel-MSE between two waveforms."""
    pad_left  = 874
    pad_right = 874

    x = Fk.pad(x, (pad_left, pad_right))
    y = Fk.pad(y, (pad_left, pad_right))

    x = x.to(device)
    y = y.to(device)

    X = torch.log(mel(x) + 1e-5)
    Y = torch.log(mel(y) + 1e-5)
    return torch.mean((X - Y)**2).item()

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_and_save_metrics(df, class_names, save_dir="metrics"):
    os.makedirs(save_dir, exist_ok=True)

    y_true = df["true_class"].values
    y_pred = df["predicted_class"].values
    n_classes = len(class_names)

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred)

    # -----------------------------
    # BASIC METRICS
    # -----------------------------
    accuracy = accuracy_score(y_true, y_pred)
    error_rate = 1 - accuracy

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # -----------------------------
    # PER-CLASS METRICS
    # -----------------------------
    metrics_per_class = {}

    for i, cls in enumerate(class_names):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)

        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0

        metrics_per_class[cls] = {
            "precision": precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0),
            "specificity": specificity,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr
        }

    # -----------------------------
    # ROC & PR CURVES (OvR)
    # -----------------------------
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    roc_auc = {}
    plt.figure(figsize=(7, 6))

    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]

    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], (y_pred == i).astype(int))
        roc_auc[cls] = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linestyle=linestyles[i],
            marker=markers[i],
            markevery=max(len(fpr) // 10, 1),
            linewidth=2,
            alpha=0.85,
            label=f"{cls} (AUC={roc_auc[cls]:.2f})",
            zorder=10 - i
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300)
    plt.close()


    # Precision–Recall Curves (with overlapping visibility)
    plt.figure(figsize=(7, 6))

    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]

    for i, cls in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i], (y_pred == i).astype(int)
        )

        plt.plot(
            recall,
            precision,
            linestyle=linestyles[i],
            marker=markers[i],
            markevery=max(len(recall) // 10, 1),
            linewidth=2,
            alpha=0.85,
            label=cls,
            zorder=10 - i
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (One-vs-Rest)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"), dpi=300)
    plt.close()

    # -----------------------------
    # SAVE METRICS
    # -----------------------------
    metrics_summary = {
        "accuracy": accuracy,
        "error_rate": error_rate,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "per_class_metrics": metrics_per_class,
        "roc_auc": roc_auc
    }

    with open(os.path.join(save_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    # Save confusion matrix
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(save_dir, "confusion_matrix.csv"))

    print("📊 All evaluation metrics saved successfully.")
    return metrics_summary, cm

# -----------------------------
# CLASSIFICATION FUNCTION
# -----------------------------
def classify(test_data_dir, models):
    results = []

    for folder in os.listdir(test_data_dir):
        folder_path = os.path.join(test_data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder[0].lower() not in class_mapping:
            continue
        true_class = class_mapping[folder[0].lower()]
        wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        print(f"🎧 Found {len(wav_files)} WAV files in {folder}")

        for wav_file in tqdm(wav_files, desc=f"Processing {folder}"):
            wav_path = os.path.join(folder_path, wav_file)
            npy_path = wav_path.replace(".wav", ".wav.spec.npy")
            if not os.path.exists(npy_path):
                print(f"⚠️ Missing spectrogram for {wav_file}, skipping.")
                continue

            spectrogram = torch.from_numpy(np.load(npy_path))
            generated_audio = {}
            for cls_name, model in models.items():
                generated_audio[cls_name] = predict_audio(spectrogram, model)

            original_waveform, sr_o_loaded = T.load(wav_path)
            min_len = min([original_waveform.shape[-1]] + [g.shape[-1] for g in generated_audio.values()])
            original_waveform = original_waveform[..., :min_len]

            mse_scores = {}
            for idx, cls_name in enumerate(class_names):
                g_wave = generated_audio[cls_name][..., :min_len]
                mse_scores[idx] = log_mel_mse(original_waveform, g_wave)

            predicted_class = min(mse_scores, key=mse_scores.get)

            results.append({
                "sample": wav_file,
                "true_class": true_class,
                "predicted_class": predicted_class,
                **{f"MSE_{name}": mse_scores[i] for i, name in enumerate(class_names)}
            })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Overall accuracy
    overall_acc = accuracy_score(df["true_class"], df["predicted_class"])
    print(f"\n✅ Overall Accuracy: {overall_acc*100:.2f}%")

    # Per-class accuracy
    per_class_acc = df.groupby("true_class").apply(lambda x: (x["predicted_class"] == x["true_class"]).mean())
    for cls_idx, acc in per_class_acc.items():
        print(f"Class {class_names[cls_idx]} Accuracy: {acc*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(df["true_class"], df["predicted_class"])
    return df, cm, overall_acc

def plot_confusion_matrix(cm, class_names, save_path=None):
    cm = np.array(cm)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion", rotation=270, labelpad=15)

    # Ticks & labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, pad=12)

    # Annotate cells (count + percentage)
    # Annotate cells (counts only)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if cm_norm[i, j] > 0.5 else "black"
            )


    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--model_norm", required=True)
    parser.add_argument("--model_ball", required=True)
    parser.add_argument("--model_inner", required=True)
    parser.add_argument("--model_outer", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()

    device = args.device
    #test_data_dir= r"C:\Users\yosef\wavegrad\wavegrad\final\dataset\norm_wav\test(10dB)"
    #model_norm= r"C:\Users\yosef\wavegrad\wavegrad\final\trained_models\normal\-model (batch= 1, cosine_NS_200_steps, 2e-4_LR, epochs= 160) Padded-norm.-wav\weights-160.pt"
    #model_ball= r"C:\Users\yosef\wavegrad\wavegrad\final\trained_models\Ball\-model (batch= 1, cosine_NS_200_steps, 2e-4_LR, epochs= 160) Padded-norm.-wav\weights-160.pt"
    #model_inner= r"C:\Users\yosef\wavegrad\wavegrad\final\trained_models\inner_race\-model (batch= 1, cosine_NS_200_steps, 2e-4_LR, epochs= 160) Padded-norm.-wav\weights-160.pt"
    #model_outer= r"C:\Users\yosef\wavegrad\wavegrad\final\trained_models\outer_race\model (batch= 1, cosine_NS_200_steps, 2e-4_LR, epochs= 160) Padded-norm.-wav\weights-160.pt"
    test_data_dir= args.test_data_dir
    model_norm= args.model_norm
    model_ball= args.model_ball
    model_inner= args.model_inner
    model_outer= args.model_outer
    save_dir= args.save_dir
    # Custom noise schedules for each model
    noise_schedule_norm = 0.5*(1-np.cos(np.linspace(0,np.pi,200)))*(0.01-1e-6) + 1e-6
    noise_schedule_ball = 0.5*(1-np.cos(np.linspace(0,np.pi,200)))*(0.02-1e-6) + 1e-6
    noise_schedule_inner = 0.5*(1-np.cos(np.linspace(0,np.pi,200)))*(0.015-1e-6) + 1e-6
    noise_schedule_outer = 0.5*(1-np.cos(np.linspace(0,np.pi,200)))*(0.018-1e-6) + 1e-6

    # Load models
    models = {
        "normal": load_model(model_norm, device, noise_schedule_norm),
        "ball": load_model(model_ball, device, noise_schedule_ball),
        "inner": load_model(model_inner, device, noise_schedule_inner),
        "outer": load_model(model_outer, device, noise_schedule_outer)
    }

    #save_dir= r"C:\Users\yosef\OneDrive - GIU AS - German International University of Applied Sciences\Desktop\metrics (log_mel_MSE)\10dB noise"
    df, cm, overall_acc = classify(test_data_dir, models)

# Save classification results
df.to_excel("classification_results_logmelMSE.xlsx", index=False)

# Evaluate and save metrics
metrics_summary, cm = evaluate_and_save_metrics(df, class_names, save_dir)

print(f"\n📈 Overall Accuracy: {overall_acc*100:.2f}%")

plot_confusion_matrix(cm, class_names, save_dir)