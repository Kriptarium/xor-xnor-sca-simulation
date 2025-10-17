# XOR/XNOR Side-Channel Simulation
# Author: Fatih Özkaynak (2025)
# This script reproduces the profiling-based side-channel simulation for XOR/XNOR selection bit leakage.

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

INPUT_IMAGE = "cover.png"
OUT_DIR = "outputs"
IMG_SIZE = (32, 32)
LOGISTIC_R = 3.99
LOGISTIC_X0 = 0.3333
SEG_LEN = 160
NOISE_STD = 0.25
XNOR_EXTRA_AMP = 0.25
PROFILING_SAMPLES = 6000
PROFILING_SEED = 7
SIM_SEED = 42
np.random.seed(SIM_SEED)

def logistic_binary(length, r=LOGISTIC_R, x0=LOGISTIC_X0):
    x = x0
    out = []
    for _ in range(length):
        x = r * x * (1 - x)
        out.append(1 if x >= 0.5 else 0)
    return np.array(out, dtype=np.int8)

def image_to_bits(image_path, size=IMG_SIZE):
    img = Image.open(image_path).convert("L")
    img = img.resize(size, Image.BICUBIC)
    arr = np.array(img)
    th = arr.mean()
    bits = (arr.flatten() > th).astype(np.int8)
    return bits, arr

def make_segment(selection_bit, output_toggle, noise_std=NOISE_STD, seg_len=SEG_LEN):
    t = np.arange(seg_len)
    center = seg_len // 2
    base = np.exp(-0.5 * ((t - center) / 14.0)**2) + 0.6 * np.exp(-0.5 * ((t - center + 12) / 10.0)**2)
    base_amp = 1.0
    toggle_amp = 0.20 if output_toggle else 0.0
    amp = base_amp + (XNOR_EXTRA_AMP if selection_bit == 0 else 0.0) + toggle_amp
    seg = amp * base + np.random.normal(0, noise_std, size=seg_len)
    return seg

def main():
    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")
    os.makedirs(OUT_DIR, exist_ok=True)
    secret_bits, secret_arr = image_to_bits(INPUT_IMAGE, size=IMG_SIZE)
    ref_arr = secret_arr.copy().flatten()
    rng = np.random.RandomState(SIM_SEED)
    flip_idx = rng.choice(np.arange(ref_arr.size), size=int(0.05 * ref_arr.size), replace=False)
    ref_arr[flip_idx] = 255 - ref_arr[flip_idx]
    ref_arr = ref_arr.reshape(secret_arr.shape)
    ref_bits = (ref_arr.flatten() > ref_arr.mean()).astype(np.int8)
    N = secret_bits.size
    chaos_bits = logistic_binary(N, r=LOGISTIC_R, x0=LOGISTIC_X0)

    out_bits = np.zeros_like(secret_bits)
    for i, (s, rbit, cbit) in enumerate(zip(secret_bits, ref_bits, chaos_bits)):
        x = int(rbit ^ cbit)
        out_bits[i] = x if s == 1 else 1 - x

    segments = []
    prev = 0
    for o, s in zip(out_bits, secret_bits):
        output_toggle = (o != prev)
        seg = make_segment(s, output_toggle)
        segments.append(seg)
        prev = o
    segments = np.array(segments)

    rng = np.random.RandomState(PROFILING_SEED)
    X_profile = np.zeros((PROFILING_SAMPLES, SEG_LEN))
    y_profile = np.zeros(PROFILING_SAMPLES, dtype=np.int8)
    for i in range(PROFILING_SAMPLES):
        s = rng.randint(0, 2)
        toggle = rng.rand() < 0.5
        X_profile[i, :] = make_segment(s, toggle)
        y_profile[i] = s

    clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000))
    clf.fit(X_profile, y_profile)
    preds = clf.predict(segments)
    acc = accuracy_score(secret_bits, preds)
    cm = confusion_matrix(secret_bits, preds)

    secret_img = (secret_bits.reshape(IMG_SIZE) * 255).astype(np.uint8)
    ref_img = (ref_bits.reshape(IMG_SIZE) * 255).astype(np.uint8)
    out_img = (out_bits.reshape(IMG_SIZE) * 255).astype(np.uint8)
    Image.fromarray(secret_img).save(os.path.join(OUT_DIR, "secret_bits.png"))
    Image.fromarray(ref_img).save(os.path.join(OUT_DIR, "ref_bits.png"))
    Image.fromarray(out_img).save(os.path.join(OUT_DIR, "out_codes.png"))

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.concatenate(segments[:200]))
    plt.title("Trace snippet (first 200 operations)")
    plt.subplot(2, 1, 2)
    plt.step(range(200), secret_bits[:200], where='post', label='True bit')
    plt.step(range(200), preds[:200], where='post', label='Predicted bit')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "trace_snippet.png")); plt.close()

    mean_xor = X_profile[y_profile == 1].mean(axis=0)
    mean_xnor = X_profile[y_profile == 0].mean(axis=0)
    plt.figure(figsize=(6,3))
    plt.plot(mean_xor, label='XOR mean')
    plt.plot(mean_xnor, label='XNOR mean')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "templates.png")); plt.close()

    print(f"Accuracy: {acc*100:.2f}%")
    print("Confusion matrix:\n", cm)
    print(f"Outputs saved in {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
