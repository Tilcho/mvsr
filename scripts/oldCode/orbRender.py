import cv2
import numpy as np
import os

# === Einstellungen ===
RENDER_DIR = "renders"
FEATURES_OUT = "render_features.npz"
ORB_FEATURES = 500

# === ORB-Initialisierung ===
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)

# === Alle gerenderten Bilder laden ===
images = sorted([f for f in os.listdir(RENDER_DIR) if f.endswith(".png")])

all_keypoints = []
all_descriptors = []

print("Extrahiere ORB-Features aus gerenderten Bildern...")
for fname in images:
    path = os.path.join(RENDER_DIR, fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    kp, des = orb.detectAndCompute(img, None)
    all_keypoints.append(kp)
    all_descriptors.append(des)
    print(f"  -> {fname}: {len(kp)} Keypoints")

# === Deskriptoren speichern ===
# Hinweis: Keypoints selbst sind keine numpy-Arrays, wir speichern nur Deskriptoren
np.savez(FEATURES_OUT, descriptors=np.array(all_descriptors, dtype=object))
print(f"Feature-Datei gespeichert als: {FEATURES_OUT}")
