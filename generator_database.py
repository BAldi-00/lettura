#!/usr/bin/env python3
# Generatore dataset sintetico per lettere viste da camera molto vicina (4–6 cm)
# OTTIMIZZATO PER ROBOCUP JUNIOR MAZE:
# - Traslazione orizzontale marcata (simulazione scorrimento sui muri)
# - Traslazione verticale leggera
# - Rotazione +-10 gradi
# - Ombre, prospettiva, rumore

import os, random, shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ============================
# CONFIG
# ============================

OUT_DIR = Path("synthetic_dataset")  # MANTENUTO CORRETTO COME RICHIESTO
FONT_FILE = "arial.ttf"  # deve contenere Ω Φ Ψ
FINAL_SIZE = 128  # input ML
TARGET_PER_CLASS = 8000  # immagini per classe

LABELS = ["OMEGA", "PHI", "PSI", "EMPTY"]
GLYPHS = {"OMEGA": "Ω", "PHI": "Φ", "PSI": "Ψ"}


# ============================
# RENDER LETTERA GRANDE (camera vicina)
# ============================

def render_letter(letter, size=200):
    """
    Disegna la lettera in grande (simile a come appare a 4–6 cm).
    """
    canvas = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(canvas)

    font = ImageFont.truetype(FONT_FILE, int(size * 0.75))

    # textbbox è compatibile con Pillow moderno
    bbox = draw.textbbox((0, 0), letter, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    draw.text(((size - w) // 2, (size - h) // 2), letter, fill=0, font=font)

    return np.array(canvas)


# ============================
# CAMERA-LIKE TRANSFORM (Ottimizzata RCJ Maze)
# ============================

def camera_like_transform(img):
    """
    Simula le condizioni del labirinto RCJ:
    - Traslazione orizzontale forte, verticale debole
    - Rotazione leggera (+- 10 gradi)
    - Prospettiva, Ombre LED, Vignetta, Rumore
    """
    h0, w0 = img.shape

    # 1) Variazione di scala leggera (80% - 100%) per dare spazio allo spostamento
    scale_target = random.uniform(0.8, 1.0)
    scale = min(FINAL_SIZE / w0, FINAL_SIZE / h0, 1.0) * scale_target
    new_w = int(w0 * scale)
    new_h = int(h0 * scale)

    img2 = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2) Posizionamento iniziale al centro del canvas 128x128
    canvas = np.full((FINAL_SIZE, FINAL_SIZE), 255, dtype=np.uint8)
    start_x = (FINAL_SIZE - new_w) // 2
    start_y = (FINAL_SIZE - new_h) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = img2

    # 3) Rotazione e Traslazione Asimmetrica (RCJ Style)
    angle = random.uniform(-10.0, 10.0)  # Rotazione +- 10 gradi
    shift_x = random.randint(-35, 35)  # Traslazione Orizzontale marcata
    shift_y = random.randint(-8, 8)  # Traslazione Verticale leggera

    # Creiamo la matrice di rotazione
    M_transform = cv2.getRotationMatrix2D((FINAL_SIZE / 2, FINAL_SIZE / 2), angle, 1.0)
    # Aggiungiamo la traslazione alla matrice di rotazione
    M_transform[0, 2] += shift_x
    M_transform[1, 2] += shift_y

    # Applichiamo rotazione e traslazione insieme (borderValue=255 mantiene il bianco fuori dai bordi)
    out = cv2.warpAffine(canvas, M_transform, (FINAL_SIZE, FINAL_SIZE), borderValue=255)

    # 4) Prospettiva leggera (simula la telecamera non perfettamente parallela al muro)
    if random.random() < 0.5:
        h, w = out.shape

        def jitter(pt):
            return (pt[0] + random.uniform(-0.04 * w, 0.04 * w),
                    pt[1] + random.uniform(-0.04 * h, 0.04 * h))

        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([jitter(p) for p in src])
        M_persp = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, M_persp, (w, h), borderValue=255)

    # 5) Ombre LED (frequenti nel maze a causa delle zone d'ombra dei muri)
    if random.random() < 0.85:
        h, w = out.shape
        ys, xs = np.indices((h, w))
        cx, cy = w / 2, h / 2
        angle_shadow = np.deg2rad(random.choice([20, 45, 70, 110, 140, 200, 250, 300]))
        vx, vy = np.cos(angle_shadow), np.sin(angle_shadow)
        proj = (xs - cx) * vx + (ys - cy) * vy
        proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
        shadow = 1.0 - proj_norm
        mask = 1.0 - random.uniform(0.10, 0.35) * shadow
        out = np.clip(out.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    # 6) Vignettatura (bordo scuro della telecamera)
    if random.random() < 0.4:
        h, w = out.shape
        X = np.linspace(-1, 1, w)
        Y = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(X, Y)
        radius = np.sqrt(xv ** 2 + yv ** 2)
        mask = 1 - random.uniform(0.10, 0.35) * (radius ** 2)
        mask = np.clip(mask, 0.5, 1.0)
        out = np.clip(out.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    # 7) Rumore Sensore
    gauss = np.random.normal(0, random.uniform(1, 5), out.shape)
    out = np.clip(out.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

    # 8) JPEG artifacts (simula la compressione stream della cam)
    pil = Image.fromarray(out)
    from io import BytesIO
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=random.randint(70, 90))
    buf.seek(0)
    out = np.array(Image.open(buf).convert("L"))

    return out


# ============================
# GENERAZIONE DATASET
# ============================

def generate_dataset():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for lab in LABELS:
        (OUT_DIR / lab).mkdir(parents=True, exist_ok=True)

    idx = {lab: 0 for lab in LABELS}

    for lab in LABELS:
        print(f"Generazione classe {lab}...")
        for _ in range(TARGET_PER_CLASS):

            if lab == "EMPTY":
                base = np.full((200, 200), 255, dtype=np.uint8)
            else:
                base = render_letter(GLYPHS[lab], size=200)

            patch = camera_like_transform(base)

            out_name = OUT_DIR / lab / f"{lab}_{idx[lab]:06d}.png"
            cv2.imwrite(str(out_name), patch)
            idx[lab] += 1

    print("Dataset generato in:", OUT_DIR)


# ============================
# MAIN
# ============================

def main():
    print("Generazione dataset sintetico per RoboCup Junior Maze...")
    generate_dataset()
    print("Fatto.")


if __name__ == "__main__":
    main()