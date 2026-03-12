#!/usr/bin/env python3
# Generatore dataset sintetico per lettere viste da camera molto vicina (4–6 cm)
# VERSIONE SUPER ROBUSTA (stile easyocr-like)
# - Rotazione fino a ±50°
# - Traslazioni forti (anche tagli netti)
# - Prospettiva, ombre, riflessi, vignetta, blur, rumore, JPEG
# - Controllo per evitare immagini quasi vuote
# - Classe EMPTY con le stesse condizioni sporche delle altre

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
TARGET_PER_CLASS = 12000  # immagini per classe (puoi aumentare se vuoi)

LABELS = ["OMEGA", "PHI", "PSI", "EMPTY"]
GLYPHS = {"OMEGA": "Ω", "PHI": "Φ", "PSI": "Ψ"}


# ============================
# RENDER LETTERA GRANDE (camera vicina)
# ============================

def render_letter(letter, size=200):
    """
    Disegna la lettera in grande (simile a come appare a 4–6 cm).
    Font fisso, sfondo bianco.
    """
    canvas = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(canvas)

    font = ImageFont.truetype(FONT_FILE, int(size * 0.75))

    bbox = draw.textbbox((0, 0), letter, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    draw.text(((size - w) // 2, (size - h) // 2), letter, fill=0, font=font)

    return np.array(canvas)


# ============================
# UTILS AUGMENTATION
# ============================

def add_global_brightness_contrast(img):
    """Jitter globale di luminosità e contrasto."""
    alpha = random.uniform(0.8, 1.3)  # contrasto
    beta = random.uniform(-25, 25)    # luminosità
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def add_gaussian_blur_or_motion(img):
    """Blur leggero: gaussiano o motion blur."""
    if random.random() < 0.5:
        # Gaussian blur
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), sigmaX=random.uniform(0.3, 1.2))
    else:
        # Motion blur
        k = random.choice([3, 5, 7])
        kernel = np.zeros((k, k), dtype=np.float32)
        if random.random() < 0.5:
            kernel[k // 2, :] = 1.0
        else:
            kernel[:, k // 2] = 1.0
        kernel /= kernel.sum()
        return cv2.filter2D(img, -1, kernel)


def add_reflections(img):
    """
    Simula riflessi: spot molto chiari, tipo flare.
    Sfondo resta concettualmente bianco, ma con zone bruciate.
    """
    h, w = img.shape
    out = img.astype(np.float32)

    num_spots = random.randint(1, 3)
    for _ in range(num_spots):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        max_radius = random.randint(int(0.1 * min(h, w)), int(0.35 * min(h, w)))

        Y, X = np.ogrid[:h, :w]
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        sigma2 = (max_radius ** 2) / random.uniform(1.5, 3.0)
        spot = np.exp(-dist2 / (2 * sigma2))

        intensity = random.uniform(0.4, 0.9)  # quanto si avvicina al bianco pieno
        # Blend verso il bianco (255)
        out = out * (1 - intensity * spot) + 255.0 * (intensity * spot)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def ensure_not_too_empty(img, min_black_ratio=0.002):
    """
    Controlla che l'immagine non sia quasi completamente bianca.
    Se troppo vuota, ritorna False.
    """
    dark = (img < 220).sum()
    ratio = dark / img.size
    return ratio >= min_black_ratio


# ============================
# CAMERA-LIKE TRANSFORM (Super RCJ Maze)
# ============================

def camera_like_transform(img, _depth=0):
    """
    Simula condizioni estreme del labirinto RCJ:
    - Traslazione orizzontale forte (anche tagli netti)
    - Traslazione verticale ampia
    - Rotazione fino a ±50°
    - Prospettiva, Ombre, Riflessi, Vignetta, Blur, Rumore, JPEG
    """
    h0, w0 = img.shape

    # 1) Variazione di scala (70% - 105%) per dare spazio allo spostamento
    scale_target = random.uniform(0.7, 1.05)
    scale = min(FINAL_SIZE / w0, FINAL_SIZE / h0, 1.0) * scale_target
    new_w = int(w0 * scale)
    new_h = int(h0 * scale)

    img2 = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2) Canvas bianco 128x128
    canvas = np.full((FINAL_SIZE, FINAL_SIZE), 255, dtype=np.uint8)
    H, W = canvas.shape
    start_x = (FINAL_SIZE - new_w) // 2
    start_y = (FINAL_SIZE - new_h) // 2

    # --- COPIA SICURA CON CLIPPING (può tagliare la lettera) ---
    dst_x1 = max(0, start_x)
    dst_y1 = max(0, start_y)
    dst_x2 = min(W, start_x + new_w)
    dst_y2 = min(H, start_y + new_h)

    src_x1 = dst_x1 - start_x
    src_y1 = dst_y1 - start_y
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    if dst_x1 < dst_x2 and dst_y1 < dst_y2:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img2[src_y1:src_y2, src_x1:src_x2]

    out = canvas

    # 3) Rotazione e Traslazione Asimmetrica (RCJ Style)
    # Generiamo il 75% delle immagini relativamente dritte e centrate,
    # e il restante 25% in condizioni estreme
    if random.random() < 0.75:
        angle = random.gauss(0, 7.0)  # Maggior parte tra -7° e +7°, massimo raro
        angle = max(-15.0, min(15.0, angle))  # Tagliamo per sicurezza a max ±15°
        shift_x = random.randint(-20, 20)  # Spostamenti leggeri
        shift_y = random.randint(-15, 15)
    else:
        angle = random.uniform(-50.0, 50.0)  # Rotazione forte
        shift_x = random.randint(-60, 60)  # Traslazioni ampie (potenziale taglio)
        shift_y = random.randint(-50, 50)

    M_transform = cv2.getRotationMatrix2D((FINAL_SIZE / 2, FINAL_SIZE / 2), angle, 1.0)
    M_transform[0, 2] += shift_x
    M_transform[1, 2] += shift_y

    out = cv2.warpAffine(out, M_transform, (FINAL_SIZE, FINAL_SIZE), borderValue=255)

    # 4) Prospettiva leggera
    if random.random() < 0.7:
        h, w = out.shape

        def jitter(pt):
            return (pt[0] + random.uniform(-0.06 * w, 0.06 * w),
                    pt[1] + random.uniform(-0.06 * h, 0.06 * h))

        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([jitter(p) for p in src])
        M_persp = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, M_persp, (w, h), borderValue=255)

    # 5) Ombre LED
    if random.random() < 0.85:
        h, w = out.shape
        ys, xs = np.indices((h, w))
        cx, cy = w / 2, h / 2
        angle_shadow = np.deg2rad(random.choice([20, 45, 70, 110, 140, 200, 250, 300]))
        vx, vy = np.cos(angle_shadow), np.sin(angle_shadow)
        proj = (xs - cx) * vx + (ys - cy) * vy
        proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
        shadow = 1.0 - proj_norm
        mask = 1.0 - random.uniform(0.15, 0.4) * shadow
        out = np.clip(out.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    # 6) Riflessi
    if random.random() < 0.7:
        out = add_reflections(out)

    # 7) Vignettatura
    if random.random() < 0.5:
        h, w = out.shape
        X = np.linspace(-1, 1, w)
        Y = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(X, Y)
        radius = np.sqrt(xv ** 2 + yv ** 2)
        mask = 1 - random.uniform(0.10, 0.35) * (radius ** 2)
        mask = np.clip(mask, 0.4, 1.0)
        out = np.clip(out.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    # 8) Jitter luminosità/contrasto globale
    if random.random() < 0.9:
        out = add_global_brightness_contrast(out)

    # 9) Blur (motion o gaussiano)
    if random.random() < 0.7:
        out = add_gaussian_blur_or_motion(out)

    # 10) Rumore sensore
    gauss = np.random.normal(0, random.uniform(1, 8), out.shape)
    out = np.clip(out.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

    # 11) JPEG artifacts
    pil = Image.fromarray(out)
    from io import BytesIO
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=random.randint(60, 90))
    buf.seek(0)
    out = np.array(Image.open(buf).convert("L"))

    # 12) Controllo: se troppo vuota, rigenera una volta (limita ricorsione)
    if not ensure_not_too_empty(out) and _depth < 1:
        return camera_like_transform(img, _depth=_depth + 1)

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
                # EMPTY con sfondo bianco ma sottoposto a tutte le trasformazioni
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
    print("Generazione dataset sintetico super-robusto per RoboCup Junior Maze...")
    generate_dataset()
    print("Fatto.")


if __name__ == "__main__":
    main()