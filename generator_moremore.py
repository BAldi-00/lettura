import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "dataset_greek_mm"

# Simboli greci reali
LETTERS = {
    "PHI": "Φ",
    "PSI": "Ψ",
    "OMEGA": "Ω"
}

SAMPLES_PER_CLASS = 7000

CANVAS = 260
FINAL_SIZE = 128

FONT_PATH = "arial.ttf"

def generate_letter(symbol):
    img = Image.new("L", (CANVAS, CANVAS), 255)
    draw = ImageDraw.Draw(img)

    base_size = random.randint(130, 180)
    font = ImageFont.truetype(FONT_PATH, base_size)

    bbox = draw.textbbox((0, 0), symbol, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Spostamento ridotto per evitare tagli
    x = (CANVAS - w) // 2 + random.randint(-25, 25)
    y = (CANVAS - h) // 2 + random.randint(-25, 25)

    draw.text((x, y), symbol, fill=0, font=font)

    img = np.array(img).astype(np.float32)

    # Rotazione e scala moderate
    angle = random.uniform(-20, 20)
    scale = random.uniform(0.85, 1.15)
    M = cv2.getRotationMatrix2D((CANVAS//2, CANVAS//2), angle, scale)
    img = cv2.warpAffine(img, M, (CANVAS, CANVAS), borderValue=255)

    # Traslazione ridotta
    tx = random.randint(-20, 20)
    ty = random.randint(-20, 20)
    M = np.float32([[1,0,tx],[0,1,ty]])
    img = cv2.warpAffine(img, M, (CANVAS, CANVAS), borderValue=255)

    # Contrasto più controllato
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-15, 15)
    img = img * alpha + beta
    img = np.clip(img, 0, 255)

    # Blur leggero e meno frequente
    if random.random() < 0.35:
        img = cv2.GaussianBlur(img, (3,3), 0)

    # Rumore moderato
    noise = np.random.normal(0, random.randint(3, 10), img.shape)
    img = img + noise
    img = np.clip(img, 0, 255)

    # Prospettiva molto ridotta per evitare tagli
    if random.random() < 0.3:
        shift = random.randint(3, 10)
        pts1 = np.float32([[0,0],[CANVAS,0],[0,CANVAS],[CANVAS,CANVAS]])
        pts2 = np.float32([
            [random.randint(0,shift), random.randint(0,shift)],
            [CANVAS-random.randint(0,shift), random.randint(0,shift)],
            [random.randint(0,shift), CANVAS-random.randint(0,shift)],
            [CANVAS-random.randint(0,shift), CANVAS-random.randint(0,shift)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (CANVAS, CANVAS), borderValue=255)

    # Zoom sicuro
    zoom = random.uniform(0.9, 1.15)
    center = CANVAS // 2
    half = int(center / zoom)

    x1 = max(0, center - half)
    y1 = max(0, center - half)
    x2 = min(CANVAS, center + half)
    y2 = min(CANVAS, center + half)

    if x2 - x1 < 40 or y2 - y1 < 40:
        return generate_letter(symbol)

    img = img[y1:y2, x1:x2]
    img = cv2.resize(img, (FINAL_SIZE, FINAL_SIZE))

    return img.astype(np.uint8)

for name, symbol in LETTERS.items():
    folder = os.path.join(OUTPUT_DIR, name)
    os.makedirs(folder, exist_ok=True)

    for i in range(SAMPLES_PER_CLASS):
        img = generate_letter(symbol)
        cv2.imwrite(os.path.join(folder, f"{name}_{i}.png"), img)
