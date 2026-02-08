#versione ancora vecchia, moremore è aggironato

import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "dataset_greek_m"
LETTERS = ["PHI", "PSI", "OMEGA"]

SAMPLES_PER_CLASS = 5000

CANVAS = 220
FINAL_SIZE = 128

FONT_PATH = "arial.ttf"

def generate_letter(letter):
    img = Image.new("L", (CANVAS, CANVAS), 255)
    draw = ImageDraw.Draw(img)

    base_size = random.randint(120, 170)
    font = ImageFont.truetype(FONT_PATH, base_size)

    bbox = draw.textbbox((0, 0), letter, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = (CANVAS - w) // 2 + random.randint(-40, 40)
    y = (CANVAS - h) // 2 + random.randint(-40, 40)

    draw.text((x, y), letter, fill=0, font=font)

    img = np.array(img).astype(np.float32)

    angle = random.uniform(-25, 25)
    scale = random.uniform(0.75, 1.25)
    M = cv2.getRotationMatrix2D((CANVAS//2, CANVAS//2), angle, scale)
    img = cv2.warpAffine(img, M, (CANVAS, CANVAS), borderValue=255)

    tx = random.randint(-30, 30)
    ty = random.randint(-30, 30)
    M = np.float32([[1,0,tx],[0,1,ty]])
    img = cv2.warpAffine(img, M, (CANVAS, CANVAS), borderValue=255)

    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-20, 20)
    img = img * alpha + beta
    img = np.clip(img, 0, 255)

    if random.random() < 0.5:
        k = random.choice([3,5])
        img = cv2.GaussianBlur(img, (k,k), 0)

    noise = np.random.normal(0, random.randint(3, 15), img.shape)
    img = img + noise
    img = np.clip(img, 0, 255)

    pts1 = np.float32([[0,0],[CANVAS,0],[0,CANVAS],[CANVAS,CANVAS]])
    shift = random.randint(0, 20)
    pts2 = np.float32([
        [random.randint(0,shift), random.randint(0,shift)],
        [CANVAS-random.randint(0,shift), random.randint(0,shift)],
        [random.randint(0,shift), CANVAS-random.randint(0,shift)],
        [CANVAS-random.randint(0,shift), CANVAS-random.randint(0,shift)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (CANVAS, CANVAS), borderValue=255)

    img = cv2.resize(img, (FINAL_SIZE, FINAL_SIZE))
    return img

for letter in LETTERS:
    folder = os.path.join(OUTPUT_DIR, letter)
    os.makedirs(folder, exist_ok=True)

    for i in range(SAMPLES_PER_CLASS):
        img = generate_letter(letter)
        cv2.imwrite(os.path.join(folder, f"{letter}_{i}.png"), img)
