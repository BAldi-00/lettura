import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = "dataset_greek"

CLASSES = {
    "PHI": "Φ",
    "PSI": "Ψ",
    "OMEGA": "Ω"
}

IMAGES_PER_CLASS = 2000
FINAL_SIZE = 128
WORK_SIZE = 256   # canvas grande per evitare tagli
FONT_PATH = "arial.ttf"
BASE_FONT_SIZE = 200

def create_scaled(symbol, font_size):
    font = ImageFont.truetype(FONT_PATH, font_size)
    img = Image.new("RGB", (WORK_SIZE, WORK_SIZE), "white")
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), symbol, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = (WORK_SIZE - w) // 2
    y = (WORK_SIZE - h) // 2

    draw.text((x, y), symbol, fill="black", font=font)
    return img

def augment(img):
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, expand=False, fillcolor="white")

    shift_x = random.randint(-25, 25)
    shift_y = random.randint(-25, 25)
    img = img.transform(
        img.size,
        Image.AFFINE,
        (1, 0, shift_x, 0, 1, shift_y),
        fillcolor="white"
    )

    arr = np.array(img).astype(np.int16)
    noise = np.random.normal(0, 8, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    return img

def generate_dataset():
    for folder_name, greek_symbol in CLASSES.items():
        class_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(IMAGES_PER_CLASS):
            scale_factor = random.uniform(0.60, 1.00)
            font_size = int(BASE_FONT_SIZE * scale_factor)

            img = create_scaled(greek_symbol, font_size)

            if random.random() < 0.7:
                img = augment(img)

            img = img.resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS)

            img.save(os.path.join(class_dir, f"{folder_name}_{i}.png"))

    print("Dataset generato senza tagli.")

generate_dataset()
