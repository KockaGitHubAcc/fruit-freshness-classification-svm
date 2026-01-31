import os
import numpy as np
from PIL import Image

ROOT_DIR = "/Users/Dimitrije/Documents/MachineVisionProject/Images"
IMAGE_SIZE = (128, 128)

label_map = {"fresh": 0, "rotten": 1}

def load_dataset(root_dir):
    rgb_images = []
    gray_images = []
    labels = []
    fruits = []

    print("\n=== LOADING ALL IMAGES INTO NUMPY ARRAYS ===\n")

    for fruit in os.listdir(root_dir):
        fruit_path = os.path.join(root_dir, fruit)
        if not os.path.isdir(fruit_path):
            continue

        for cls in ["fresh", "rotten"]:
            cls_path = os.path.join(fruit_path, cls)
            if not os.path.isdir(cls_path):
                continue

            print(f"→ Loading {fruit}/{cls}")

            for filename in os.listdir(cls_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(cls_path, filename)

                    try:
                        img = Image.open(img_path).convert("RGB")
                        img = img.resize(IMAGE_SIZE)

                        rgb = np.array(img)
                        gray = np.array(img.convert("L"))

                        rgb_images.append(rgb)
                        gray_images.append(gray)
                        labels.append(label_map[cls])
                        fruits.append(fruit)

                    except Exception as e:
                        print(f"❌ Error loading {img_path}: {e}")

    print(f"\n✔ Loaded {len(rgb_images)} images successfully.\n")

    return (
        np.array(rgb_images),
        np.array(gray_images),
        np.array(labels),
        np.array(fruits)
    )


# Example use:
if __name__ == "__main__":
    ROOT_DIR = "/Users/Dimitrije/Documents/MachineVisionProject/Images"

    X_rgb, X_gray, y, fruits = load_dataset(ROOT_DIR)

    print("RGB images shape:", X_rgb.shape)
    print("Grayscale shape:", X_gray.shape)
    print("Labels shape:", y.shape)

    # CHECK LABEL BALANCE
    unique, counts = np.unique(y, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} images")

