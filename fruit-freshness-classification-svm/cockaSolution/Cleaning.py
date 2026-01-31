import os
from PIL import Image

ROOT_DIR = "/Users/Dimitrije/Documents/MachineVisionProject/Images"
TARGET_SIZE = (128, 128)

def resize_all_images(root_dir, target_size):
    resized = 0
    errors = []

    print("\n=== RESIZING ALL IMAGES TO 128×128 ===\n")

    for fruit in os.listdir(root_dir):
        fruit_path = os.path.join(root_dir, fruit)
        if not os.path.isdir(fruit_path):
            continue

        for cls in ["fresh", "rotten"]:
            cls_path = os.path.join(fruit_path, cls)
            if not os.path.isdir(cls_path):
                continue

            for filename in os.listdir(cls_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(cls_path, filename)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img = img.resize(target_size, Image.LANCZOS)
                        img.save(img_path)
                        resized += 1
                    except Exception as e:
                        errors.append((fruit, cls, filename, str(e)))

    print(f"✔ Successfully resized {resized} images.")
    if errors:
        print(f"⚠ Errors during resizing: {len(errors)}")
        for fruit, cls, filename, error in errors:
            print(f"  ERROR: {fruit}/{cls}/{filename} → {error}")
    else:
        print("✔ No errors detected during resizing.")

    return errors


def verify_images(root_dir):
    print("\n=== VERIFYING ALL IMAGES ===\n")

    broken = 0

    for fruit in os.listdir(root_dir):
        fruit_path = os.path.join(root_dir, fruit)
        if not os.path.isdir(fruit_path):
            continue

        for cls in ["fresh", "rotten"]:
            cls_path = os.path.join(fruit_path, cls)
            if not os.path.isdir(cls_path):
                continue

            for filename in os.listdir(cls_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(cls_path, filename)
                    try:
                        img = Image.open(img_path)
                        img.verify()  # checks file integrity
                    except Exception as e:
                        broken += 1
                        print(f"❌ BROKEN IMAGE → {fruit}/{cls}/{filename}")
                        print(f"    Error: {e}")

    if broken == 0:
        print("\n✔ All images verified — no corrupted files found!")
    else:
        print(f"\n⚠ Total corrupted images: {broken}")

def remove_duplicates(root_dir):
    hashes = {}      # store hash → path
    removed = 0

    print("\n=== REMOVING DUPLICATE IMAGES ===\n")

    for fruit in os.listdir(root_dir):
        fruit_path = os.path.join(root_dir, fruit)
        if not os.path.isdir(fruit_path):
            continue

        for cls in ["fresh", "rotten"]:
            cls_path = os.path.join(fruit_path, cls)
            if not os.path.isdir(cls_path):
                continue

            for filename in os.listdir(cls_path):
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                img_path = os.path.join(cls_path, filename)

                try:
                    img = Image.open(img_path)
                    h = imagehash.phash(img)  # perceptual hash

                    if h in hashes:
                        # duplicate found → remove it
                        os.remove(img_path)
                        removed += 1
                        print(f"❌ DUPLICATE REMOVED: {fruit}/{cls}/{filename}")
                        print(f"   Matches: {hashes[h]}")
                    else:
                        hashes[h] = img_path

                except Exception as e:
                    print(f"⚠ Error reading {img_path}: {e}")

    print(f"\n✔ Duplicate removal complete. Removed {removed} images.\n")


# Run it


# ==========================
# MAIN PROCESS
# ==========================
if __name__ == "__main__":
    resize_all_images(ROOT_DIR, TARGET_SIZE)
    verify_images(ROOT_DIR)
    print("\n🎉 PROCESS COMPLETE — DATASET CLEAN AND RESIZED\n")
    remove_duplicates(ROOT_DIR)