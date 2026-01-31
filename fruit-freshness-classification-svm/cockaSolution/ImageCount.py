import os

BASE_DIR = "/Users/Dimitrije/Documents/MachineVisionProject/Images"

FRUITS = ["green_apple", "red_apple", "orange", "banana", "kiwi"]

def count_images():
    total_images = 0

    print("=== IMAGE COUNT CHECK ===\n")

    for fruit in FRUITS:
        fruit_path = os.path.join(BASE_DIR, fruit)

        fresh_path = os.path.join(fruit_path, "fresh")
        rotten_path = os.path.join(fruit_path, "rotten")

        # count files in each folder
        fresh_count = len([
            f for f in os.listdir(fresh_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        rotten_count = len([
            f for f in os.listdir(rotten_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        total = fresh_count + rotten_count
        total_images += total

        print(f"{fruit.upper()}:")
        print(f"  Fresh :  {fresh_count}")
        print(f"  Rotten: {rotten_count}")
        print(f"  Total :  {total}\n")

        # warnings for missing data
        if fresh_count == 0:
            print(f"⚠️ WARNING: No fresh images found for {fruit}!")
        if rotten_count == 0:
            print(f"⚠️ WARNING: No rotten images found for {fruit}!")

    print("=====================================")
    print(f"TOTAL IMAGES DOWNLOADED: {total_images}")
    print("=====================================")


if __name__ == "__main__":
    count_images()
