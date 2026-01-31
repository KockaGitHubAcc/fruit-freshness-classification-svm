import cv2
import numpy as np
import matplotlib
matplotlib.use("macosx")  # mac-friendly backend
import matplotlib.pyplot as plt



def extract_hsv_histograms(X_rgb, bins=(8, 8, 8)):
    """
    Extract 3D HSV histograms for all images.
    Also prints histogram stats and supports visualization.
    """

    features = []

    print("\n=== EXTRACTING HSV COLOR HISTOGRAM FEATURES ===\n")

    for i, img in enumerate(X_rgb):

        # 1. Convert RGB → HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # 2. Compute 3D histogram on H, S, V
        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],             # Channels: H, S, V
            None,
            bins,                  # (8,8,8)
            [0, 180, 0, 256, 0, 256]
        )

        # 3. Normalize histogram
        cv2.normalize(hist, hist)

        # 4. Flatten (8×8×8 → 512)
        features.append(hist.flatten())

        # Optional progress update
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(X_rgb)} images")

    features = np.array(features)

    print(f"\n✔ Extracted HSV histograms for {features.shape[0]} images.")
    print(f"Feature vector length per image: {features.shape[1]}")

    return features



# ============================
# EXTRA FUNCTIONS: PRINT + VISUALIZE HISTOGRAM
# ============================

def print_histogram_vector(X_hsv, index=0):
    """
    Print the histogram feature vector for a selected image.
    """
    print(f"\n=== Histogram Vector for Image {index} ===")
    print(X_hsv[index])
    print("Length:", len(X_hsv[index]))


def visualize_histogram(X_hsv, index=0):
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"\n=== VISUALIZING HISTOGRAM FOR IMAGE {index} ===")

    hist_3d = X_hsv[index].reshape(8, 8, 8)

    HS = np.sum(hist_3d, axis=2)
    HV = np.sum(hist_3d, axis=1)
    SV = np.sum(hist_3d, axis=0)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    axs[0].imshow(HS, cmap="hot")
    axs[0].set_title("H–S Projection")
    axs[0].set_xlabel("S bins")
    axs[0].set_ylabel("H bins")

    axs[1].imshow(HV, cmap="hot")
    axs[1].set_title("H–V Projection")

    axs[2].imshow(SV, cmap="hot")
    axs[2].set_title("S–V Projection")

    plt.tight_layout()
    plt.show()  # REQUIRED

def show_original_image(X_rgb, index=0):
    import matplotlib.pyplot as plt

    img = X_rgb[index]  # this is already a 128x128x3 numpy array

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title(f"Original Image #{index}")
    plt.axis("off")
    plt.show()

def pca_united_heatmap(X_hsv, index=0):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Flattened histogram (512-dim vector)
    hist_vector = X_hsv[index]

    # PCA expects a 2D array → reshape to (bins, features)
    # Here we reshape 512 → 64×8 grid
    hist_matrix = hist_vector.reshape(64, 8)

    # Apply PCA to reduce 8 dimensions → 1 dimension (so we get intensity)
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(hist_matrix)

    # Now we reshape the 64 resulting values back to an 8×8 grid
    united_pca_map = pca_result.reshape(8, 8)

    # Normalize for visualization
    united_pca_map = (united_pca_map - united_pca_map.min()) / (united_pca_map.max() - united_pca_map.min())

    # Plot
    plt.figure(figsize=(5,5))
    plt.imshow(united_pca_map, cmap="hot")
    plt.title("PCA United Histogram Heatmap")
    plt.colorbar()
    plt.xlabel("PCA bin X")
    plt.ylabel("PCA bin Y")
    plt.show()





# ============================
# TEST BLOCK — RUN DIRECTLY
# ============================

if __name__ == "__main__":
    from ImageLoading import load_dataset

    ROOT_DIR = "/Users/Dimitrije/Documents/MachineVisionProject/Images"

    # Load the dataset
    X_rgb, X_gray, y, fruit_names = load_dataset(ROOT_DIR)

    # Extract features
    X_hsv = extract_hsv_histograms(X_rgb)

    # Print one histogram vector
    print_histogram_vector(X_hsv, index=0)

    # Visualize histogram as heatmaps
    visualize_histogram(X_hsv, index=3)

    show_original_image(X_rgb, index=3)     # show the image
  
    pca_united_heatmap(X_hsv, index=3)
