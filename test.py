import cv2
import numpy as np
from train import segment_pomegranate


def run_on_new_images(image_paths):

    genome = np.load("best_genome.npy").tolist()
    print("Loaded genome:", genome)

    for path in image_paths:
        img = cv2.imread(path)

        if img is None:
            print(f"Could not load image: {path}")
            continue

        pred_mask = segment_pomegranate(img, genome)
        mask_255 = (pred_mask * 255).astype("uint8")
        segmented = cv2.bitwise_and(img, img, mask=mask_255)

        cv2.imshow("Original Image", img)
        cv2.imshow("Predicted Mask", mask_255)
        cv2.imshow("Segmented Result", segmented)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_images = [
        "test4.jpg",
        "test2.jpg",
        "test5.jpg"
    ]

    run_on_new_images(test_images)
