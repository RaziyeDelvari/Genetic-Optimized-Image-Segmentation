import cv2
import numpy as np
import os
from train import segment_pomegranate

def run_on_new_images(image_paths):
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    genome = np.load("best_genome.npy").tolist()
    print("Loaded genome:", genome)

    for path in image_paths:
        img = cv2.imread(path)

        if img is None:
            print(f"Could not load image: {path}")
            continue

        pred_mask = segment_pomegranate(img, genome)
        

        mask_255 = (pred_mask * 255).astype("uint8")


        filename = os.path.basename(path)
        name_without_ext = os.path.splitext(filename)[0]
        save_path = os.path.join(output_dir, f"{name_without_ext}.png")

 
        cv2.imwrite(save_path, mask_255)
        print(f"Saved mask to: {save_path}")


        segmented = cv2.bitwise_and(img, img, mask=mask_255)
        cv2.imshow("Original Image", img)
        cv2.imshow("Predicted Mask", mask_255)
        cv2.waitKey(500) 
    cv2.destroyAllWindows()
    print("\nAll masks saved. You can now run the evaluation script.")

if __name__ == "__main__":
    test_images = [
        "anar1.jpg",
        "anar2.jpg",     
    ]

    run_on_new_images(test_images)