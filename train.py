import cv2
import numpy as np
import random

import os
print("CWD:", os.getcwd())



random.seed(42)
np.random.seed(42)


def load_training_data(image_paths, mask_paths):
    images, masks = [], []
    for ip, mp in zip(image_paths, mask_paths):
        img = cv2.imread(ip)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) // 255
        images.append(img)
        masks.append(mask)
    return images, masks



def segment_pomegranate(image, genome, use_blur=True):
    h_center, h_width, s_min, v_min = genome

    if use_blur:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Circular Hue distance because hue colors are on circle
    dh = np.abs(h.astype(np.int16) - h_center)
    dh = np.minimum(dh, 180 - dh)

    mask = (
        (dh <= h_width) &
        (s >= s_min) &
        (v >= v_min)
    )

    mask = mask.astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask // 255



def iou_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 0.0

    return intersection / union


def fitness(genome, images, masks, k=3):
    idxs = np.random.choice(len(images), k, replace=False)
    scores = []

    for i in idxs:
        pred = segment_pomegranate(images[i], genome)
        scores.append(iou_score(pred, masks[i]))

    return np.mean(scores)


POP_SIZE = 20
GENERATIONS = 30
MUTATION_RATE = 0.15


def random_genome():
    return [
        random.randint(0, 180),   # h_center
        random.randint(5, 60),    # h_width
        random.randint(0, 255),   # s_min
        random.randint(0, 255),   # v_min
    ]


def crossover(p1, p2):
    pt = random.randint(1, len(p1) - 1)
    return p1[:pt] + p2[pt:]


def mutate(g):
    for i in range(len(g)):
        if random.random() < MUTATION_RATE:
            g[i] += random.randint(-10, 10)

            if i == 0:          # Hue center
                g[i] %= 180
            elif i == 1:        # Hue width
                g[i] = max(1, min(g[i], 90))
            else:               # S, V
                g[i] = max(0, min(g[i], 255))

    return g


def genetic_algorithm(images, masks):
    population = [random_genome() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        scored = [(fitness(g, images, masks), g) for g in population]
        scored.sort(reverse=True, key=lambda x: x[0])

        print(f"Generation {gen:02d} | Best IoU: {scored[0][0]:.4f}")

        # Elitism
        new_pop = [scored[0][1][:], scored[1][1][:]]

        while len(new_pop) < POP_SIZE:
            p1 = random.choice(scored[:5])[1]
            p2 = random.choice(scored[:5])[1]
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)

        population = new_pop

    return scored[0][1]



def show_results(images, masks, genome):
    for i, (img, gt) in enumerate(zip(images, masks)):
        pred = segment_pomegranate(img, genome)
        iou = iou_score(pred, gt)

        print(f"Image {i+1} | IoU: {iou:.4f}")

        cv2.imshow("Original", img)
        cv2.imshow("Ground Truth", gt * 255)
        cv2.imshow("Prediction", pred * 255)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    train_images = [
        "2.png", "3.png", "4.png",
        "5.jpg", "6.jpg", "7.jpg"
    ]

    train_masks = [
        "gt2.png", "gt3.png", "gt4.png",
        "gt5.png", "gt6.png", "gt7.png"
    ]

    images, masks = load_training_data(train_images, train_masks)

    best_genome = genetic_algorithm(images, masks)
    print("\nBest Genome [h_center, h_width, s_min, v_min]:")
    print(best_genome)

    np.save("best_genome.npy", np.array(best_genome))
    print("Best genome saved to best_genome.npy")

    show_results(images, masks, best_genome)
