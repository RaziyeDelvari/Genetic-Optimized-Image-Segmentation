# Genetic-Optimized-Image-Segmentation

Image segmentation system for pomegranates using Genetic Algorithm optimization of HSV color space parameters. Developed as part of Artificial Intelligence course assignment.

## Overview

This project implements an automated pomegranate segmentation system that uses a Genetic Algorithm to find optimal HSV (Hue, Saturation, Value) thresholds. The algorithm evolves a population of parameter sets to maximize the Intersection over Union (IoU) score between predicted masks and ground truth.

## How It Works

The Genetic Algorithm optimizes four parameters:
- Hue Center: Central hue value for pomegranate color
- Hue Width: Tolerance range around the hue center (circular distance)
- Saturation Minimum: Minimum saturation threshold
- Value Minimum: Minimum value threshold

The segmentation pipeline:
1. Convert image from BGR to HSV color space
2. Apply color thresholding using optimized parameters
3. Apply morphological operations (opening and closing) to remove noise
4. Generate binary mask
