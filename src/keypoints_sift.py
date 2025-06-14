
"""This script processes an image to detect keypoints using SIFT, computes their neighborhoods, gradients, and orientation histograms, and visualizes the results.
It uses OpenCV for image processing and Matplotlib for visualization.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

image_path = '../images/eiffel-tower-paris.png'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(gray, None)


def is_valid_keypoint(kp, img_shape, patch_size=16):
    y, x = int(kp.pt[1]), int(kp.pt[0])
    half_size = patch_size // 2
    return (half_size <= y < img_shape[0] - half_size and
            half_size <= x < img_shape[1] - half_size)


valid_keypoints = [kp for kp in keypoints if kp.pt[1] < gray.shape[0]/2 and is_valid_keypoint(kp, gray.shape)]


np.random.seed(0)
selected_keypoints = np.random.choice(valid_keypoints, 3, replace=False)


def draw_keypoint_neighborhood(img, keypoint, size=8):
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    y_start = max(0, y - size)
    y_end = min(img.shape[0], y + size)
    x_start = max(0, x - size)
    x_end = min(img.shape[1], x + size)
    return img[y_start:y_end, x_start:x_end]

def compute_gradients(neighborhood):
    gx = cv2.Sobel(neighborhood, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(neighborhood, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    return mag, angle

def create_orientation_histogram(mag, angle, num_bins=8):
    hist, _ = np.histogram(angle, bins=num_bins, range=(0, 360), weights=mag)
    return hist

img_with_kp = img.copy()
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
for kp, color in zip(selected_keypoints, colors):
    cv2.circle(img_with_kp, (int(kp.pt[0]), int(kp.pt[1])), 5, color, -1)

fig, axs = plt.subplots(6, 3, figsize=(18, 25))

axs[0, 0].imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Selected Keypoints on Image')
axs[0, 0].axis('off')

for j in range(1, 3):
    axs[0, j].axis('off')

for idx, kp in enumerate(selected_keypoints):
    neighborhood = draw_keypoint_neighborhood(gray, kp, size=8)

    mag, angle = compute_gradients(neighborhood)

    hist = create_orientation_histogram(mag, angle)

    axs[1, idx].imshow(neighborhood, cmap='gray')
    axs[1, idx].set_title(f'Keypoint {idx + 1} Neighborhood')
    axs[1, idx].axis('off')

    x_coords, y_coords = np.meshgrid(np.arange(0, neighborhood.shape[1]), np.arange(0, neighborhood.shape[0]))
    axs[2, idx].imshow(neighborhood, cmap='gray')
    axs[2, idx].quiver(x_coords, y_coords, np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)),
                       mag, scale=20, color='r', pivot='middle')
    axs[2, idx].set_title(f'Keypoint {idx + 1} Orientations')
    axs[2, idx].axis('off')

    axs[3, idx].imshow(mag, cmap='hot')
    axs[3, idx].set_title(f'Keypoint {idx + 1} Gradient Magnitudes')
    axs[3, idx].axis('off')

    bins = np.linspace(0, 360, 8)
    axs[4, idx].bar(bins, hist, width=40)
    axs[4, idx].set_title(f'Keypoint {idx + 1} Orientation Histogram')
    axs[4, idx].set_xlabel('Angle (degrees)')
    axs[4, idx].set_ylabel('Magnitude')

    axs[5, idx].plot(descriptors[keypoints.index(kp)])
    axs[5, idx].set_title(f'Keypoint {idx + 1} SIFT Descriptor')
    axs[5, idx].set_xlabel('Dimension')
    axs[5, idx].set_ylabel('Value')

plt.tight_layout()
plt.savefig("../outputs/keypoints_sift/keypoints.png")