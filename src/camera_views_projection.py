
"""This script implements a Hough Transform to detect lines in an image,
finds their intersections, and visualizes the results.
It uses OpenCV for image processing and Matplotlib for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def hough_transform(edge_image, theta_res=1, rho_res=1):
    height, width = edge_image.shape
    max_rho = int(np.sqrt(height**2 + width**2))

    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    rhos = np.arange(-max_rho, max_rho, rho_res)

    accumulator = np.zeros((len(rhos), len(thetas)))

    y_idxs, x_idxs = np.nonzero(edge_image)

    for y, x in zip(y_idxs, x_idxs):
        for theta_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx, theta_idx] += 1

    return accumulator, rhos, thetas

def find_peaks(accumulator, threshold=0.5, min_distance=10):
    acc_norm = accumulator / accumulator.max()
    peaks = []

    for i in range(1, accumulator.shape[0]-1):
        for j in range(1, accumulator.shape[1]-1):
            if acc_norm[i,j] > threshold:
                window = acc_norm[max(0,i-1):min(accumulator.shape[0],i+2),
                                max(0,j-1):min(accumulator.shape[1],j+2)]
                if acc_norm[i,j] == window.max():
                    peaks.append((i,j))

    peaks = sorted(peaks, key=lambda p: accumulator[p[0], p[1]], reverse=True)

    filtered_peaks = []
    for peak in peaks:
        if not filtered_peaks:
            filtered_peaks.append(peak)
        else:
            far_enough = True
            for existing_peak in filtered_peaks:
                dist = np.sqrt((peak[0]-existing_peak[0])**2 +
                             (peak[1]-existing_peak[1])**2)
                if dist < min_distance:
                    far_enough = False
                    break
            if far_enough:
                filtered_peaks.append(peak)

    return filtered_peaks

def convert_to_line_params(rho, theta):
    if np.abs(np.sin(theta)) < 1e-10:
        return None
    elif np.abs(np.cos(theta)) < 1e-10:
        return ('vertical', rho/np.cos(theta))
    else:
        m = -np.cos(theta) / np.sin(theta)
        b = rho / np.sin(theta)
        return (m, b)

def find_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2

    if np.abs(np.sin(theta1-theta2)) < 1e-10:
        return None

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])

    try:
        x, y = np.linalg.solve(A, b)
        if abs(x) > 10000 or abs(y) > 10000:
            return None
        return (x, y)
    except np.linalg.LinAlgError:
        return None

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    accumulator, rhos, thetas = hough_transform(edges)


    peaks = find_peaks(accumulator, threshold=0.4, min_distance=10)

    lines = [(rhos[rho_idx], thetas[theta_idx]) for rho_idx, theta_idx in peaks]

    return lines, image

def visualize_results(image, selected_lines, intersections):
    plt.figure(figsize=(12, 8))
    plt.imshow(image, cmap='gray')

    height, width = image.shape
    y_range = np.array([0, height])

    colors = ['r', 'g', 'b', 'y']
    for i, (rho, theta) in enumerate(selected_lines):
        color = colors[i % len(colors)]
        if np.abs(np.sin(theta)) < 1e-10:
            x = rho / np.cos(theta)
            plt.axvline(x=x, color=color, alpha=0.5, label=f'Line {i+1}')
        else:
            m, b = -np.cos(theta) / np.sin(theta), rho / np.sin(theta)
            x_range = np.array([0, width])
            y_values = m * x_range + b
            plt.plot(x_range, y_values, color=color, alpha=0.5, label=f'Line {i+1}')

    for x, y in intersections:
        inside = (0 <= x < width) and (0 <= y < height)
        color = 'go' if inside else 'ro'
        marker_label = 'Inside' if inside else 'Outside'
        plt.plot(x, y, color, markersize=10, label=marker_label)

    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.legend()
    plt.title('Detected Lines and Intersections')
    plt.savefig("../outputs/camera_views_projection/projection.png")

def main(image_path):
    lines, image = process_image(image_path)

    lines.sort(key=lambda x: (x[0], x[1]))

    print("\nDetected lines (sorted by rho, theta):")
    for rho, theta in lines:
        print(f"rho={rho:.2f}, theta={np.rad2deg(theta):.2f}°", end="")
        params = convert_to_line_params(rho, theta)
        if params is None:
            print(" -> Vertical line")
        elif params[0] == 'vertical':
            print(f" -> x = {params[1]:.2f}")
        else:
            m, b = params
            print(f" -> y = {m:.2f}x + {b:.2f}")

    if len(lines) >= 4:
        selected_indices = np.random.choice(len(lines), 4, replace=False)
        selected_lines = [lines[i] for i in selected_indices]

        intersections = []
        print("\nIntersection points:")
        for i in range(4):
            for j in range(i+1, 4):
                intersection = find_intersection(selected_lines[i], selected_lines[j])
                if intersection:
                    x, y = intersection
                    if abs(x) < 10000 and abs(y) < 10000:
                        intersections.append((x, y))
                        inside = (0 <= x < image.shape[1]) and (0 <= y < image.shape[0])
                        print(f"\nLines {i+1} and {j+1}:")
                        print(f"Coordinates: ({x:.2f}, {y:.2f})")
                        print(f"Status: {'Inside' if inside else 'Outside'} image borders")

        visualize_results(image, selected_lines, intersections)
    else:
        print("Not enough lines detected to find intersections")

if __name__ == "__main__":
    main("../images/chessboard.png")  # Change the path to your image file as needed



