# Geometric Vision Tools

> A visual toolkit for understanding camera geometry, projection, and image-based line detection in computer vision.

## 🧭 Overview

**Geometric Vision Tools** is a curated collection of Python utilities and visualizations that demonstrate how computers interpret images in 3D space. The toolkit focuses on **camera projection**, **epipolar geometry**, and **image structure detection** — foundational techniques behind applications like 3D reconstruction, autonomous navigation, and augmented reality.

---

## 🔍 Features

- **Epipolar Geometry Visualizer**  
  - Simulates a 3-camera setup observing the same 3D scene  
  - Visualizes camera poses, image planes, epipoles, and 3D-to-2D projections  

- **Essential Matrix Estimation**  
  - Computes the essential matrix from corresponding points  
  - Uses custom SVD-based decomposition  
  - Returns possible camera rotation and translation pairs

- **3D Bounding Cube + Projection Tools**  
  - Generates random 3D points and projects them into camera views  
  - Adds bounding cubes for spatial context  
  - Demonstrates image-to-world and world-to-image transformations

- **Hough Line Detection**  
  - Custom Hough Transform implementation  
  - Detects lines in real images and estimates their intersection points  
  - Includes visualization and classification of in-image vs out-of-bounds intersections

---

## 🛠️ Technologies Used

- `Python 3.10+`  
- `NumPy`, `SciPy` – for vector math and linear algebra  
- `Matplotlib` – for 2D and 3D visualizations  
- `OpenCV` – for image loading, filtering, and edge detection

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/geometric-vision-tools.git
cd geometric-vision-tools

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run one of the tools
python src/epipolar_vis.py
python src/hough_demo.py
