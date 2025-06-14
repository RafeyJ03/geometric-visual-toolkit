
"""3D Epipolar Geometry Visualization with Three Cameras
This script generates a 3D visualization of epipolar geometry using three cameras arranged in a triangular configuration.
It includes camera poses, 3D points, image planes, epipoles, and baselines.
It uses Matplotlib for 3D plotting and NumPy for numerical operations.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Color palette for visualizing different 3D points consistently across views
COLOR_PALETTE = [
    "red", "green", "blue", "cyan", "magenta", "yellow", "black", "brown"
]

def _generate_camera_frames(baseline_length=10.0, camera_height=0.0):
    """Creates three camera poses arranged in an equilateral triangle configuration.
    Each camera looks at a central point, forming a triangular baseline.
    """
    center_point = np.array([10.0, 0.0, 8.0])

    angle = 2 * np.pi / 3
    radius = baseline_length/2 / np.sin(angle/2)

    camera1_pos = center_point + np.array([radius * np.cos(0), radius * np.sin(0), -4.0])
    camera2_pos = center_point + np.array([radius * np.cos(angle), radius * np.sin(angle), -4.0])
    camera3_pos = center_point + np.array([radius * np.cos(2*angle), radius * np.sin(2*angle), -4.0])

    def create_camera_matrix(camera_pos, look_at_point):
        """Creates a camera transformation matrix given position and look-at point"""
        forward = look_at_point - camera_pos
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        R = np.column_stack((right, up, forward))

        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = R
        camera_matrix[:3, 3] = camera_pos

        return camera_matrix

    camera1 = create_camera_matrix(camera1_pos, center_point)
    camera2 = create_camera_matrix(camera2_pos, center_point)
    camera3 = create_camera_matrix(camera3_pos, center_point)

    T12 = np.dot(np.linalg.inv(camera1), camera2)
    T23 = np.dot(np.linalg.inv(camera2), camera3)
    T31 = np.dot(np.linalg.inv(camera3), camera1)

    return camera1, camera2, camera3, T12, T23, T31

def _get_rotation_matrix(r, p, y):
    """Converts roll, pitch, yaw angles to rotation matrix using scipy's Rotation"""
    return R.from_euler("zyx", [r, p, y]).as_matrix()

def _generate_3d_points_in_world(num_points):
    """Generates random 3D points within a cubic volume for visualization.
    Points are centered around a specified point with random distribution.
    """
    center = np.array([10.0, 0.0, 8.0])
    size = 3.0

    xs = np.random.uniform(low=center[0]-size/2, high=center[0]+size/2, size=num_points)
    ys = np.random.uniform(low=center[1]-size/2, high=center[1]+size/2, size=num_points)
    zs = np.random.uniform(low=center[2]-size/2, high=center[2]+size/2, size=num_points)
    h = np.ones((num_points,))
    return np.array((xs, ys, zs, h))

def _generate_bounding_cube(points):
    """Creates a bounding cube around the given set of points with padding.
    """
    min_coords = np.min(points[:3, :], axis=1)
    max_coords = np.max(points[:3, :], axis=1)

    ranges = max_coords - min_coords
    padding = ranges * 0.1
    min_coords -= padding
    max_coords += padding

    vertices = np.array([
        [min_coords[0], min_coords[1], min_coords[2]],  # 0
        [max_coords[0], min_coords[1], min_coords[2]],  # 1
        [max_coords[0], max_coords[1], min_coords[2]],  # 2
        [min_coords[0], max_coords[1], min_coords[2]],  # 3
        [min_coords[0], min_coords[1], max_coords[2]],  # 4
        [max_coords[0], min_coords[1], max_coords[2]],  # 5
        [max_coords[0], max_coords[1], max_coords[2]],  # 6
        [min_coords[0], max_coords[1], max_coords[2]]   # 7
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    return vertices, edges

def _plot_cube(ax, vertices, edges):
    """Plots a cube on the given 3D axes using vertices and edges.
    """
    for start_idx, end_idx in edges:
        start = vertices[start_idx]
        end = vertices[end_idx]
        ax.plot([start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                'purple', linewidth=1, linestyle='--', alpha=0.5)

def _plot_camera_frame(ax, camera, scale=1):
    """Plots camera coordinate frame axes on 3D plot.
    Red, green, and blue lines represent x, y, and z axes respectively.
    """
    center = camera[:3, 3]
    x = scale * camera[:3, 0]
    y = scale * camera[:3, 1]
    z = scale * camera[:3, 2]

    ax.plot(
        (center[0], center[0] + x[0]),
        (center[1], center[1] + x[1]),
        (center[2], center[2] + x[2]),
        "r",
    )
    ax.plot(
        (center[0], center[0] + y[0]),
        (center[1], center[1] + y[1]),
        (center[2], center[2] + y[2]),
        "g",
    )
    ax.plot(
        (center[0], center[0] + z[0]),
        (center[1], center[1] + z[1]),
        (center[2], center[2] + z[2]),
        "b",
    )

    return ax

def _transform_points_in_cameras(points, camera1, camera2):
    """Projects 3D points into two camera views with perspective division.
    """
    points_in_cam_1 = np.dot(np.linalg.inv(camera1), points)
    points_in_cam_2 = np.dot(np.linalg.inv(camera2), points)

    for i in range(points.shape[1]):
        points_in_cam_1[:2, i] /= points_in_cam_1[2, i]
        points_in_cam_1[2, i] = 1
        points_in_cam_2[:2, i] /= points_in_cam_2[2, i]
        points_in_cam_2[2, i] = 1

    return points_in_cam_1[:3, :], points_in_cam_2[:3, :]

def _plot_camera_images(points_1, points_2, points_3, vertices, edges, camera1, camera2, camera3):
    """Creates a figure showing the projected points and bounding cube in all three camera views.

    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    vertices_h = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    vertices_cam1 = np.dot(np.linalg.inv(camera1), vertices_h.T)
    vertices_cam2 = np.dot(np.linalg.inv(camera2), vertices_h.T)
    vertices_cam3 = np.dot(np.linalg.inv(camera3), vertices_h.T)

    vertices_cam1[:2, :] /= vertices_cam1[2, :]
    vertices_cam2[:2, :] /= vertices_cam2[2, :]
    vertices_cam3[:2, :] /= vertices_cam3[2, :]

    for i in range(points_1.shape[1]):
        axs[0].scatter(-points_1[0, i], points_1[1, i], color=COLOR_PALETTE[i])
        axs[1].scatter(-points_2[0, i], points_2[1, i], color=COLOR_PALETTE[i])
        axs[2].scatter(-points_3[0, i], points_3[1, i], color=COLOR_PALETTE[i])

    for start_idx, end_idx in edges:
        axs[0].plot([-vertices_cam1[0, start_idx], -vertices_cam1[0, end_idx]],
                   [vertices_cam1[1, start_idx], vertices_cam1[1, end_idx]],
                   'purple', linestyle='--', alpha=0.5)
        axs[1].plot([-vertices_cam2[0, start_idx], -vertices_cam2[0, end_idx]],
                   [vertices_cam2[1, start_idx], vertices_cam2[1, end_idx]],
                   'purple', linestyle='--', alpha=0.5)
        axs[2].plot([-vertices_cam3[0, start_idx], -vertices_cam3[0, end_idx]],
                   [vertices_cam3[1, start_idx], vertices_cam3[1, end_idx]],
                   'purple', linestyle='--', alpha=0.5)

    for ax in axs:
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)

    axs[0].set_title("Image from camera 1")
    axs[1].set_title("Image from camera 2")
    axs[2].set_title("Image from camera 3")

    plt.tight_layout()
    plt.savefig("../outputs/3d_multi_camera/3d_multi_camera.png")

def _compute_kronecker(points_1, points_2):
    """Computes the Kronecker product for essential matrix estimation.

    """
    x1 = points_1[0, :]
    y1 = points_1[1, :]
    x2 = points_2[0, :]
    y2 = points_2[1, :]

    chi = np.ones((8, 9))
    chi[:, 0] = np.multiply(x2, x1)
    chi[:, 1] = np.multiply(x2, y1)
    chi[:, 2] = x2
    chi[:, 3] = np.multiply(y2, x1)
    chi[:, 4] = np.multiply(y2, y1)
    chi[:, 5] = y2
    chi[:, 6] = x1
    chi[:, 7] = y1

    return chi

def _estimate_decomponsed_essential_matrix(chi):
    """Estimates the essential matrix using SVD decomposition.

    """
    _, _, V1 = np.linalg.svd(chi)
    F = V1[8, :].reshape(3, 3).T
    U, _, V = np.linalg.svd(F)
    if np.linalg.det(np.dot(U, V)) < 1:
        V = -V
    return U, np.diag((1, 1, 0)), V

def _extract_rot_transl(U, V):
    """Extracts possible rotation and translation from essential matrix decomposition.

    """
    W = np.array(([0, -1, 0], [1, 0, 0], [0, 0, 1]))
    return [
        [np.dot(U, np.dot(W, V)), U[-1, :]],
        [np.dot(U, np.dot(W, V)), -U[-1, :]],
        [np.dot(U, np.dot(W.T, V)), U[-1, :]],
        [np.dot(U, np.dot(W.T, V)), -U[-1, :]],
    ]

def _generate_image_plane(camera_pose, width=3, height=3):
    """Generates corners of an image plane for a given camera pose.

    """
    center = camera_pose[:3, 3]
    forward = camera_pose[:3, 2]
    up = camera_pose[:3, 1]
    right = camera_pose[:3, 0]

    distance = 2.0
    plane_center = center + distance * forward

    corners = []
    for i in [-width/2, width/2]:
        for j in [-height/2, height/2]:
            point = plane_center + i*right + j*up
            corners.append(point)

    return np.array(corners)

def _plot_image_plane(ax, corners, color='blue', alpha=0.8):
    """Visualizes an image plane in 3D space using its corner points.

    """
    edges = [(0,1), (1,3), (3,2), (2,0)]
    for i, j in edges:
        ax.plot([corners[i,0], corners[j,0]],
                [corners[i,1], corners[j,1]],
                [corners[i,2], corners[j,2]],
                color=color, alpha=alpha)

    x = corners[:,0].reshape(2,2)
    y = corners[:,1].reshape(2,2)
    z = corners[:,2].reshape(2,2)

    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def _calculate_epipoles(camera1, camera2):
    """Calculates epipoles for a pair of cameras by finding the intersection
    of the baseline with each image plane.
    """
    c1 = camera1[:3, 3]
    c2 = camera2[:3, 3]

    distance = 2.0

    forward1 = camera1[:3, 2]
    forward2 = camera2[:3, 2]

    plane1_center = c1 + distance * forward1
    plane2_center = c2 + distance * forward2

    n1 = forward1
    n2 = forward2

    t1 = np.dot(n1, (plane1_center - c1)) / np.dot(n1, (c2 - c1))
    epipole1 = c1 + t1 * (c2 - c1)

    t2 = np.dot(n2, (plane2_center - c1)) / np.dot(n2, (c2 - c1))
    epipole2 = c1 + t2 * (c2 - c1)

    return epipole1, epipole2

def _plot_3d_world(camera_1_pose, camera_2_pose, camera_3_pose, points):
    """Creates a comprehensive 3D visualization of the epipolar geometry setup including
    cameras, image planes, epipoles, points, and baselines.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Plot 3D points
    for i in range(points.shape[1]):
        ax.scatter(points[0, i], points[1, i], points[2, i], color=COLOR_PALETTE[i])

    # Plot camera coordinate frames
    ax = _plot_camera_frame(ax, camera_1_pose, scale=2)
    ax = _plot_camera_frame(ax, camera_2_pose, scale=2)
    ax = _plot_camera_frame(ax, camera_3_pose, scale=2)

    # Generate and plot image planes
    corners1 = _generate_image_plane(camera_1_pose, width=6, height=5)
    corners2 = _generate_image_plane(camera_2_pose, width=6, height=5)
    corners3 = _generate_image_plane(camera_3_pose, width=6, height=5)
    _plot_image_plane(ax, corners1, color='blue', alpha=0.1)
    _plot_image_plane(ax, corners2, color='red', alpha=0.1)
    _plot_image_plane(ax, corners3, color='green', alpha=0.1)

    # Calculate and plot epipoles
    epipole12_1, epipole12_2 = _calculate_epipoles(camera_1_pose, camera_2_pose)
    epipole23_2, epipole23_3 = _calculate_epipoles(camera_2_pose, camera_3_pose)
    epipole31_3, epipole31_1 = _calculate_epipoles(camera_3_pose, camera_1_pose)

    # Add epipole markers and labels
    for epipole, label in [
        (epipole12_1, 'e12'), (epipole12_2, 'e21'),
        (epipole23_2, 'e23'), (epipole23_3, 'e32'),
        (epipole31_3, 'e31'), (epipole31_1, 'e13')
    ]:
        ax.scatter(epipole[0], epipole[1], epipole[2],
                  color='yellow', s=100, marker='*')
        ax.text(epipole[0], epipole[1], epipole[2],
                label, fontsize=10)

    # Plot camera baselines
    camera1_center = camera_1_pose[:3, 3]
    camera2_center = camera_2_pose[:3, 3]
    camera3_center = camera_3_pose[:3, 3]

    ax.plot([camera1_center[0], camera2_center[0]],
            [camera1_center[1], camera2_center[1]],
            [camera1_center[2], camera2_center[2]],
            'k--', label='Baselines')
    ax.plot([camera2_center[0], camera3_center[0]],
            [camera2_center[1], camera3_center[1]],
            [camera2_center[2], camera3_center[2]],
            'k--')
    ax.plot([camera3_center[0], camera1_center[0]],
            [camera3_center[1], camera1_center[1]],
            [camera3_center[2], camera1_center[2]],
            'k--')

    # Add bounding cube
    vertices, edges = _generate_bounding_cube(points)
    _plot_cube(ax, vertices, edges)

    # Set up plot labels and limits
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    center_point = np.array([10.0, 0.0, 8.0])
    limit_size = 12.0
    ax.set_xlim([center_point[0] - limit_size, center_point[0] + limit_size])
    ax.set_ylim([center_point[1] - limit_size, center_point[1] + limit_size])
    ax.set_zlim([0, center_point[2] + limit_size])

    ax.view_init(elev=25, azim=100)

    plt.title("3D Epipolar Geometry Visualization with Three Cameras")
    plt.legend()

    plt.savefig("../outputs/3d_multi_camera/3d_epipolar.png")

def main():
    # Set up camera system and generate 3D points
    camera_1, camera_2, camera_3, T12, T23, T31 = _generate_camera_frames(baseline_length=10.0, camera_height=0.0)
    points = _generate_3d_points_in_world(num_points=8)

    # Create visualizations
    _plot_3d_world(camera_1, camera_2, camera_3, points)

    points_1, points_2 = _transform_points_in_cameras(points, camera1=camera_1, camera2=camera_2)
    _, points_3 = _transform_points_in_cameras(points, camera1=camera_2, camera2=camera_3)

    vertices, edges = _generate_bounding_cube(points)
    _plot_camera_images(points_1, points_2, points_3, vertices, edges,
                       camera_1, camera_2, camera_3)

    chi = _compute_kronecker(points_1, points_2)
    U, _, V = _estimate_decomponsed_essential_matrix(chi)
    possible_r_t = _extract_rot_transl(U, V)

if __name__ == "__main__":
    main()