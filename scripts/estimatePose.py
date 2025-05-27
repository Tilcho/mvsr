import cv2
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# --- Configuration ---
image_folder = '/home/simon/Documents/MVSR Lab/mvsr/data/rgb/'
model_folder = '/home/simon/Documents/MVSR Lab/mvsr/data/models/'
image_files = [f"{i}.png" for i in range(1, 11)]
object_classes = {
    '1A': ('morobot-s_Achse-1A.obj', 800),
    '2B': ('morobot-s_Achse-2B.obj', 1000),
    '3B': ('morobot-s_Achse-3B_gray.obj', 1800),
    '4C': ('morobot-s_Achse-4C.obj', 1200)
}

# --- Step 0: Terminal Interface ---
def user_selection():
    print("Select an image to open:")
    for idx, img in enumerate(image_files):
        print(f"{idx + 1}: {img}")
    img_idx = int(input("Enter image number (1-10): ")) - 1
    img_path = os.path.join(image_folder, image_files[img_idx])

    print("\nSelect an object class to estimate:")
    for idx, cls in enumerate(object_classes.keys()):
        print(f"{idx + 1}: {cls}")
    cls_idx = int(input("Enter object class number (1-4): ")) - 1
    cls_name = list(object_classes.keys())[cls_idx]
    model_path, step = object_classes[cls_name]

    return img_path, os.path.join(model_folder, model_path), step

# --- Step 1: Load Mesh and Select Keypoints ---
def load_mesh_and_select_keypoints(model_path, step):
    mesh = trimesh.load(model_path)
    vertices = mesh.vertices
    faces = mesh.faces

    selected_indices = np.arange(0, len(vertices), step)
    points = vertices[selected_indices]

    return mesh, vertices, faces, selected_indices, points

# --- Step 2: User Defines 2D Keypoints ---
def get_image_points(image_path, num_points):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    clicked_points = []

    def on_click(event):
        if event.button == 1 and event.xdata is not None:
            clicked_points.append((event.xdata, event.ydata))
            print(f"Point: ({event.xdata:.1f}, {event.ydata:.1f})")
            if len(clicked_points) == num_points:
                plt.close()

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.title("Click on the 2D points (same order as 3D)")
    plt.show()

    return np.array(clicked_points, dtype=np.float32), img

# --- Step 3: Pose Estimation ---
def estimate_pose(object_points, image_points):
    camera_matrix = np.array([
        [616.7415, 0.0, 324.8176],
        [0.0, 616.9197, 238.0456],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(5)

    success, rvec, tvec = cv2.solvePnP(
        object_points.astype(np.float32),
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    return rvec, tvec, camera_matrix, dist_coeffs

# --- Step 4: Projection and Visualization ---
def project_and_visualize(vertices, faces, rvec, tvec, camera_matrix, dist_coeffs, img):
    projected_all, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
    projected_all = projected_all.reshape(-1, 2).astype(int)

    img_with_faces = img.copy()
    for face in faces:
        pts = projected_all[face].reshape(-1, 1, 2)
        cv2.polylines(img_with_faces, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

    plt.imshow(cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB))
    plt.title("Projected 3D Mesh (Faces)")
    plt.axis("off")
    plt.show()

def draw_coordinate_frame(vertices, rvec, tvec, camera_matrix, dist_coeffs, img):
    mesh_center = vertices.mean(axis=0)
    axis_length = 20
    axis_points_3d = np.array([
        mesh_center,
        mesh_center + [axis_length, 0, 0],
        mesh_center + [0, axis_length, 0],
        mesh_center + [0, 0, axis_length],
    ], dtype=np.float32)

    projected_axes, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    origin_2d = tuple(projected_axes[0].ravel().astype(int))
    x_2d = tuple(projected_axes[1].ravel().astype(int))
    y_2d = tuple(projected_axes[2].ravel().astype(int))
    z_2d = tuple(projected_axes[3].ravel().astype(int))

    img_axes = img.copy()
    cv2.arrowedLine(img_axes, origin_2d, x_2d, (0, 0, 255), 2, tipLength=0.1)
    cv2.arrowedLine(img_axes, origin_2d, y_2d, (0, 255, 0), 2, tipLength=0.1)
    cv2.arrowedLine(img_axes, origin_2d, z_2d, (255, 0, 0), 2, tipLength=0.1)

    plt.imshow(cv2.cvtColor(img_axes, cv2.COLOR_BGR2RGB))
    plt.title("Projected Coordinate Frame at Mesh Center")
    plt.axis("off")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    img_path, model_path, step = user_selection()
    mesh, vertices, faces, selected_indices, object_points = load_mesh_and_select_keypoints(model_path, step)
    image_points, img = get_image_points(img_path, len(object_points))
    rvec, tvec, camera_matrix, dist_coeffs = estimate_pose(object_points, image_points)

    print("\n=== Mesh Center in Camera Coordinates ===")
    mesh_center_cam = cv2.Rodrigues(rvec)[0] @ vertices.mean(axis=0).reshape(3, 1) + tvec
    print(f"Translation (X, Y, Z): {mesh_center_cam.ravel()}")
    print("\nRotation Matrix:")
    print(cv2.Rodrigues(rvec)[0])

    project_and_visualize(vertices, faces, rvec, tvec, camera_matrix, dist_coeffs, img)
    draw_coordinate_frame(vertices, rvec, tvec, camera_matrix, dist_coeffs, img)
