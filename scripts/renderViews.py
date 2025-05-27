import open3d as o3d
import numpy as np
import cv2
import os

# === Einstellungen ===
OBJ_PATH = "/home/simon/Documents/MVSR Lab/mvsr/data/models/morobot-s_Achse-1A_gray.obj"
OUTPUT_DIR = "renders1"
IMG_WIDTH, IMG_HEIGHT = 640, 480
VIEWS = 72

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Mesh laden ===
mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
mesh.compute_vertex_normals()

# === Renderer vorbereiten ===
render = o3d.visualization.rendering.OffscreenRenderer(IMG_WIDTH, IMG_HEIGHT)
render.scene.set_background([0, 0, 0, 1])
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"
render.scene.add_geometry("model", mesh, mat)

# === Bounding Box und Kamerapositionen ===
bbox = mesh.get_axis_aligned_bounding_box()
center = bbox.get_center()
radius = bbox.get_max_extent()

# Kamera-Objekt
cam = render.scene.camera
fov = 60.0  # Field of view
eye_distance = radius * 2.5

# === Bilder rendern aus verschiedenen Blickwinkeln ===
print("Rendere Modell aus verschiedenen Blickwinkeln...")
for i in range(VIEWS):
    angle = i * (2 * np.pi / VIEWS)
    eye = center + eye_distance * np.array([np.cos(angle), 0, np.sin(angle)])
    up = np.array([0, 1, 0])

    cam.look_at(center, eye, up)
    img = render.render_to_image()
    img_np = np.asarray(img)
    filename = os.path.join(OUTPUT_DIR, f"view_{i}.png")
    cv2.imwrite(filename, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    print(f"  -> gespeichert: {filename}")

print("Rendering abgeschlossen.")
