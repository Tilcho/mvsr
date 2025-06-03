import trimesh

# Load your mesh
mesh = trimesh.load("meshes/3Bg.obj")

# Scale from mm to m
mesh.apply_scale(0.001)

# Export the new mesh
mesh.export('meshes/3Bg_m.obj')
