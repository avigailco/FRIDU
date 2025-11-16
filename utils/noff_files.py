from pyFM.mesh import TriMesh
import os, sys
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "."))
# from utils.utils import normalize_to_unit_sphere
# from utils import normalize_to_unit_sphere


def load_noff(file_path):
    vertices = []
    normals = []
    faces = []

    with open(file_path, 'r') as f:
        header = f.readline().strip()

        if header != "NOFF":
            raise TypeError("Not a valid NOFF header")

        # Read the number of vertices, faces, and edges
        counts = f.readline().strip().split()
        num_vertices, num_faces, _ = map(int, counts)

        # Read the vertices and their normals
        for _ in range(num_vertices):
            line = f.readline().strip().split()
            vertex = list(map(float, line[:3]))  # First three values are the position
            normal = list(map(float, line[3:]))  # Next three values are the normal
            vertices.append(vertex)
            normals.append(normal)

        # Read the faces
        for _ in range(num_faces):
            line = f.readline().strip().split()
            face = list(map(int, line[1:]))  # Skip the first value (number of vertices in the face)
            faces.append(face)

    return {
        "vertices": vertices,
        "normals": normals,
        "faces": faces
    }


def noff_to_TriMesh(file_path, normalize_mesh=False):
    noff_mesh = load_noff(file_path)
    if normalize_mesh:
        raise ValueError(f'mesh normalization not implemented')
        # vertices = normalize_to_unit_sphere(np.array(noff_mesh['vertices']))
    else:
        vertices = noff_mesh['vertices']

    mesh = TriMesh(vertices, noff_mesh['faces'])
    mesh.vertex_normals = np.asarray(noff_mesh['normals'])
    return mesh

