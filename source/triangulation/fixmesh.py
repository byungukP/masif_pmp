import numpy as np
from numpy.linalg import norm
import pymesh

"""
fixmesh.py: Regularize a protein surface mesh. 
- based on code from the PyMESH documentation. 
- updated for better remove degenerate vertices and triangles
"""


def fix_mesh(mesh, resolution, detail="normal"):
    bbox_min, bbox_max = mesh.bbox;
    diag_len = norm(bbox_max - bbox_min);
    if detail == "normal":
        target_len = diag_len * 5e-3;
    elif detail == "high":
        target_len = diag_len * 2.5e-3;
    elif detail == "low":
        target_len = diag_len * 1e-2;
    
    target_len = resolution
    #print("Target resolution: {} mm".format(target_len));
    # PGC 2017: Remove duplicated vertices first, two vertices can be considered as duplicates of each other if their Euclidean distance is less than a tolerance (0.001 angstrom)
    
    # Step 1: Remove duplicated vertices (small tolerance)
    # Duplicate vertices can often be merged into a single vertex.
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)

    # Step 2: Remove degenerate triangles
    # Degenerate triangles are triangles with collinear vertices (zero area and their normals are undefined)
    print("Removing degenerated triangles")
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100);
    
    # Step 3: Split long edges (to control resolution)
    mesh, __ = pymesh.split_long_edges(mesh, target_len);

    # Step 4: Collapse short edges and remove obtuse triangles iteratively
    count = 0;
    num_vertices = mesh.num_vertices;
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6);
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                preserve_feature=True);
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100);
        if mesh.num_vertices == num_vertices:
            break;

        num_vertices = mesh.num_vertices;
        #print("#v: {}".format(num_vertices));
        count += 1;
        if count > 10: break;

    # Step 5: Resolve self-intersections & outer hull
    mesh = pymesh.resolve_self_intersection(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh = pymesh.compute_outer_hull(mesh);

    # Step 6: Additional cleaning AFTER outer hull
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5);
    mesh, __ = pymesh.remove_isolated_vertices(mesh);
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)

    # Final check for degenerate faces (zero-area triangles)
    r = pymesh.get_degenerated_faces(mesh)
    if len(r) > 0:
        print(f"WARNING: {len(r)} degenerate (zero-area) faces detected.")
        print(r)

        print("mesh.faces: ",mesh.faces)
        print((mesh.faces).shape)
        print("mesh.vertice: ",mesh.vertices)
        print((mesh.vertices).shape)

        # Remove degenerate triangles manually
        valid_faces = np.delete(mesh.faces, r, axis=0)
        mesh = pymesh.form_mesh(mesh.vertices, valid_faces)
        mesh, __ = pymesh.remove_isolated_vertices(mesh);
        mesh, _ = pymesh.remove_duplicated_vertices(mesh, 0.001)


        print("After manual removing")
        r = pymesh.get_degenerated_faces(mesh)
        print(r)
        print("mesh.faces: ",mesh.faces)
        print((mesh.faces).shape)
        print("mesh.vertice: ",mesh.vertices)
        print((mesh.vertices).shape)


    return mesh
