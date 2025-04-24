#!/usr/bin/python
import numpy as np
import os
import Bio
import shutil
from Bio.PDB import * 
import sys
import importlib
from IPython.core.debugger import set_trace

# Local includes
from default_config.masif_opts import masif_opts
from triangulation.computeMSMS import computeMSMS
from triangulation.fixmesh import fix_mesh
import pymesh
from input_output.extractPDB import extractPDB
from input_output.editPDB import editPDB
from input_output.save_ply import save_ply
from input_output.read_ply import read_ply
from input_output.protonate import protonate
from triangulation.computeHydrophobicity import computeHydrophobicity
from triangulation.computeCharges import computeCharges, assignChargesToNewMesh
from triangulation.computeAPBS import computeAPBS
from triangulation.compute_normal import compute_normal
from sklearn.neighbors import KDTree
import pandas as pd
from pmp_ibs.assignIBS import *

"""
01f-conf_ensemble_pdb_extract_and_triangulate_center.py
: script for preprocessing HTMD-generated & CLoNe-clustered centers of PDB structure (for PPI case)
- updated PDB extract and input file format for the script
all the trianulation and precomputation steps are same with orginal script
except the iface determination
Updated by ByungUk Park (Apr, 23, 2025)

"""

if len(sys.argv) <= 1: 
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)


# Save the chains as separate files. 
in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1]

# if (len(sys.argv)>2) and (sys.argv[2]=='masif_ligand'):
#     pdb_filename = os.path.join(masif_opts["ligand"]["assembly_dir"],pdb_id+".pdb")
# else:


# Check representative center number
pdb_dir = masif_opts['ensemble_pdb_dir']+pdb_id
center_pdb_list = [f for f in os.listdir(pdb_dir) if f.startswith("Center_")]

print(f"{pdb_id} frame_num-based filtered cluster center number: {len(center_pdb_list):d}")

# loop for each cluster center
for center_pdb in center_pdb_list:
    raw_pdb = os.path.join(f'{pdb_dir}/', f'{center_pdb}')    
    pdb_filename = f"{pdb_dir}/{pdb_id}.pdb"
    shutil.copy(raw_pdb, pdb_filename)
       
     ### Preprocessing Steps ###
     # can be modulated in the future

    # Edit the pdb file chain ids to be the same as the chain_ids1 (HOH, ions removed) --> not helpful in case of full complex
    # editPDB(pdb_filename, chain_ids1)

    # protonate the pdb file
    tmp_dir= masif_opts['tmp_dir']
    protonated_file = tmp_dir+"/"+pdb_id+"_"+chain_ids1+".pdb"
    protonate(pdb_filename, protonated_file)
    pdb_filename = protonated_file
        
    # Extract chains of interest.
    out_filename1 = tmp_dir+"/"+pdb_id+"_"+chain_ids1
    extractPDB(pdb_filename, out_filename1+".pdb", chain_ids1)
    ### might need edits to match chain_ids1 and chain_id in the HTMD-generated PDB file
        
    # Compute MSMS of surface w/hydrogens, 
    try:
        vertices1, faces1, normals1, names1, areas1 = computeMSMS(out_filename1+".pdb",\
            protonate=True)
    except:
        set_trace()
        
    # Compute "charged" vertices: shape (vertice_num,)
    if masif_opts['use_hbond']:
        vertex_hbond = computeCharges(out_filename1, vertices1, names1)
        
    # For each surface residue, assign the hydrophobicity of its amino acid. (residue-level Kyte Doolitle scale, shape (vertice_num,))
    if masif_opts['use_hphob']:
        vertex_hphobicity = computeHydrophobicity(names1)
        
    # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
    vertices2 = vertices1
    faces2 = faces1
        
    # Fix the mesh.
    mesh = pymesh.form_mesh(vertices2, faces2)
    regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])
        
    # Compute the normals
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
    # Assign charges on new vertices based on charges of old vertices (nearest
    # neighbor)
        
    # assign surface features to vertices from new mesh: interpolating from 4 old NN vertices
    if masif_opts['use_hbond']:
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
            vertex_hbond, masif_opts)
        
    if masif_opts['use_hphob']:
        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
            vertex_hphobicity, masif_opts)
        
    if masif_opts['use_apbs']:
        vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1+".pdb", out_filename1)
        
    iface = np.zeros(len(regular_mesh.vertices))
    if 'compute_iface' in masif_opts and masif_opts['compute_iface']:
        # Compute the surface of the entire complex and from that compute the interface.
        v3, f3, _, _, _ = computeMSMS(pdb_filename,\
            protonate=True)
        # Regularize the mesh
        mesh = pymesh.form_mesh(v3, f3)
        # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
        full_regular_mesh = mesh
        # Find the vertices that are in the iface.
        v3 = full_regular_mesh.vertices
        # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
        kdt = KDTree(v3)
        d, r = kdt.query(regular_mesh.vertices)
        d = np.square(d) # Square d, because this is how it was in the pyflann version.
        assert(len(d) == len(regular_mesh.vertices))
        iface_v = np.where(d >= 2.0)[0]
        iface[iface_v] = 1.0
        # Convert to ply and save.
        save_ply(out_filename1+".ply", regular_mesh.vertices,\
                            regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                            normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,\
                            iface=iface)        
    else:
        # Convert to ply and save.
        save_ply(out_filename1+".ply", regular_mesh.vertices,\
                            regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                            normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
    ply_chain_dir = masif_opts['ply_chain_dir']+pdb_id+"_"+chain_ids1+f"/{(center_pdb).rstrip('.pdb')}/"
    pdb_chain_dir = masif_opts['ensemble_pdb_dir']+pdb_id+"_"+chain_ids1+f"/{(center_pdb).rstrip('.pdb')}/"
    if not os.path.exists(ply_chain_dir):
        os.makedirs(ply_chain_dir)
    if not os.path.exists(pdb_chain_dir):
        os.makedirs(pdb_chain_dir)
    shutil.copy(out_filename1+'.ply', ply_chain_dir) 
    shutil.copy(out_filename1+'.pdb', pdb_chain_dir) 

    # Clean up the template pdb file
    if os.path.isfile(f"{pdb_dir}/{pdb_id}_{chain_ids1}.pdb"):
        os.remove(f"{pdb_dir}/{pdb_id}_{chain_ids1}.pdb")
    # Clean up the tmp pdb file
    if os.path.isfile(pdb_filename):
        os.remove(pdb_filename)
