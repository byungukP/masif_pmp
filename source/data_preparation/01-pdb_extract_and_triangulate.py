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

# Modified from files in MaSIF (Pablo Gainza - LPDI STI EPFL 2019)
# Released under an Apache License 2.0

if len(sys.argv) <= 1: 
    print("Usage: {config} "+sys.argv[0]+" PDBID_A")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)


# Save the chains as separate files.
in_fields = sys.argv[1].split("_")
pdb_id = in_fields[0]
chain_ids1 = in_fields[1]

pdb_filename = masif_opts['raw_pdb_dir']+pdb_id+".pdb"
tmp_dir= masif_opts['tmp_dir']
protonated_file = tmp_dir+"/"+pdb_id+".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

# Extract chains of interest.
out_filename1 = tmp_dir+"/"+pdb_id+"_"+chain_ids1
extractPDB(pdb_filename, out_filename1+".pdb", chain_ids1)

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

# Assign surface features to vertices from new mesh: interpolating from 4 old NN vertices
if masif_opts['use_hbond']:
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
        vertex_hbond, masif_opts)

if masif_opts['use_hphob']:
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1,\
        vertex_hphobicity, masif_opts)

if masif_opts['use_apbs']:
    vertex_charges = computeAPBS(regular_mesh.vertices, out_filename1+".pdb", out_filename1)

"""
PMP IBS/nonIBS labeling
- read csv file and get the info on chain_id, resid_id, resid_name
- match the vertices from regularized_mesh with the IBS from csv
- save into ply file using save_ply(): ifaces saved as attribute of the mesh object
- use PDBParser for matching process

"""

csv_path = masif_opts["pmp_dataset"]

if not os.path.exists(csv_path):
    print(".csv file for PMP dataset does not exist. Please check /masif_pmp/source/default_config/masif_opts.py.")
    sys.exit(1)

# For each surface residue, assign the IBS labels of its amino acid. (residue-level IBS, shape (vertice_num,))
if 'compute_ibs' in masif_opts and masif_opts['compute_ibs']:
    # load csv file and save info as array
    pmp_df = pd.read_csv(csv_path)

    # compute iface based on IBS type in dataset: "boolean" or "score"
    if (not 'annotation_type' in masif_opts) and (not masif_opts['annotation_type'] in ['boolean', 'score']):
        print("IBS annotation type of dataset not defined")
        sys.exit(1)
    iface_ = computeIBS(names1, pmp_df, pdb_id, chain_ids1, type = masif_opts['annotation_type'])
    # assign IBS_label on vertices of regularized mesh (nearest neighbor)
    iface = assignIBSToNewMesh(regular_mesh.vertices, vertices1,\
                                   iface_)
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
if not os.path.exists(masif_opts['ply_chain_dir']):
    os.makedirs(masif_opts['ply_chain_dir'])
if not os.path.exists(masif_opts['pdb_chain_dir']):
    os.makedirs(masif_opts['pdb_chain_dir'])
shutil.copy(out_filename1+'.ply', masif_opts['ply_chain_dir']) 
shutil.copy(out_filename1+'.pdb', masif_opts['pdb_chain_dir']) 
