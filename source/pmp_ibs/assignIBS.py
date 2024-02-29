from Bio.PDB import *
import numpy as np
from sklearn.neighbors import KDTree

"""
assignIBS.py: Wrapper function to compute hydrogen bond potential (free electrons/protons) in the surface
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""


def generate_IBSresix(pmp_df, pdb_id, chain_ids1):
    # parsing IBS residues of target pdb based on pdb_id, chain_id, IBS label
    cond1 = pmp_df["pdb"] == pdb_id
    cond2 = pmp_df["chain_id"] == chain_ids1
    cond3 = pmp_df["IBS"] == True
    ibs_df = pmp_df[cond1 & cond2 & cond3]
    ibs_res_ix = np.array(ibs_df["residue_number"])
    return ibs_res_ix


def crosscheck_residue(pdb_filename, pmp_df, ibs_res_ix):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_filename, pdb_filename + ".pdb")
    for res in struct.get_residues():
        res_id = res.get_id()[1]
        # check in terms of chain_id (already done in previous codes),
        # res id, resname before taking the res_id as index for IBS label
        if res_id in ibs_res_ix:
            assert res.get_resname() == pmp_df[pmp_df["residue_number"] == res_id]["residue_name"].values, \
                                        f"Residue with res_id {res_id} from PDBParser doesn't match with residue_number used in pmp_dataset.csv"
    return True


# names: atom_id in format (e.g.B_125_x_ASN_ND2_Green),shape=(vert_num,)

def computeIBS(names, ibs_res_ix):
    iface = np.zeros(len(names))
    for vix, name in enumerate(names):
        res_id = name.split("_")[1]
        if res_id in ibs_res_ix:
            iface[vix] = 1.0
    return iface


# Assign IBS_label on new vertices based on IBS_label of old vertices (nearest
# neighbor)

def assignIBSToNewMesh(new_vertices, old_vertices, old_charges):
    dataset = old_vertices
    testset = new_vertices
    new_charges = np.zeros(len(new_vertices))        
    # Assign k old vertices to each new vertex.
    kdt = KDTree(dataset)
    dists, result = kdt.query(testset)
    new_charges = old_charges[result]
    return new_charges
