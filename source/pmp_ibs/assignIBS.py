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


# names from msms vertices, pmp_df from dataset.csv

def computeIBS(names, pmp_df, pdb_id, chain_ids1):
    ibs_res_ix = generate_IBSresix(pmp_df, pdb_id, chain_ids1)
    iface = np.zeros(len(names))
    pdb_df = pmp_df[pmp_df["pdb"] == pdb_id]
    for vix, name in enumerate(names):
        fields = name.split('_')
        chain_id, res_id, resname, atomname = fields[0], int(fields[1]), fields[3], fields[4]
        if res_id in ibs_res_ix:
            if crosscheck(pdb_df, pdb_id, res_id, chain_ids1, resname):    # chain_ids1, resname
                iface[vix] = 1.0
    return iface


def crosscheck(pdb_df, pdb_id, res_id, chain_ids1, resname):
    data_chain = pdb_df[pdb_df["residue_number"] == res_id]["chain_id"].values[0]
    data_resname = pdb_df[pdb_df["residue_number"] == res_id]["residue_name"].values[0]
    assert data_chain == chain_ids1, \
        f"Mismatch Error\nchain id mismatch between residues (res_id: {res_id}) from mesh vertices and pmp_dataset.csv\n \
            mesh vertices: {chain_ids1}\n \
            pmp_dataset.csv: {data_chain}"
    assert data_resname == resname, \
        f"Mismatch Error\nresidue name mismatch between residues (res_id: {res_id}) from mesh vertices and pmp_dataset.csv\n \
            mesh vertices: {resname}\n \
            pmp_dataset.csv: {data_resname}"
    return True


def crosscheck_residue(pdb_filename, pmp_df, ibs_res_ix):
    logfile = open("logfile.txt", 'w')
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_filename, pdb_filename)
    for res in struct.get_residues():
        res_id = res.get_id()[1]
        # check in terms of chain_id (already done in previous codes),
        # res id, resname before taking the res_id as index for IBS label
        logfile.write("{}   {}\n".format(res_id, res.get_resname))
    logfile.close()


# names: atom_id in format (e.g.B_125_x_ASN_ND2_Green),shape=(vert_num,)

def crosscheck_residue2(names1, pmp_df, ibs_res_ix):
    logfile = open("logfile_names.txt", 'w')
    logfile.write("#chain_id    res_id    resname    atomname\n")
    logfile.write("total surface vertices number before mesh regularization: {:d}\n".format(len(names1)))
    for name in names1:
        fields = name.split('_')
        chain_id, res_id, resname, atomname = fields[0], int(fields[1]), fields[3], fields[4]
        # check in terms of chain_id (already done in previous codes),
        # res id, resname before taking the res_id as index for IBS label
        logfile.write("{}    {}    {}    {}\n".format(chain_id,res_id,resname,atomname))
    logfile.close()


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
