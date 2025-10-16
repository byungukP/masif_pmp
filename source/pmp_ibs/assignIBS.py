from Bio.PDB import *
import numpy as np
from sklearn.neighbors import KDTree

"""
assignIBS.py: Functions to compute and assign IBS labels of the surface points
ByungUk Park UW-Madison, 2024
Released under an MIT License
"""


def generate_IBSresix(pmp_df, pdb_id, chain_ids1):
    # parsing IBS residues of target pdb based on pdb_id, chain_id, IBS label
    cond = pmp_df["IBS"] == True
    ibs_df = pmp_df[cond]
    ibs_res_ix = np.array(ibs_df["residue_number"])
    return ibs_res_ix


# names from msms vertices, pmp_df from dataset.csv

def computeIBS(names, pmp_df, pdb_id, chain_ids1, type="boolean"):
    iface = np.zeros(len(names))
    pmp_df["cathpdb"] = pmp_df["cathpdb"].str.upper()
    # parse data of target pdb_chain
    pdb_id_match = pmp_df["cathpdb"].str[:4] == pdb_id
    chain_id_match = pmp_df["chain_id"] == chain_ids1
    pdb_df = pmp_df[pdb_id_match & chain_id_match]

    if type == "boolean":
        ibs_res_ix = generate_IBSresix(pdb_df, pdb_id, chain_ids1)
        for vix, name in enumerate(names):
            fields = name.split('_')
            chain_id, res_id, resname, atomname = fields[0], int(fields[1]), fields[3], fields[4]
            if res_id in ibs_res_ix:
                if crosscheck(pdb_df, pdb_id, res_id, chain_ids1, resname):    # pdb_id, chain_ids1, resname
                    iface[vix] = 1.0
        return iface

    elif type == "score":
        for vix, name in enumerate(names):
            fields = name.split('_')
            chain_id, res_id, resname, atomname = fields[0], int(fields[1]), fields[3], fields[4]
            cond = pdb_df["residue_number"] == res_id
            if np.sum(cond) == 1:
                if crosscheck(pdb_df, pdb_id, res_id, chain_ids1, resname):    # pdb_id, chain_ids1, resname
                    iface[vix] = pdb_df[cond]["IBS"].values[0]
        return iface

def crosscheck(pdb_df, pdb_id, res_id, chain_ids1, resname):
    data_pdb = pdb_df[pdb_df["residue_number"] == res_id]["cathpdb"].values[0][:4]
    data_chain = pdb_df[pdb_df["residue_number"] == res_id]["chain_id"].values[0]
    data_resname = pdb_df[pdb_df["residue_number"] == res_id]["residue_name"].values[0]
    assert data_pdb == pdb_id, \
        f"Mismatch Error: PDB_ID mismatch\n \
            mesh vertices: {pdb_id}_{chain_ids1} residue {resname}{res_id}\n \
            pmp_dataset.csv: {data_pdb}_{data_chain} residue {data_resname}{res_id}"
    assert data_chain == chain_ids1, \
        f"Mismatch Error\nchain id mismatch between residues (res_id: {res_id}) from mesh vertices and pmp_dataset.csv\n \
            mesh vertices: {pdb_id}_{chain_ids1} residue {resname}{res_id}\n \
            pmp_dataset.csv: {data_pdb}_{data_chain} residue {data_resname}{res_id}"
    assert data_resname == resname, \
        f"Mismatch Error\nresidue name mismatch between residues (res_id: {res_id}) from mesh vertices and pmp_dataset.csv\n \
            mesh vertices: {pdb_id}_{chain_ids1} residue {resname}{res_id}\n \
            pmp_dataset.csv: {data_pdb}_{data_chain} residue {data_resname}{res_id}"
    return True


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
