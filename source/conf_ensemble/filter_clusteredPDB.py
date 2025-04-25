"""
relocate_clusteredPDB.py: Move PDB of the frames sampled from HTMD and clustred by CLoNe to the data_preparation directory.
"""
import os
import sys
import shutil
import numpy as np
import pandas as pd
import re
from Bio.PDB import PDBParser, PDBIO
# Configuration imports. Config should be in run_args.py
from default_config.masif_opts import masif_opts


def read_table_to_matrix(filename):
    """
    Reads a structured text table and converts it into a numerical matrix.
    """
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
    # skip header and footer
    for line in lines[4:-1]:
        # Identify valid data rows (skip headers and separators)
        if re.match(r"^|\s*\d+\s*-\s*\d+", line):  
            # Extract numbers correctly (including negative, floating-point, and integer values)
            values = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
            # Convert to float or int appropriately
            data.append([float(val) if '.' in val else int(val) for val in values])
    return np.array(data)

def filter_cluster(ensemble_dir, summary_name="Summary_clusters.txt", tol=0.05):
    """
    filter the indices of "meaningful" clusters in the ensemble directory based on the frame number sampled during the simulation.
    """
    # Load the summary file
    summary_file = os.path.join(ensemble_dir, summary_name)
    summary_data = read_table_to_matrix(summary_file)
    cluster_n = summary_data.shape[0]
    total_frame_n = np.sum(summary_data[:,-2])
    # Filter the clusters based on the frame number
    meaningful_clusters = []
    for i in range(cluster_n):
        if summary_data[i,-1]/total_frame_n >= tol:
            meaningful_clusters.append(int(summary_data[i,0]))
    return meaningful_clusters

def remove_ions_water(input_pdb, output_pdb):
    """
    remove lines for water and ions in the input PDB file
    """
    hetero_ls = ['HOH', 'WAT', 'SOD', 'CLA', 'NA', 'CL', 'K', 'CA', 'MG', 'ZN']
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            if line[15:20].strip() in hetero_ls:
                # remove the line
                continue
            outfile.write(line)

def restore_chain_ids(input_pdb, raw_pdb, remove_ions_waters=False):
    # remove water and ions from the input PDB file
    if remove_ions_waters:
        input_pdb_ = input_pdb.rstrip('.pdb') + "_tmp.pdb"
        remove_ions_water(input_pdb, input_pdb_)
    else:
        input_pdb_ = input_pdb
    
    # parse the original and modified PDB files
    parser = PDBParser(QUIET=True)
    structure_orig = parser.get_structure("original", raw_pdb)
    structure_mod = parser.get_structure("modified", input_pdb_)

    # Map original protein chain IDs to modified structure
    orig_chains = [chain.id for chain in structure_orig[0]]
    mod_chains = [chain for chain in structure_mod[0]]

    # Reassign chain IDs: assume order is preserved
    protein_chain_map = {}
    chain_idx = 0

    for chain in mod_chains:
        is_protein = any(res.id[0] == ' ' for res in chain.get_residues())  # skip heteroatoms (e.g., water)
        if is_protein and chain_idx < len(orig_chains):
            protein_chain_map[chain.id] = orig_chains[chain_idx]
            chain.id = orig_chains[chain_idx]
            chain_idx += 1
        elif not is_protein and not remove_ions_waters:
            # assign new chain ID starting from 'X', 'Y', 'Z', etc.
            chain.id = chr(ord('Z') - (len(mod_chains) - chain_idx - 1))
    
    # Save the revised PDB file
    io = PDBIO()
    io.set_structure(structure_mod)
    io.save(input_pdb)
    
    # clean up
    if remove_ions_waters:
        os.remove(input_pdb_)



if __name__ == "__main__":
    params = masif_opts['ensemble']
    params['raw_pdb_dir'] = masif_opts["raw_pdb_dir"]
    params['pdb_chain_dir'] = masif_opts['pdb_chain_dir']
    params['clone_dir'] = masif_opts['clone_dir']
    params['ensemble_pdb_dir'] = masif_opts['ensemble_pdb_dir']
    # Get the list of PDB files in the directory
    PDB_CHAIN_ID = sys.argv[1]
    PDB_ID = PDB_CHAIN_ID.split("_")[0]
    cluster_dir = os.path.join(params['clone_dir'], PDB_CHAIN_ID, "DEFAULT_1")
    # Copy the PDB files to the data_preparation directory
    meainingful_clusters = filter_cluster(cluster_dir)
    
    if not os.path.exists(os.path.join(params['ensemble_pdb_dir'], PDB_CHAIN_ID)):
        os.makedirs(os.path.join(params['ensemble_pdb_dir'], PDB_CHAIN_ID))
    for idx in meainingful_clusters:
        pdb_file = f"Center_{idx:d}.pdb"
        shutil.copy(os.path.join(cluster_dir, pdb_file), os.path.join(params['ensemble_pdb_dir'], PDB_CHAIN_ID, pdb_file))
        # restore the chain_ids from the original PDB file & remove ions and waters
        restore_chain_ids(os.path.join(params['ensemble_pdb_dir'], PDB_CHAIN_ID, pdb_file),
                          os.path.join(params['raw_pdb_dir'], f"{PDB_ID}.pdb"),
                          remove_ions_waters=True,  # remove ions and waters since their chain_ids may overlap with protein chains
                          )
    # Copy the Summary_clusters.txt for frame number filteration in the future
    shutil.copy(os.path.join(cluster_dir, "Summary_clusters.txt"), os.path.join(params['ensemble_pdb_dir'], PDB_CHAIN_ID, "Summary_clusters.txt"))

        # ply_chain_dir = masif_opts['ply_chain_dir']+pdb_id+"_"+chain_ids1+"/Cluster_"+str(i)+"/"+f"frame_{j:04d}/"
        # pdb_chain_dir = masif_opts['pdb_chain_dir']+pdb_id+"_"+chain_ids1+"/Cluster_"+str(i)+"/"+f"frame_{j:04d}/"
