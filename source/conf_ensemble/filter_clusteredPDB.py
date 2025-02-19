"""
relocate_clusteredPDB.py: Move PDB of the frames sampled from HTMD and clustred by CLoNe to the data_preparation directory.
"""
import os
import sys
import shutil
import numpy as np
import pandas as pd
import re
# Configuration imports. Config should be in run_args.py
from default_config.masif_opts import masif_opts


def read_table_to_matrix(filename):
    """
    Reads a structured text table and converts it into a numerical matrix.
    """
    data = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    # Extract only relevant data lines (skip headers and separators)
    for line in lines:
        if re.match(r"^\|\s*\d+", line):  # Match rows that start with "| <number>"
            values = re.findall(r"[-+]?\d*\.\d+|\d+", line)  # Extract numbers
            data.append(list(map(float, values)))  # Convert to float
    # Convert to numpy array: each row is a cluster center, each column is a feature, [:,0] = cluster center index (1-based)
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
            meaningful_clusters.append(summary_data[i,0])
    return meaningful_clusters


if __name__ == "__main__":
    params = masif_opts['site']
    params['pdb_chain_dir'] = masif_opts['pdb_chain_dir']
    params['clone_dir'] = masif_opts['clone_dir']
    params['ensemble_pdb_dir'] = masif_opts['ensemble_pdb_dir']
    # Get the list of PDB files in the directory
    PDB_CHAIN_ID = sys.argv[1]
    cluster_dir = os.path.join(params['clone_dir'], PDB_CHAIN_ID, "DEFAULT_1")
    # Copy the PDB files to the data_preparation directory
    meainingful_clusters = filter_cluster(cluster_dir)
    
    if not os.path.exists(os.path.join(params['ensemble_pdb_dir'], PDB_CHAIN_ID)):
        os.makedirs(os.path.join(params['ensemble_pdb_dir'], PDB_CHAIN_ID))
    for idx in meainingful_clusters:
        pdb_file = f"Center_{idx:d}.pdb"
        shutil.copy(os.path.join(cluster_dir, pdb_file), os.path.join(params['ensemble_pdb_dir'], PDB_CHAIN_ID, pdb_file))
    # Copy the Summary_clusters.txt for frame number filteration in the future
    shutil.copy(os.path.join(cluster_dir, "Summary_clusters.txt"), os.path.join(params['ensemble_pdb_dir'], PDB_CHAIN_ID, "Summary_clusters.txt"))

        # ply_chain_dir = masif_opts['ply_chain_dir']+pdb_id+"_"+chain_ids1+"/Cluster_"+str(i)+"/"+f"frame_{j:04d}/"
        # pdb_chain_dir = masif_opts['pdb_chain_dir']+pdb_id+"_"+chain_ids1+"/Cluster_"+str(i)+"/"+f"frame_{j:04d}/"
