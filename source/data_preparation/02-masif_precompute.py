# Modified from files in MaSIF (Pablo Gainza - LPDI STI EPFL 2019)
# Released under an Apache License 2.0

import sys
import os
import numpy as np
from IPython.core.debugger import set_trace
import warnings 
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore",category=FutureWarning)
from masif_modules.read_data_from_surface import read_data_from_surface
# Configuration imports. Config should be in run_args.py
from default_config.masif_opts import masif_opts

np.random.seed(0)

print(sys.argv[2])

if len(sys.argv) <= 1:
    print("Usage: {config} "+sys.argv[0]+" {masif_pmp | masif_ensemble} PDBID_A")
    print("A or AB are the chains to include in this surface.")
    sys.exit(1)

masif_app = sys.argv[1]

if masif_app == 'masif_pmp':
    params = masif_opts['pmp']
    params['ply_chain_dir'] = masif_opts['ply_chain_dir']
elif masif_app == 'masif_ensemble':
    params = masif_opts['ensemble']
    params['ply_chain_dir'] = masif_opts['ply_chain_dir']

ppi_pair_list = [sys.argv[2]]

total_shapes = 0
total_ppi_pairs = 0
np.random.seed(0)
print('Reading data from input ply surface files.')

for ppi_pair_id in ppi_pair_list:

    all_list_desc = []
    all_list_coords = []
    all_list_shape_idx = []
    all_list_names = []
    idx_positives = []

    my_precomp_dir = params['masif_precomputation_dir']+ppi_pair_id+'/'
    if not os.path.exists(my_precomp_dir):
        os.makedirs(my_precomp_dir)
    
    # Read directly from the ply file.
    fields = ppi_pair_id.split('_')
    ply_file = {}
    ply_file['p1'] = masif_opts['ply_file_template'].format(fields[0], fields[1])

    if len (fields) == 2 or fields[2] == '':
        pids = ['p1']
    else:
        ply_file['p2']  = masif_opts['ply_file_template'].format(fields[0], fields[2])
        pids = ['p1', 'p2']
        
    # Compute shape complementarity between the two proteins. 
    rho = {}
    neigh_indices = {}
    mask = {}
    input_feat = {}
    theta = {}
    iface_labels = {}
    verts = {}

    for pid in pids:
        input_feat[pid], rho[pid], theta[pid], mask[pid], neigh_indices[pid], iface_labels[pid], verts[pid] = read_data_from_surface(ply_file[pid], params)

    # Save data only if everything went well. 
    for pid in pids: 
        np.save(my_precomp_dir+pid+'_rho_wrt_center', rho[pid])
        np.save(my_precomp_dir+pid+'_theta_wrt_center', theta[pid])
        np.save(my_precomp_dir+pid+'_input_feat', input_feat[pid])
        np.save(my_precomp_dir+pid+'_mask', mask[pid])
        # np.save(my_precomp_dir+pid+'_list_indices', neigh_indices[pid], allow_pickle=True )
        np.save(my_precomp_dir+pid+'_list_indices', np.array(neigh_indices[pid], dtype=object), allow_pickle=True)
        np.save(my_precomp_dir+pid+'_iface_labels', iface_labels[pid])
        # Save x, y, z
        np.save(my_precomp_dir+pid+'_X.npy', verts[pid][:,0])
        np.save(my_precomp_dir+pid+'_Y.npy', verts[pid][:,1])
        np.save(my_precomp_dir+pid+'_Z.npy', verts[pid][:,2])
