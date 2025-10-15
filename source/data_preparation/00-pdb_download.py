#!/usr/bin/python
import Bio
from Bio.PDB import * 
import sys
import importlib
import os
from default_config.masif_opts import masif_opts
# Local includes
from input_output.protonate import protonate

# Modified from files in MaSIF (Pablo Gainza - LPDI STI EPFL 2019)
# Released under an Apache License 2.0

if len(sys.argv) <= 1: 
    print("Usage: "+sys.argv[0]+" PDBID_A_B")
    print("A or B are the chains to include in this pdb.")
    sys.exit(1)

if not os.path.exists(masif_opts['raw_pdb_dir']):
    os.makedirs(masif_opts['raw_pdb_dir'])

if not os.path.exists(masif_opts['tmp_dir']):
    os.mkdir(masif_opts['tmp_dir'])

in_fields = sys.argv[1].split('_')
pdb_id = in_fields[0]

# Download pdb 
pdbl = PDBList()
obsolete_pdbs = pdbl.get_all_obsolete()
# Check whether obsolete PDB ID or not
if pdb_id in obsolete_pdbs:
    pdb_filename = pdbl.retrieve_pdb_file(pdb_id, obsolete=True, pdir=masif_opts['tmp_dir'], file_format='pdb')
else:
    pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=masif_opts['tmp_dir'], file_format='pdb')

# Rename the file
os.rename('{}/pdb{}.ent'.format(masif_opts["tmp_dir"], pdb_id.lower()), '{}/{}.pdb'.format(masif_opts["tmp_dir"], pdb_id))
pdb_filename = '{}/{}.pdb'.format(masif_opts["tmp_dir"], pdb_id)

# Protonate with reduce, if hydrogens included.
protonated_file = masif_opts['raw_pdb_dir']+"/"+pdb_id+".pdb"
protonate(pdb_filename, protonated_file)
pdb_filename = protonated_file

