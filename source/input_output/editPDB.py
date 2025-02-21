"""
editPDB.py: Edit the HTMD-generated pdb file chain ids to be the same as the chain_ids1 of the input PDB
ByungUk Park - UW-Madison
Released under an Apache License 2.0
"""
from Bio.PDB import *
from IPython.core.debugger import set_trace

def check_chain_id(struct):
    # Check whether the input PDB (HTMD-generated) has more than 3 chains (A, B, C)
    chains = Selection.unfold_entities(struct, "C")
    for chain in chains:
        try:
            chain.id in ['A', 'B', 'C']     # HTMD updates the chain id of: protein to 'A', HOH to 'B', Ion to 'C'
        except:
            set_trace()
            ValueError('\n      The input PDB has more than 3 chains (A, B, C).\n      HTMD-generated PDB should have A for single protein, B for water, C for ions. Please check the input PDB')

def initialize_chain_id(struct):
    # Initialize the chain id of the input PDB (HTMD-generated) to the default chain id befpre editting the chain id
    # to avoid ValueError: change id because the id is already used for a sibling of this entity
    # e.g., ValueError: Cannot change id from `A` to `B`. The id `B` is already used for a sibling of this entity
    chains = Selection.unfold_entities(struct, "C")
    default_chain_id_dic = {'A': 'PROA', 'B': 'SOL', 'C': 'ION'}
    for chain in chains:
        if chain.id in default_chain_id_dic.keys():
            chain.id = default_chain_id_dic[chain.id]
    return struct

def remove_hoh_ion(struct):
    # Remove HOH and ION from the input PDB (HTMD-generated)
    chains = Selection.unfold_entities(struct, "C")
    for chain in chains:
        if chain.id in ['HOH', 'ION']:
            struct[0].detach_child(chain.id)
    return struct

def editPDB(
    infilename, chain_id
):
    # Edit the HTMD-generated pdb file protein chain id to be the same as its actual PDB chain_id
    # HTMD updates the chain id of: protein to 'A', HOH to 'B', Ion to 'C'
    # so we need to change the chain id of the protein to be the same as the input PDB for the following steps
    
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    check_chain_id(struct)
    struct = initialize_chain_id(struct)
    struct = remove_hoh_ion(struct)
    chains = Selection.unfold_entities(struct, "C")

    for chain in chains:
        # HTMD updates the chain id of: protein to 'A', HOH to 'B', Ion to 'C'
        if chain.id == 'PROA':
            chain.id = chain_id
    io = PDBIO()
    io.set_structure(struct)
    io.save(infilename)

