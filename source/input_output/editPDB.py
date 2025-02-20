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


def editPDB(
    infilename, chain_id
):
    # Edit the HTMD-generated pdb file protein chain id to be the same as its actual PDB chain_id
    # HTMD updates the chain id of: protein to 'A', HOH to 'B', Ion to 'C'
    # so we need to change the chain id of the protein to be the same as the input PDB for the following steps
    
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    check_chain_id(struct)

    for model in struct:
        for chain in model:
            # HTMD updates the chain id of: protein to 'A', HOH to 'B', Ion to 'C'
            if chain.id == 'A':
                chain.id = chain_id
    io = PDBIO()
    io.set_structure(struct)
    io.save(infilename)

