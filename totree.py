import numpy as np
from ase.db import connect
import taskblaster as tb
import fhi_h2o_splitting
import importlib
import json
import typing
from pathlib import Path
from abc import ABC, abstractmethod
from taskblaster.hashednode import PackedReference
from taskblaster.encoding import encode_object, decode_object
from taskblaster import JSONCodec

def add_materials_from_db(filename: str) -> dict:
    """
    Helper function for a high-throughput workflow that loads an ase.db,
    reads the structures, and adds them to the asr workflow.

    :param filename: the file path + filename of the ase database.

    :return: a dictionary containing chemical formula: ase atoms object for
    each structure.
    """
    cforms = []
    db_structures = {}
    # connect local db file and
    with connect(filename) as con:
        for row in con.select():
            atoms = row.toatoms().copy() # extra copy to remove calculators
            cform = atoms.get_chemical_formula()
            cforms.append(cform)
            db_structures[cform] = atoms

    # tell the user to change the naming convention for the dictionary if 2
    # structs have same chem formula
    if len(cforms) != len(np.unique(cforms)):
        print()
        sys.exit('The naming convention for the dictionary has duplicates and a structure was overwritten. '
                 'Change the naming convention to prevent this.')

    return db_structures


filename = 'phases.db'
workflow = tb.totree(add_materials_from_db(filename), name='material')
                                                                       
