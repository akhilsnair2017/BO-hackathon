from ase.db import connect
from itertools import product, chain, combinations
from ase.formula import Formula
import pandas as pd
import re,os,sys
#from new_pourbaix import Pourbaix
from pymatgen.core.composition import Composition
from ase.io import read,write
#reading the database
#get chemical symbols
parent_db='pbe_final.db'

def completed_calc(path):
    calc_status=''
    if os.path.exists(path+'/aims.out'):
        with open(path+'/aims.out','r',encoding='windows-1252') as f:
            f=f.readlines()
            if '          Have a nice day.\n' in f:
                calc_status=calc_status+'done'
            else:
                calc_status=calc_status+'not_done'
    return calc_status

def add_phases(db_name):
    db=connect(db_name)
    formula_list=[row.red_formula for row in db.select()]
    for dir in os.listdir():
        if dir.startswith('iter'):
            phase_path=os.path.join(os.getcwd(),dir,'tree')
            for phase in os.listdir(phase_path):
                calc_path=os.path.join(phase_path,phase,'pbe_relax')
                if completed_calc(calc_path)=='done' and Composition(phase).reduced_formula not in formula_list:
                    atoms=read(calc_path+'/aims.out')
                    red_formula=Composition(phase).reduced_formula
                    nspecies=len(Composition(phase).as_dict().keys())
                    db.write(atoms=atoms,nspecies=nspecies,red_formula=red_formula)

add_phases('pbe_final.db')
