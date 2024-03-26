import numpy as np
from ase.formula import Formula
from ase.db import connect
import re,os,subprocess,shutil
from itertools import product, chain, combinations, permutations

class dft_run:
    def __init__(self, material,done_db,design_db,undone_db,iter_no):
        self.material = material
        self.done_db=done_db
        self.design_db = design_db
        self.undone_db=undone_db
        self.iter_no=iter_no

    def get_chemsys(self):
        elements = set(Formula(self.material).count().keys())
        elements.update(['H', 'O'])
        chemsys = list(
            chain.from_iterable(
                [combinations(elements, i+1) for i,_ in enumerate(list(elements))]
            )
        )
        return chemsys


    def get_pb_solid_refs(self):
        done_db=connect(self.done_db)
        undone_db=connect(self.undone_db)
        design_db=connect(self.design_db)
        done_list=[row.red_formula for row in done_db.select()]
        undone_list=[row.red_formula for row in undone_db.select()]
        refs = []
        for subsys in self.get_chemsys():
            nspecies = len(subsys)
            query_str = ",".join(subsys) + f',nspecies={nspecies}'
            for row in design_db.select(query_str):
                if row.red_formula in done_list:
                    print(f"{row.formula} already done")
                elif row.red_formula in undone_list:
                    print(f"{row.formula} not converged")
                else:
                    refs.append(row.formula)
        return refs
    
    def write_phases(self):
        calc_path = f'iter_{self.iter_no}_{self.material}'
        cur_dir=os.getcwd()
        os.makedirs(calc_path, exist_ok=True)
        shutil.copy('tasks.py', calc_path)
        shutil.copy('workflow.py', calc_path)
        shutil.copy('totree.py', calc_path)
        shutil.copy('tb.sh',calc_path)
        shutil.copy('rename.sh',calc_path)
        shutil.copy('submit.sh',calc_path)
        refs = list(set(self.get_pb_solid_refs()))  # Fix: Call the method to get refs
        print("Refs",refs)
        db = connect(f'{calc_path}/phases.db')
        design_db=connect(self.design_db)
        row_ids = [design_db.get(formula=ref).id for ref in refs]
        for row_id in row_ids:
            atoms =design_db.get(id=row_id).toatoms()
            db.write(atoms=atoms)
        os.chdir(calc_path)
        subprocess.run('conda run -n asr_env tb init fhi_h2o_splitting', shell=True)
        subprocess.run('conda run -n asr_env tb workflow totree.py', shell=True)
        subprocess.run('conda run -n asr_env tb workflow workflow.py', shell=True)
        subprocess.run(['bash', 'rename.sh'], cwd='./')
#        subprocess.run('conda run -n asr_env tb submit tree/*', shell=True)
        os.chdir(cur_dir)
        return refs

#est=dft_run('Fe4As4O16','done.db','phases.db','undone.db',3)
#print(est.get_pb_solid_refs())
