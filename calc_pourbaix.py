from ase.db import connect
from itertools import product, chain, combinations
from ase.formula import Formula
import pandas as pd
import re,os,sys
from new_pourbaix import Pourbaix
from pymatgen.core.composition import Composition
from ase.io import read,write
#reading the database
#get chemical symbols

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

def add_phases(material,iter_no,done_db,undone_db,phase_db):
    done_db=connect(done_db)
    undone_db=connect(undone_db)
    phase_db=connect(phase_db)
    phase_path=f'iter_{iter_no}_{material}/tree'
    done_formula=[row.red_formula for row in done_db.select()]
    for phase in os.listdir(phase_path):
        red_formula=Composition(phase).reduced_formula
        if red_formula not in done_formula:
            calc_path=os.path.join(phase_path,phase,'hse_scf')
            if completed_calc(calc_path)=='done':
                atoms=read(calc_path+'/aims.out')
                nspecies=len(Composition(phase).as_dict().keys())
                done_db.write(atoms=atoms,nspecies=nspecies,red_formula=red_formula)
                print(f'Added {phase} to done')
            else:
                atoms=phase_db.get(red_formula=red_formula).toatoms()
                undone_db.write(atoms=atoms,red_formula=red_formula)
                print(f'Unconverged {phase}')

def get_chemsys(material):
    elements = set(Formula(material).count().keys())
    elements.update(['H', 'O'])
    chemsys = list(
        chain.from_iterable(
            [combinations(elements, i+1) for i,_ in enumerate(list(elements))]
        )
    )
    
    chemsys = [item for item in chemsys if set(item) != {'H', 'O'}]
    print(f"Chemical systems: {chemsys}")
    return chemsys

def parse_formula(formula):
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    return dict([(el, int(cnt) if cnt else 1) for el, cnt in elements])

#get formation energy for the target material
def get_formation_energy(material,db_name,per_atom=False):
    element_energies = {}
    db=connect(db_name)
    for row in db.select():
        if row.nspecies == 1:  # Single element
            element_energies[list(parse_formula(row.formula).keys())[0]] = row.energy / row.natoms
    count=Formula(material).count()
    elem_energy = sum([element_energies[s] * n for s, n in count.items()])
    energy=db.get(formula=material).energy
    natoms=db.get(formula=material).natoms
    form_energy=(energy-elem_energy)
    if per_atom:
        return form_energy/natoms
    else:
        return form_energy

#get dictionary of all solid phases w.r.t the target material
def get_solid_refs(material, db_name):
    from ase.db import connect
    db = connect(db_name)
    element_energies = {}
    refs = {}
    for subsys in get_chemsys(material):
        nspecies = len(subsys)
        query_str = ",".join(subsys) + f',nspecies={nspecies}'
        for row in db.select(query_str):
            energy = row.energy
            ref = row.formula
            energy=row.energy
            refs[ref]=energy
    return refs

#get solvated references from ASE database
def get_solvated_refs(material):
    from ase.phasediagram import solvated
    ref_dct = {}
    solv = solvated(material)
    for name, energy in solv:
        if name not in ['H+(aq)', 'H2O(aq)']:
            ref_dct[name] = energy
    return ref_dct

#get complete formation energy dictionary for Pourbaix analysis 
def get_refs_formation(material,db_name):
    refs=get_solid_refs(material,db_name)
    form_dict={}
    for phase in refs.keys():
        form_dict[phase]=get_formation_energy(phase,db_name)
    form_dict.update(get_solvated_refs(material))
    return form_dict

#get Pourbaix Energy and plot Pourbaix diagram
def get_pourbaix_energy(material,db_name,U=1.23,pH=0):
    db=connect(db_name)
    refs=get_refs_formation(material,db_name)
    print(f"Formation Energy Dictionary: {refs}\n")
    natoms=db.get(formula=material).natoms
    pb=Pourbaix(material,refs)
    energies = pb._decompose(1.23, 0)
    phases = pb.phases
#     for e, p in zip(energies, phases):
#         print(p.equation(),':',round(e,3),'eV')
    pb.plot(Urange=[-3,3],pHrange=[0,14],cap=[0,1], include_text=False, figsize=[8, 6], include_h2o=True, labeltype='phases',savefig=f'{material}.png')
#     pb.plot(Urange=[-3,3],pHrange=[0,14])
    gpbx=pb.get_pourbaix_energy(U,pH)[0]
    reduced_composition=Composition(material).reduced_composition
    n_atoms_reduced=reduced_composition.num_atoms
    gpbx_per_atom=gpbx/n_atoms_reduced
    return gpbx_per_atom

def update_train_test(material, iter_no, train_df, test_df, new_gpbx):
    next_iter = int(iter_no) + 1
    x_test = test_df.loc[test_df['formula']==material]
    x_test.insert(1, 'g_pbx (eV/atom)', new_gpbx)

    if isinstance(x_test, pd.Series):
        x_test = pd.DataFrame(x_test).T
    if material not in train_df['formula']:
        train_df = pd.concat([train_df, x_test], ignore_index=True)
        test_df.drop(x_test.index, inplace=True) 
        test_df.to_csv(f'test_{next_iter}.csv', index=False)
        train_df.to_csv(f'train_{next_iter}.csv', index=False)
        print(f'Updated train_{next_iter}.csv')
        print(f'Updated test_{next_iter}.csv')
    else:
        print(f"{material} already present in training")


done_db='done.db'
undone_db='undone.db'
phase_db='phases.db'
material=sys.argv[2]
iter_no=sys.argv[1]
train_df=pd.read_csv(f'train_{iter_no}.csv')
test_df=pd.read_csv(f'test_{iter_no}.csv')
add_phases(material,iter_no,done_db,undone_db,phase_db)
new_gpbx=get_pourbaix_energy(material,done_db)
print(f"Adding new Gpbx of {material} = {new_gpbx}")
update_train_test(material,iter_no,train_df,test_df,new_gpbx)
