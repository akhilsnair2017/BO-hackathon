from ase.db import connect
ei_db=connect('ei_done_first.db')
done_db=connect('done.db')
phase_list=['Cu3Pt', 'Cu2Pt2O4', 'CuPt', 'CuPt7', 'CuPt3', 'Cu8Pt2O10']
for phase in phase_list:
    row=ei_db.get(formula=phase)
    atoms=row.toatoms()
    red_formula=row.red_formula
    nspecies=row.nspecies
    done_db.write(atoms=atoms,red_formula=red_formula,nspecies=nspecies)
