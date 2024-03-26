from ase.db import connect
from pymatgen.core.composition import Composition

def remove_ehull(db_file,e_hull_cutoff):
   db=connect(db_file)
   for row in db.select():
       if row.mp_ehull>e_hull_cutoff:
            db.delete([row.id])

def remove_natoms(db_file,natoms_cutoff):
    db=connect(db_file)
    for row in db.select():
        if row.natoms>natoms_cutoff:
            db.delete([row.id])

def remove_duplicates_from_db(db_file, key):
    db = connect(db_file)
    seen_values = set()
    entries_to_delete = []
    for row in db.select():
        value = row.get(key)
        if value not in seen_values:
            seen_values.add(value)
        else:
            entries_to_delete.append(row.id)

    for entry_id in entries_to_delete:
        db.delete([entry_id])
        print(f"Deleted entry with ID {entry_id}.")
#
def remove_already(train_db,test_db):
    train_db=connect(train_db)
    test_db=connect(test_db)
    train_formula=[row.red_formula for row in train_db.select()]
    for test_row in test_db.select():
        if test_row.red_formula in train_formula:
            print(f"Deleting {test_row.formula}")
            test_db.delete([test_row.id])
## Specify your ASE database file and the key to check for duplicates
def delete_elements(db):
    db=connect(db)
    remove_symbols = [ 'He', 'N', 'F', 'Ne', 'Cl', 'Ar', 'Kr', 'Xe', 'Rn', 'B', # Ideal Gases
                    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',  # Lanthanides
                    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',  # Actinides
                    'F', 'Cl', 'Br', 'I', 'At','Bi','Tl','Pb','Po']  # Halogens

    for row in db.select():
        comp=Composition(row.formula).as_dict()
        for elem in remove_symbols:
            if elem in comp.keys():
                print(row.formula)
                db.delete([row.id])
#
your_db_file = 'done.db'
key_to_check = 'formula'  # Replace with the key you want to check
#remove_ehull(your_db_file,0.00)
#remove_natoms(your_db_file,50)
#remove_duplicates_from_db(your_db_file, key_to_check)
remove_already('done.db','phases.db')
#delete_elements('oxides.db')
