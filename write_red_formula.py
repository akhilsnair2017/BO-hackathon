from ase.db import connect
from pymatgen.core.composition import Composition
db=connect('undone.db')
for row in db.select():
    try:
        print(row.red_formula)
    except AttributeError:
        id=db.get(formula=row.formula).id
        db.update(id,red_formula=Composition(row.formula).reduced_formula)
