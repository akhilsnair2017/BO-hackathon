from ase.db import connect
import glob, os, sys
from pymatgen.core.composition import Composition
base_path = os.getcwd()

# Construct input directory path
input_dir = os.path.join(base_path, sys.argv[1])

# Use glob to find directories starting with 'iter'
target_directories = glob.glob(os.path.join(base_path, 'iter*'))

# Remove the input directory from the list
target_directories.remove(input_dir)

# Initialize lists for input and target phases
input_phases = [row.formula for row in connect(f'{input_dir}/phases.db').select()]
target_phases = {}

# Loop over target directories and extract phases
for target_dir in target_directories:
    target_phases[target_dir.split('/')[-1]] = [row.formula for row in connect(f'{target_dir}/phases.db').select()]

remove=[]
# Loop over input phases and check if they are in target phases
for phase in input_phases:
    found_in_target = False
    for target_dir, target_phases_list in target_phases.items():
        if phase in target_phases_list:
            print(f"{phase} already in {target_dir}")
            found_in_target = True
            remove.append(phase)
            break
    if not found_in_target:
        print(f"{phase} not found in any target directory")


done_db=connect('done.db')
done_red_formula=[row.red_formula for row in done_db.select()]
for phase in input_phases:
    if Composition(phase).reduced_formula in done_red_formula:
        print(f"{phase} in done.db")
        remove.append(phase)

print(f"Remove: {remove}")
