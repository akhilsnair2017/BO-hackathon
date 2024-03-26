from ase import Atoms
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
#ASE terms
from ase import Atoms
from ase.db import connect
from ase.io import read,write, Trajectory
from ase.optimize import BFGS as BFGS
from ase.calculators.aims import Aims,AimsProfile
from ase.build import molecule,bulk
from ase import Atoms
from ase.calculators.aims import AimsCube
from ase.optimize import QuasiNewton
from pathlib import Path
import numpy as np
def get_pbe_par():
    aims_pars={'xc':'pbe',
    'relax_geometry':'bfgs 1E-2',
    'relax_unit_cell':'full',
    'compute_forces':'.true.',
    'sc_accuracy_rho':'1E-5',
    'sc_accuracy_eev':'1E-3',
    'sc_accuracy_etot':'1E-6',
    'relativistic':"atomic_zora scalar",
    'adjust_scf':'always 1',
    'sc_iter_limit':500,
    'k_grid_density': 5,
    'run_command':'srun aims.230629.scalapack.mpi.x >aims.out'}
    return aims_pars

def get_hse_scf_par():
    aims_pars={'xc':'hse06 0.11',
    'hse_unit':'bohr-1',
    'hybrid_xc_coeff':'0.25',
    'sc_accuracy_rho':'1E-5',
    'sc_accuracy_eev':'1E-3',
    'sc_accuracy_etot':'1E-6',
    'relativistic':"atomic_zora scalar",
    'adjust_scf':'always 1',
    'sc_iter_limit':500,
    'k_grid_density': 5,
    'run_command':'srun aims.230629.scalapack.mpi.x >aims.out'}
    return aims_pars
#
#def get_hse_relax_par():
#    aims_pars={'xc':'hse06 0.11',
#    'hse_unit':'bohr-1',
#    'hybrid_xc_coeff':'0.25',
#    'relax_geometry':'bfgs 1E-2',
#    'relax_unit_cell':'full',
#    'compute_forces':'.true.',
#    'sc_accuracy_rho':'1E-5',
#    'sc_accuracy_eev':'1E-3',
#    'sc_accuracy_etot':'1E-6',
#    'relativistic':"atomic_zora scalar",
#    'adjust_scf':'always 1',
#    'k_grid_density': 5,
#    'run_command':'srun aims.230629.scalapack.mpi.x >aims.out'}
#    return aims_pars
#
def pbe_light_relax(atoms):
    aims_par=get_pbe_par()
    aims_par['species_dir']="/home/becaks23/softwares/FHIaims/species_defaults/defaults_2020/light"
    initial_magnetic_moments = np.array([round(moment) for moment in atoms.get_initial_magnetic_moments()])
    atoms.set_initial_magnetic_moments(initial_magnetic_moments)
    if initial_magnetic_moments is not None and (initial_magnetic_moments != 0).any():
        aims_par['spin'] = 'collinear'
    aims_calc=Aims(**aims_par)
    atoms.set_calculator(aims_calc)
    energy=atoms.get_potential_energy()
    final_geom=read('aims.out')
    initial_geom=read('geometry.in')
    final_geom.set_initial_magnetic_moments(initial_geom.get_initial_magnetic_moments())
    return final_geom
##Task 2
def hse_light_scf(atoms):
    aims_par=get_hse_scf_par()
    aims_par['species_dir']="/home/becaks23/softwares/FHIaims/species_defaults/defaults_2020/light"
    initial_magnetic_moments = atoms.get_initial_magnetic_moments()
#    atoms.set_initial_magnetic_moments(initial_magnetic_moments)
    if initial_magnetic_moments is not None and (initial_magnetic_moments != 0).any():
        aims_par['spin'] = 'collinear'
    aims_calc=Aims(**aims_par)
    atoms.set_calculator(aims_calc)
    energy=atoms.get_potential_energy()
    return read('aims.out')
#
#def hse_light_relax(atoms):
#    aims_par=get_hse_relax_par()
#    aims_par['species_dir']="/home/becaks23/softwares/FHIaims/species_defaults/defaults_2020/light"
#    initial_magnetic_moments = atoms.get_initial_magnetic_moments()
#    if initial_magnetic_moments is not None and (initial_magnetic_moments != 0).any():
#        aims_par['spin'] = 'collinear'
#    aims_calc=Aims(**aims_par)
#    atoms.set_calculator(aims_calc)
#    energy=atoms.get_potential_energy()
#    return read('aims.out')
