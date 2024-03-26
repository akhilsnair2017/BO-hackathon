import os
import taskblaster as tb
from ase.build import molecule
#def tb.node(name, **kwargs):
#    return tb.tb.node(f'fhi_h2o_splitting.workflow.tasks.{name}', **kwargs)

#def tb.node(name,**kwargs):
#    return tb.tb.node(name,**kwargs)

@tb.workflow
class DummyWorkflow:
    atoms = tb.var()

    @tb.task
    def pbe_relax(self):
        """
        PBE relaxation : Relax geometry with PBE
        """
        return tb.node('pbe_light_relax', atoms=self.atoms)

    @tb.task
    def hse_scf(self):
        """
#       SCF: SCF calculation taking path from pbe_relax as input
#        """
        return tb.node('hse_light_scf',
                    atoms=self.pbe_relax)
#
#

@tb.parametrize_glob('*/material')
def workflow(material):
    return DummyWorkflow(atoms=material)

