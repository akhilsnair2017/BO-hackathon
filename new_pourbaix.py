from collections import Counter
from itertools import product, chain, combinations
from typing import Union
import re

import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix

from ase.units import kB
from ase.formula import Formula


CONST = kB * np.log(10) # Nernst constant

PREDEF_ENERGIES = {     # Default chemical potentials
    'H+': 0.0,          # for water, protons and electrons
    'e-': 0.0,
    'H2O': -2.4583
}

U_STD_AGCL = 0.222  # Standard redox potential of AgCl electrode


def initialize_refs(refs_dct):
    """Convert dictionary entries to Species instances"""
    refs = {}
    for name, energy in refs_dct.items():
        spec = Species(name)
        spec.set_chemical_potential(energy, None)
        refs[spec.name] = spec
    return refs


def get_product_combos(reactant, refs):
    """Obtain, from the available references, different combinations
       of products that are compatible with valid (electro)chemical reactions.
    """
    array = [[] for i in range(len(reactant.elements))]
    for ref in refs.values():
        contained = ref.contains(reactant.elements)
        for w in np.argwhere(contained).flatten():
            array[w].append(ref)
    return product(*array)


def get_phases(reactant, refs, T, conc, counter, normalize=True):
    """Obtain all the possible decomposition pathways
       for a given reactant as a collection of RedOx objects.
    """
    phases = []
    phase_matrix = []
    reac_elem = [-reactant._count_array(reactant.elements)]

    for products in get_product_combos(reactant, refs):
        if len(np.unique(products)) < len(products):
            products = (products[0],)

        prod_elem = [p._count_array(reactant.elements) for p in products]
        elem_matrix = np.array(reac_elem + prod_elem).T
        solutions = Matrix(elem_matrix).nullspace()

        for solution in solutions:
            coeffs = np.array(solution).flatten()
            if all(coeffs > 0):
                if normalize:
                    coeffs /= abs(coeffs[0])
                coeffs[0] = -coeffs[0]
                species = (reactant, *products)
                phase = RedOx(species, coeffs, T, conc, counter)
                phases.append(phase)
                phase_matrix.append(phase._vector)

    return phases, np.array(phase_matrix).astype('float64')


def edge_detection(array):
    """Add phase boundaries to a Pourbaix diagram."""
    from collections import defaultdict
    edges_raw = defaultdict(list)
    edges = defaultdict(list)

    for i in range(array.shape[0] - 1):
        for j in range(array.shape[1] - 1):
            xpair = (array[i, j], array[i+1, j])
            ypair = (array[i, j], array[i, j+1])
            for pair in [xpair, ypair]:
                if np.ptp(pair) != 0:
                    edges_raw[pair].append([i+1, j])

    for pair, values in edges_raw.items():
        varr = np.array(values)
        left = values[np.argmin(varr[:, 1])]
        right = values[np.argmax(varr[:, 1])]
        if left == right:
            left = values[np.argmin(varr[:, 0])]
            right = values[np.argmax(varr[:, 0])]
        edges[pair] = np.array([left, right]).T

    return edges


def add_numbers(ax, text):
    """Add phase indexes to the different domains of a Pourbaix diagram."""
    import matplotlib.patheffects as pfx
    for i, (x, y, prod) in enumerate(text):
        txt = ax.text(
            y, x, f'{i}', 
            fontsize=20,
            horizontalalignment='center'
        )
        txt.set_path_effects([pfx.withStroke(linewidth=2.0, foreground='w')])
    return


def add_labels(ax, text):
    """Add phase indexes to the different domains of a Pourbaix diagram."""
    import matplotlib.patheffects as pfx
    for i, (x, y, prod) in enumerate(text):
        label = format_label(prod)
        annotation = ax.annotate(
                label, xy=(y, x), color='w',
                fontsize=16, horizontalalignment='center'
        )
        annotation.set_path_effects([pfx.withStroke(linewidth=2.0, foreground='k')])
        annotation.draggable()
        ax.add_artist(annotation)
    return


def format_label(products):
    """Obtain phase labels formatted in LaTeX style."""
    formatted = []
    for p in products:
        label = re.sub(r'(\S)([+-]+)', r'\1$^{\2}$', p)
        label = re.sub(r'(\d+)', r'$_{\1}$', label)
        for symbol in ['+', '-']:
            count = label.count(symbol)
            if count > 1:
                label = label.replace(count*symbol, f'{count}{symbol}')
            if count == 1:
                label = label.replace(count*symbol, symbol)
        formatted.append(label)
    label = ', '.join(f for f in formatted)
    return label


def add_text(ax, text, offset=0):
    """Add phase labels to the right of the diagram"""
    import textwrap

    textlines = []
    for i, (x, y, prod) in enumerate(text):
        formatted = []
        label = format_label(prod)
        textlines.append(
            textwrap.fill(
                f'({i})  {label}',
                width=40,
                subsequent_indent='      '
            )
        )
    text = "\n".join(textlines)
    plt.gcf().text(
            0.75 + offset, 0.5,
            text,
            fontsize=18,
            va='center',
            ha='left')
    return 0


def add_redox_lines(axes, pH, counter, color='k'):
    """Add water redox potentials to a Pourbaix diagram"""
    const = - 0.5 * PREDEF_ENERGIES['H2O']
    corr = {
        'SHE': 0,
        'RHE': 0,
        'Pt': 0,
        'AgCl': -U_STD_AGCL,
    }
    kwargs = {
        'c': color,
        'ls': '--',
        'zorder': 2
    }
    if counter in ['SHE', 'AgCl']:    
        slope = -59.2e-3
        axes.plot(pH, slope*pH + corr[counter], **kwargs)
        axes.plot(pH, slope*pH + const + corr[counter], **kwargs)
    elif counter in ['Pt', 'RHE']:
        axes.axhline(0 + corr[counter], **kwargs)
        axes.axhline(const + corr[counter], **kwargs)
    else:
        raise ValueError('The specified counter electrode doesnt exist')
    return 0


class Species:
    """Class representing an individual chemical species, grouping relevant properties.

    Initialization
    --------------

    formula: str
        The chemical formula of the species (e.g. ``ZnO``).
        For solid species, the formula is automatically reduced to the unit formula,
        and the chemical potential normalized accordingly.
        Acqueous species are specified by expliciting
        all the positive or negative charges and by appending ``(aq)``.
        Parentheses for grouping functional groups are acceptable.
            e.g. 
                Be3(OH)3[3+]    ➜  wrong
                Be3(OH)3+++     ➜  wrong
                Be3(OH)3+++(aq) ➜  correct
                Be3O3H3+++(aq)  ➜  correct
    fmt: str
        Formula formatting according to the available options in 
        ase.formula.Formula

    """
    def __init__(self, formula, fmt='metal'):
        self.aq = formula.endswith('(aq)')
        formula_strip = formula.replace('(aq)', '').rstrip('+-')
        self.charge = formula.count('+') - formula.count('-')
        formula_obj = Formula(formula_strip, format=fmt)
        self._count = formula_obj.count()

        if self.aq:
            self.name = formula
            self.n_fu = 1
            self.count = self._count
        else:
            reduced, self.n_fu = formula_obj.reduce()
            self.count = reduced.count()
            self.name = str(reduced)

        self._elements = [elem for elem in self.count]
        self.elements = [elem for elem in self._elements if elem not in ['H', 'O']]
        self.natoms = sum(self.count.values())
        self.energy = None
        self.mu = None

    def get_chemsys(self):
        """Get the possible combinations of elements based on the stoichiometry.
           Useful for database queries.
        """
        elements = set(self.count.keys())
        elements.update(['H', 'O'])
        chemsys = list(
            chain.from_iterable(
                [combinations(elements, i+1) for i,_ in enumerate(list(elements))]
            )
        )
        return chemsys

    def balance_electrochemistry(self):
        """Obtain number of H2O, H+, e- "carried" by the species
           in electrochemical reactions.
        """
        n_H2O = -self.count.get('O', 0)
        n_H = -2 * n_H2O - self.count.get('H', 0)
        n_e = n_H + self.charge
        return n_H2O, n_H, n_e

    def _count_array(self, elements):
        return np.array([self.count.get(e, 0) for e in elements])

    def contains(self, elements):
        return [True if elem in self.elements else False for elem in elements]

    def get_fractional_composition(self, elements):
        """Obtain the fractional content of each element."""
        N_all = sum(self.count.values())
        N_elem = sum([self.count.get(e, 0) for e in elements])
        return N_elem / N_all

    def get_formation_energy(self, energy, refs):
        """Evaluate the formation energy.

        Requires the material chemical potential (e.g. DFT total energy)
        and a dictionary with the elemental references and corresponding
        chemical potentials.
        """
        elem_energy = sum([refs[s] * n for s, n in self._count.items()])
        hof = (energy - elem_energy) / self.n_fu
        return hof

    def set_chemical_potential(self, energy, refs=None):
        """Set the chemical potential of the species.

        If a reference dictionary with the elements and their chemical potentials
        is provided, the chemical potential will be calculated by as a formation
        energy. Otherwise the provided energy will be used directly as the
        chemical potential.
        """
        self.energy = energy
        if refs is None:
            self.mu = energy / self.n_fu
        else:
            self.mu = self.get_formation_energy(energy, refs)

    def __repr__(self):
        return f'({self.name}, μ={self.mu})'

    def __lt__(self, other):
        return self.name < other.name

    def __gt__(self, other):
        return self.name > other.name


class RedOx:
    def __init__(self, species, coeffs,
                 T=298.15, conc=1e-6,
                 counter='SHE'):
        """RedOx class representing an (electro)chemical reaction.

        Initialization
        --------------

        species
            The reactant and products excluded H2O, protons and electrons.

        coeffs
            The stoichiometric coefficients of the species.
            Positive coefficients are associated with the products,
            negative coefficients with the reactants.

        T
            The temperature in Kelvin.

        conc
            The concentration of ionic species.
            

        Relevant methods
        ----------------

        equation()
            Print the chemical equation of the reaction.

        get_free_energy(U, pH):
            Obtain the reaction free energy at a given applied potential U and pH

        """

        alpha = CONST * T   # 0.059 eV @ T=298.15K
        const_term = 0
        pH_term = 0
        U_term = 0
        self.species = Counter()

        for spec, coef in zip(species, coeffs):
            self.species[spec.name] = coef
            amounts = spec.balance_electrochemistry()

            const_term += coef * ( \
                spec.mu + alpha * (spec.aq * np.log10(conc)))
            pH_term += - coef * alpha * amounts[1]
            U_term += - coef * amounts[2]

            for name, n in zip(['H2O', 'H+', 'e-'], amounts):
                const_term += coef * n * PREDEF_ENERGIES[name]
                self.species[name] += coef * n

        const_corr, pH_corr = self.get_counter_correction(counter, alpha)
        self._vector = [
            float(const_term + const_corr), 
            float(U_term), 
            float(pH_term + pH_corr)
        ]

    def get_counter_correction(self, counter, alpha):
        """Correct the constant and pH contributions to the reaction free energy
           based on the counter electrode of choice and the temperature
           (alpha=k_B*T*ln(10))
        """
        n_e = self.species['e-']
        gibbs_corr = 0.0
        pH_corr = 0.0
        if counter in ['RHE', 'Pt']:
            pH_corr += n_e * alpha
            if counter == 'Pt' and n_e < 0:
                gibbs_corr +=  n_e * 0.5 * PREDEF_ENERGIES['H2O']
        if counter == 'AgCl':
            gibbs_corr -= n_e * U_STD_AGCL

        return gibbs_corr, pH_corr

    def equation(self):
        """Print the chemical reaction."""
        reactants = []
        products = []
        for s, n in self.species.items():
            if n == 0:
                continue
            if abs(n) == 1:
                substr = s
            else:
                substr = f'{abs(n)}{s}'
            if n > 0:
                products.append(substr)
            else:
                reactants.append(substr)
        return "  ➜  ".join([" + ".join(reactants), " + ".join(products)])

    def get_main_products(self):
        """Obtain the reaction products excluded protons, water and electrons."""
        return [spec for spec, coef in self.species.items() 
                if coef > 0 and spec not in ['H+', 'H2O', 'e-']]

    def get_free_energy(self, U, pH):
        """Evaluate the reaction free energy at a given applied potential U and pH"""
        return self._vector[0] + self._vector[1]*U + self._vector[2]*pH


class Pourbaix:
    """Pourbaix class for acqueous stability evaluations.

    Allows to determine the most stable phase in a given set
    of pH and potential conditions and to evaluate a complete diagram.

    Initialization
    --------------

    material_name: str
        The formula of the target material. It is preferrable
        to provide the reduced formula (e.g. RuO2 instad of Ru2O4).

    refs_dct: dict
        A dictionary containing the formula of the target material
        and its competing phases (solid and/or ionic) as keys,
        and their (formation) energies as values.

    T: float
        Temperature in Kelvin. Default: 298.15 K.

    conc: float
        Concentration of the ionic species. Default: 1e-6 mol/l.

    counter: str
        The counter electrode. Default: SHE.
        available options: SHE, RHE, AgCl, Pt.


    Relevant methods
    ----------------

    get_pourbaix_energy(U, pH)
        obtain the energy of the target material
        relative to the most stable phase at a given potential U and pH.
        If negative, the target material can be regarded as stable.
pbx.plot(cap=1.0, include_text=False, figsize=[12, 8], include_h2o=False, labeltype='phases')
    plot(**kwargs)
        plot a complete Pourbaix diagram in a given pH and potential window.


    Relevant attributes
    -------------------

    material: Species
        the target material as a Species object

    phases: list[RedOx]
        the available decomposition pathways of the target material
        into its competing phases as a list of RedOx objects

    """
    def __init__(self,
            material_name:str, 
            refs_dct:dict,
            T:float=298.15,
            conc:float=1.0e-6,
            counter:str='SHE'
        ):

        refs = initialize_refs(refs_dct)

        try:
            self.material = refs.pop(material_name)
        except KeyError:
            material_name = str(Formula(material_name).reduce()[0])
            self.material = refs.pop(material_name)

        self.counter = counter
        self.phases, phase_matrix = get_phases(
            self.material, refs, T, conc, counter
        )
        self._const = phase_matrix[:, 0]
        self._var = phase_matrix[:, 1:]

    def _decompose(self, U, pH):
        """Evaluate the reaction energy for decomposing
           the target material into each of the available products
           at a given pH and applied potential.
        """
        return self._const + np.dot(self._var, [U, pH])

    def _get_pourbaix_energy(self, U, pH):
        """Evaluate the Pourbaix energy"""
        energies = self._decompose(U, pH)
        i_min = np.argmin(energies)
        return -energies[i_min], i_min

    def get_pourbaix_energy(self, U, pH, verbose=True):
        """Evaluate the Pourbaix energy and print info
           about the most stable phase, decomposition pathway
           and corresponding energy.
        
        The Pourbaix energy represents the energy of the target material
        relative to the most stable competing phase. If negative,
        the target material can be considered as stable.
        """
        energy, index = self._get_pourbaix_energy(U, pH)
        phase = self.phases[index]
        if verbose:
            print(f'Stable phase: \n{phase.equation()}'
                  f'\nEnergy: {energy} eV')
        return energy, phase

    def get_diagrams(self, U, pH):
        """Actual evaluation of the complete diagram
        
        Returns
        -------

        pour: 
            The stability domains of the diagram on the pH vs. U grid.
            domains are represented by indexes (as integers)
            that map to Pourbaix.phases

        meta:
            The Pourbaix energy on the pH vs. U grid. 

        text:
            The coordinates and phases information for
            text placement on the diagram.

        """

        pour = np.zeros((len(U), len(pH)))
        meta = pour.copy()

        for i, u in enumerate(U):
            for j, p in enumerate(pH):
                meta[i, j], pour[i, j] = self._get_pourbaix_energy(u, p)

        # Identifying the region where the target material
        # is stable and updating the diagram accordingly
        where_stable = (meta <= 0)
        pour[where_stable] = -1

        text = []
        for phase_id in np.unique(pour):
            if phase_id == -1:
                where = where_stable
                txt = [self.material.name]
            else:
                where = (pour == phase_id)
                phase = self.phases[int(phase_id)]
                txt = phase.get_main_products()
            x = np.dot(where.sum(1), U) / where.sum()
            y = np.dot(where.sum(0), pH) / where.sum()
            text.append((x, y, txt))

        return pour, meta, text

    def _draw_diagram_axes(
            self,
            Urange, pHrange,
            npoints, cap,
            figsize, normalize,
            include_text, 
            include_h2o,
            labeltype, cmap):
        """Backend for drawing Pourbaix diagrams"""

        pH = np.linspace(*pHrange, num=npoints)
        U = np.linspace(*Urange, num=npoints)

        pour, meta, text = self.get_diagrams(U, pH)

        if normalize:
            meta /= self.material.natoms

        ax = plt.figure(figsize=figsize).add_subplot(111)
        extent = [*pHrange, *Urange]

        plt.subplots_adjust(
            left=0.1, right=0.97,
            top=0.97, bottom=0.14
        )

        if isinstance(cap, list):
            vmin = cap[0]
            vmax = cap[1]
        else:
            vmin = -cap
            vmax = cap

        colorplot = ax.imshow(
            meta, cmap=cmap,
            extent=extent,
            vmin=vmin, vmax=vmax,
            origin='lower', aspect='auto',
            interpolation='gaussian'
        )
            
        cbar = plt.gcf().colorbar(
               colorplot,
               ax=ax,
               pad=0.02
        )

        edges = edge_detection(pour)
        for _, indexes in edges.items():
            ax.plot(
                pH[indexes[1]],
                U[indexes[0]],
                ls='-',
                marker=None,
                zorder=1,
                color='k'
            )

        if include_text:
            plt.subplots_adjust(right=0.75)
            add_text(ax, text, offset=0.05)

        if labeltype == 'numbers':
            add_numbers(ax, text)
        elif labeltype == 'phases':
            add_labels(ax, text)
        elif labeltype == None:
            pass
        else: 
            raise ValueError("The provided label type doesn't exist")
            
        if include_h2o:
            add_redox_lines(ax, pH, self.counter, 'w')

        ax.set_xlim(*pHrange)
        ax.set_ylim(*Urange)
        ax.set_xlabel('pH', fontsize=18)
        ax.set_ylabel(r'$\it{U}$' + f' vs. {self.counter} (V)', fontsize=18)
        ax.set_xticks(np.arange(pHrange[0], pHrange[1] + 1, 2))
        ax.set_yticks(np.arange(Urange[0], Urange[1] + 1, 1))
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        ticks = np.linspace(vmin, vmax, num=9)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.set_ylabel(r'$\Delta G_{pbx}$ (eV/atom)', fontsize=18)

        return ax

    def plot(self,
             Urange=[-2, 2],
             pHrange=[0, 14],
             npoints=300,
             cap=1.0,
             figsize=[12, 6],
             normalize=True,
             include_text=True,
             include_h2o=True,
             labeltype='numbers',
             cmap="RdYlGn_r",
             savefig=None,
             show=True):
        """Plot a complete Pourbaix diagram.

        Keyword arguments
        -----------------

        Urange: list
            The potential range onto which to draw the diagram.

        pHrange: list
            The pH range onto which to draw the diagram.

        npoints: int
            The resolution of the diagram. Higher values
            mean higher resolution and thus higher compute times.

        cap: float/list
            If float, the limit (in both the positive and negative direction)
            of the Pourbaix energy colormap. 
            If list, the first and second value determine the colormap limits.

        figsize: list
            The horizontal and vertical size of the graph.

        normalize: bool
            Normalize energies by the number of
            atoms in the target material unit formula.

        include_text: bool
            Report to the right of the diagram the main products
            associated with the stability domains.

        include_h20: bool
            Include in the diagram the stability domain of water.

        labeltype: str/None
            The labeling style of the diagram domains. Options:
                'numbers': just add numbers associated with the different phases,
                           the latter shown on the right if include_text=True.
                'phases':  Write the main products directly on the diagram.
                           These labels can be dragged around if the placement
                           is unsatisfactory. Redundant if include_text=True.
                 None:     Don't draw any labels.

        savefig: str/None
            If passed as a string, the figure will be saved with that name.

        show: bool
            Spawn a window showing the diagram.

        """
        ax = self._draw_diagram_axes(
             Urange, pHrange,
             npoints, cap,
             figsize, normalize,
             include_text,
             include_h2o,
             labeltype, cmap)

        if savefig:
            plt.savefig(savefig)
        if show:
            plt.show()

