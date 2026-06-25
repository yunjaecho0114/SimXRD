"""High-level powder X-ray diffraction simulation entry points.

Bin Cao, PhD of HKUST(Guangzhou), https://bin-cao.github.io
URL : https://github.com/Bin-Cao/PyWPEM
"""

import spglib
from . import print_package_info
from .utils.funs import *
from .utils.MatgenKit import *
from .utils.WPEMsim import *


def _spacegroup_from_atoms(atoms, symprec=1e-3):
    """Return the space-group number and lattice-centering symbol.

    Args:
        atoms: ASE ``Atoms`` object with periodic cell, scaled positions, and
            atomic numbers.
        symprec (float): Position tolerance passed to spglib.

    Returns:
        tuple[int, str]: International space-group number and the first
        character of the international symbol, used here as the centering
        symbol for extinction rules.

    Raises:
        ValueError: If spglib cannot determine a space group or returns an
        empty international symbol.
    """
    cell = (atoms.get_cell(), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if dataset is None:
        raise ValueError("spglib could not determine the space group for this structure")

    number = dataset.number if hasattr(dataset, "number") else dataset["number"]
    international = dataset.international if hasattr(dataset, "international") else dataset["international"]
    symbol = str(international).strip()
    if not symbol:
        raise ValueError("spglib returned an empty space-group symbol")
    return int(number), symbol[0]


def parser(database,entry_id,grainsize=20,prefect_orientation=[0.1,0.1],thermo_vibration=0.1,
          zero_shift=0.1,dis_detector2sample=500,half_height_slit_detector = 5,half_height_sample=2.5,
          deformation=False,sim_model=None,xrd='reciprocal', background_order = 6,
          background_ratio=0.05, mixture_noise_ratio=0.02, lattice_extinction_ratio=0.01,lattice_torsion_ratio=0.01,
          verbose=False,):
    """Simulate a powder X-ray diffraction pattern for one ASE database entry.

    The default branch uses pymatgen for peak positions and the shared peak
    broadening model. The ``"WPEM"`` branch uses the in-repository WPEM
    diffraction condition, extinction, multiplicity, and peak-intensity logic.
    For WPEM deformation, symmetry information is read from the undeformed
    conventional cell while peak positions use the deformed lattice constants.

    Args:
        database: Open ASE database connection.
        entry_id (int): ASE database row id.
        grainsize (float): Grain size of the specimen in Angstroms.
        prefect_orientation (list[float]): Preferred-orientation broadening control.
            The name is kept for backward compatibility with existing callers.
        thermo_vibration (float): Average thermal displacement of atoms in Angstroms.
        zero_shift (float): Angular zero-shift in degrees.
        dis_detector2sample (float): Detector-to-sample distance in millimeters.
        half_height_slit_detector (float): Detector slit half-height in millimeters.
        half_height_sample (float): Sample half-height in millimeters.
        deformation (bool): Whether to apply random lattice deformation.
        sim_model (str | None): ``"WPEM"`` for the WPEM model, otherwise the
            pymatgen-based conventional simulation is used.
        xrd (str): ``"reciprocal"`` returns 2-theta; ``"real"`` returns
            lattice-plane distance.
        background_order (int): Random polynomial background degree.
        background_ratio (float): Background intensity ratio relative to peaks.
        mixture_noise_ratio (float): Uniform mixed-noise ratio relative to peaks.
        lattice_extinction_ratio (float): Random stretching/compression ratio.
        lattice_torsion_ratio (float): Random shear ratio.
        verbose (bool): Print package/run metadata before simulation.

    Returns:
        tuple[np.ndarray, np.ndarray | list]: Simulated x-axis values and
        normalized intensities.

    Example:
        >>> from ase.db import connect
        >>> from Pysimxrd import generator
        >>> database = connect("demo.db")
        >>> x, y = generator.parser(database, 1)
    """
    if verbose:
        print_package_info()

    atoms = database.get_atoms(id=entry_id)
    dis_detector2sample = dis_detector2sample / 10
    if sim_model is None:
        x,y = matgen_pxrdsim(
            atoms,grainsize,prefect_orientation,thermo_vibration,deformation,
            dis_detector2sample,half_height_slit_detector,half_height_sample,background_order,
            background_ratio, mixture_noise_ratio, lattice_extinction_ratio,lattice_torsion_ratio,xrd
            )
    elif sim_model == 'WPEM':
        # for general db files contains conventional lattice cells
        # G_latt_consts = atoms.cell.cellpar()
        # in case the input is primitive unit cell
        G_latt_consts,_, c_atom = prim2conv(atoms,False,lattice_extinction_ratio,lattice_torsion_ratio,)
        if deformation:
            G_latt_consts,_, _ = prim2conv(atoms,True,lattice_extinction_ratio,lattice_torsion_ratio,)
        N_symbols = c_atom.get_chemical_symbols()
        positions = c_atom.get_scaled_positions()
        G_latt_vol = c_atom.get_volume()
        G_spacegroup, spacegroup_symbol = _spacegroup_from_atoms(c_atom)
        crystal_system = space_group_to_crystal_system(G_spacegroup)
        AtomCoordinates = c_atom_covert2WPEMformat(positions, N_symbols)
        x,y = pxrdsim(
                        G_latt_vol, spacegroup_symbol, AtomCoordinates, G_latt_consts, crystal_system,
                        grainsize, prefect_orientation, thermo_vibration, zero_shift,
                        dis_detector2sample,half_height_slit_detector,half_height_sample,
                        background_order,background_ratio, mixture_noise_ratio,  xrd
                        )
    else:
        raise ValueError("sim_model must be None or 'WPEM'")
    return x,y
