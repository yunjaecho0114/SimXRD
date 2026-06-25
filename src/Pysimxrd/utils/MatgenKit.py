"""pymatgen-backed helpers for conventional powder XRD simulation.

Bin Cao, PhD of HKUST(Guangzhou), https://bin-cao.github.io
URL : https://github.com/Bin-Cao/PyWPEM
"""

from pymatgen.core.structure import Structure, Lattice
from pymatgen.analysis.diffraction import xrd
import numpy as np
from .funs import *

def _atom2str(atoms, deformation,lattice_extinction_ratio,lattice_torsion_ratio):
    """Convert an ASE ``Atoms`` object to a pymatgen ``Structure``.

    The primitive input is first standardized to a conventional cell by
    :func:`prim2conv`; optional random deformation is applied inside that helper.

    Args:
        atoms: ASE atomic structure.
        deformation (bool): Whether to apply random lattice deformation.
        lattice_extinction_ratio (float): Stretching/compression ratio.
        lattice_torsion_ratio (float): Shear ratio.

    Returns:
        Structure: pymatgen structure with lattice, species, and scaled positions.
    """
    _, _, c_atom = prim2conv(atoms, deformation,lattice_extinction_ratio,lattice_torsion_ratio)

    cell = c_atom.get_cell()
    symbols = c_atom.get_chemical_symbols()
    positions = c_atom.get_scaled_positions()
    lattice = Lattice(cell)

    return Structure(lattice, symbols, positions)

def get_diff(atom,deformation,lattice_extinction_ratio,lattice_torsion_ratio):
    """Calculate raw peak positions and intensities with pymatgen.

    Args:
        atom: ASE ``Atoms`` object from the database.
        deformation (bool): Whether to deform the conventional lattice.
        lattice_extinction_ratio (float): Stretching/compression ratio.
        lattice_torsion_ratio (float): Shear ratio.

    Returns:
        tuple[np.ndarray, np.ndarray]: Peak positions and intensities in the
        default 10-80 degree 2-theta range.
    """
    calculator = xrd.XRDCalculator()
    struc = _atom2str(atom,deformation,lattice_extinction_ratio,lattice_torsion_ratio)

    pattern = calculator.get_pattern(struc, two_theta_range=(10, 80))
    # The package returns peak centers at its internal precision; downstream
    # broadening maps those centers onto the configured simulation grid.
    return pattern.x, pattern.y


def matgen_pxrdsim(atom,GrainSize,orientation,thermo_vib,deformation,L,H,S,
                   background_order,background_ratio, mixture_noise_ratio,
                   lattice_extinction_ratio,lattice_torsion_ratio,xrd):
    """Simulate a powder XRD profile using pymatgen peak positions.

    Args:
        atom: ASE ``Atoms`` object.
        GrainSize (float): Grain size in Angstroms.
        orientation (list[float]): Preferred-orientation coefficient bounds.
        thermo_vib (float): Thermal vibration amplitude.
        deformation (bool): Whether to deform the conventional lattice.
        L (float): Detector-to-sample distance in centimeters.
        H (float): Detector slit half-height.
        S (float): Sample half-height.
        background_order (int): Random polynomial background degree.
        background_ratio (float): Background intensity ratio.
        mixture_noise_ratio (float): Uniform mixed-noise ratio.
        lattice_extinction_ratio (float): Stretching/compression ratio.
        lattice_torsion_ratio (float): Shear ratio.
        xrd (str): ``"reciprocal"`` for 2-theta or ``"real"`` for d-spacing.

    Returns:
        tuple[np.ndarray, list | np.ndarray]: Simulated x-axis and normalized
        intensity profile.
    """
    wavelength = 1.54184
    two_theta_range = (10, 80.0,0.02)

    mu_array,_Ints = get_diff(atom,deformation,lattice_extinction_ratio,lattice_torsion_ratio)

    Γ = 0.888*wavelength/(GrainSize*np.cos(np.radians(np.array(mu_array)/2)))
    gamma_list = Γ / 2 + 1e-10
    sigma2_list = Γ**2 / (8*np.sqrt(2)) + 1e-10

    Ints = []
    for k in range(len(_Ints)):

        Ori_coe = np.clip(np.random.normal(loc=1, scale=0.2), 1-orientation[0], 1+orientation[0])
        M = 8/3 * np.pi**2*thermo_vib**2 * (np.sin(np.radians(mu_array[k]/2)) / wavelength)**2
        Deb_coe = np.exp(-2*M)
        Ints.append(_Ints[k] * Ori_coe * Deb_coe)

    x_sim = np.arange(two_theta_range[0],two_theta_range[1],two_theta_range[2])
    y_sim = 0
    for num in range(len(Ints)):
        #_ = draw_peak_density(x_sim, Ints[num], mu_array[num], gamma_list[num], sigma2_list[num])

        _ = combined_peak(x_sim, Ints[num], mu_array[num], gamma_list[num], sigma2_list[num], L,H,S,two_theta_range[2])
        y_sim += _
    # normalize the profile
    nor_y = y_sim / theta_intensity_area(x_sim,y_sim)

    random_polynomial = generate_random_polynomial(degree=background_order)
    _bac = random_polynomial(x_sim)
    _bac -= _bac.min()
    _bacI = _bac / _bac.max() * nor_y.max() *background_ratio
    mixture = np.random.uniform(0, nor_y.max() * mixture_noise_ratio, size=len(x_sim))
    nor_y +=  np.flip(_bacI) + mixture
    nor_y = scale_list(nor_y)
    if xrd=='real':
        x_sim = wavelength/(2*np.sin(x_sim/2  * np.pi/ 180 ))
    return x_sim, nor_y
