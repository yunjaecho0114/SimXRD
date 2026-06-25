"""WPEM-based powder X-ray diffraction simulation utilities.

Bin Cao, PhD of HKUST(Guangzhou), https://bin-cao.github.io
URL : https://github.com/Bin-Cao/PyWPEM
"""

from .funs import *
from sympy import *
import copy
import numpy as np


def c_atom_covert2WPEMformat(c_atom_sites,  c_atom_N_symbols):
    """Convert conventional-cell sites to WPEM atom-coordinate rows.

    Args:
        c_atom_sites (np.ndarray): Fractional positions from an ASE ``Atoms`` object.
        c_atom_N_symbols (Sequence[str]): Chemical symbols aligned to positions.

    Returns:
        list[list]: Rows in ``[symbol, frac_x, frac_y, frac_z]`` format.
    """
    sites_list = c_atom_sites.tolist()
    for i in range(len(sites_list)):
        sites_list[i].insert(0, c_atom_N_symbols[i])
    return sites_list

def covert2WPEMformat(sites, kinds, N_symbols):
    """Convert asymmetric-unit sites and kinds to WPEM atom-coordinate rows.

    Args:
        sites (np.ndarray): Fractional coordinates of asymmetric-unit sites.
        kinds (Sequence[int]): Integer mapping from each site to a symbol index.
        N_symbols (Sequence[str]): Symbol table referenced by ``kinds``.

    Returns:
        list[list]: Rows in ``[symbol, frac_x, frac_y, frac_z]`` format.
    """
    sites_list = sites.tolist()
    for i in range(len(sites_list)):
        sites_list[i].insert(0, N_symbols[kinds[i]])
    return sites_list

def space_group_to_crystal_system(space_group):
    """Map an international space-group number to the internal crystal-system code.

    Returns:
        int | str: Codes are ``1`` cubic, ``2`` hexagonal, ``3`` tetragonal,
        ``4`` orthorhombic, ``5`` trigonal/rhombohedral, ``6`` monoclinic,
        and ``7`` triclinic. Invalid numbers return the legacy error string.
    """
    if space_group < 1 or space_group > 230:
        return "Invalid space group number"
    elif space_group <= 2:
        return 7
    elif space_group <= 15:
        return 6
    elif space_group <= 74:
        return 4
    elif space_group <= 142:
        return 3
    elif space_group <= 167:
        return 5
    elif space_group <= 194:
        return 2
    else:
        return 1


def pxrdsim(Volume,Point_group,AtomCoordinates,latt,crystal_system,GrainSize,orientation,thermo_vib,
            zero_shift,L,H,S,background_order,background_ratio, mixture_noise_ratio, xrd):
    """Simulate a powder XRD profile with the WPEM peak-intensity model.

    Args:
        Volume (float): Conventional-cell volume.
        Point_group (str): Lattice centering symbol used for extinction rules.
        AtomCoordinates (Sequence): WPEM-format atom coordinates.
        latt (Sequence[float]): Lattice constants ``a, b, c, alpha, beta, gamma``.
        crystal_system (int): Internal crystal-system code.
        GrainSize (float): Grain size in Angstroms.
        orientation (list[float]): Preferred-orientation coefficient bounds.
        thermo_vib (float): Thermal vibration amplitude.
        zero_shift (float): 2-theta zero shift in degrees.
        L (float): Detector-to-sample distance in centimeters.
        H (float): Detector slit half-height.
        S (float): Sample half-height.
        background_order (int): Random polynomial background degree.
        background_ratio (float): Background intensity ratio.
        mixture_noise_ratio (float): Uniform mixed-noise ratio.
        xrd (str): ``"reciprocal"`` for 2-theta or ``"real"`` for d-spacing.

    Returns:
        tuple[np.ndarray, np.ndarray]: Simulated x-axis and normalized intensity.
    """
    wavelength = 1.54184
    two_theta_range = (8, 82.02,0.02) # 10-80, 0.02
    grid, d_list = Diffraction_index(crystal_system,latt,wavelength,two_theta_range)
    res_HKL, _, d_res_HKL, _ = cal_extinction(Point_group, grid,d_list,crystal_system,AtomCoordinates,wavelength,)
    mu_array = 2 * np.arcsin(wavelength /2/np.array(d_res_HKL)) * 180 / np.pi

    FHKL_square = []
    Mult = mult_dic(res_HKL,crystal_system,latt,AtomCoordinates)
    for angle in range(len(res_HKL)):
        FHKL_square_left = 0
        FHKL_square_right = 0
        # _Atom_coordinate all atoms in lattice cell
        for atom in range(len(AtomCoordinates)):
            fi = cal_atoms(AtomCoordinates[atom][0],mu_array[angle], wavelength)
            FHKL_square_left += fi * np.cos(2 * np.pi * (AtomCoordinates[atom][1] * res_HKL[angle][0] +
                                                AtomCoordinates[atom][2] * res_HKL[angle][1] + AtomCoordinates[atom][3] * res_HKL[angle][2]))
            FHKL_square_right += fi * np.sin(2 * np.pi * (AtomCoordinates[atom][1] * res_HKL[angle][0] +
                                                AtomCoordinates[atom][2] * res_HKL[angle][1] + AtomCoordinates[atom][3] * res_HKL[angle][2]))
        FHKL_square.append(FHKL_square_left ** 2 + FHKL_square_right ** 2)



    _Ints = [] # cal peak intensity
    for angle in range(len(FHKL_square)):
        _Ints.append(float(FHKL_square[angle] * Mult[angle] / Volume ** 2
                    * (1 + np.cos(mu_array[angle] * np.pi/180) ** 2) / (np.sin(mu_array[angle] / 2 * np.pi/180) **2 * np.cos(mu_array[angle] / 2 * np.pi/180))))

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
    step = two_theta_range[2]
    y_sim = 0
    for num in range(len(Ints)):
        _ = combined_peak(x_sim, Ints[num], mu_array[num], gamma_list[num], sigma2_list[num],L,H,S, step)
        y_sim += _
    # normalize the profile
    nor_y = y_sim / theta_intensity_area(x_sim,y_sim)

    x_sim += zero_shift
    random_polynomial = generate_random_polynomial(degree=background_order)
    _bac = random_polynomial(x_sim)
    _bac -= _bac.min()
    _bacI = _bac / _bac.max() * nor_y.max() * background_ratio
    mixture = np.random.uniform(0, nor_y.max() * mixture_noise_ratio, size=len(x_sim))
    nor_y +=  np.flip(_bacI) + mixture
    nor_y = scale_list(nor_y)


    index_10 = np.abs(x_sim - 10).argmin()
    index_80 = np.abs(x_sim - 80).argmin() + 1
    #x_sim = x_sim[index_10:index_80]
    x_sim = np.arange(10,80,two_theta_range[2])
    nor_y = nor_y[index_10:index_80]
    if xrd=='real':
        x_sim = wavelength/(2*np.sin(x_sim/2  * np.pi/ 180 ))
    return x_sim, adjust_length(nor_y,len(x_sim))



def adjust_length(nor_y, desired_length=1000):
    """Pad or trim an intensity array to the requested length.

    Args:
        nor_y (np.ndarray): Input intensity array.
        desired_length (int): Target array length.

    Returns:
        np.ndarray: Adjusted intensity array.
    """
    if len(nor_y) < desired_length:
        nor_y = np.pad(nor_y, (0, desired_length - len(nor_y)), 'constant',constant_values=[nor_y[-1]])
    elif len(nor_y) > desired_length:
        nor_y = nor_y[:desired_length]
    return nor_y


def Diffraction_index(system,latt,cal_wavelength,two_theta_range):
    """Calculate diffraction HKL indices that satisfy Bragg geometry.

    Uses ``S - S' = G*``, where ``G*`` is the reciprocal lattice vector.

    Args:
        system (int): Internal crystal-system code.
        latt (Sequence[float]): Lattice constants ``a, b, c, alpha, beta, gamma``.
        cal_wavelength (float): X-ray wavelength.
        two_theta_range (tuple[float, float, float]): Lower bound, upper bound,
            and step of the 2-theta search range.

    Returns:
        tuple[list[np.ndarray], list[float]]: Non-redundant HKL values and
        matching d-spacings after Bragg filtering.
    """

    grid = grid_atom()
    index_of_origin = np.where((grid[:, 0] == 0) & (grid[:, 1] == 0) & (grid[:, 2] == 0))[0][0]
    grid[[0, index_of_origin]] = grid[[index_of_origin, 0]]
    d_f = BraggLawDerivation().d_spcing(system)
    sym_h, sym_k, sym_l, sym_a, sym_b, sym_c, angle1, angle2, angle3 = \
            symbols('sym_h sym_k sym_l sym_a sym_b sym_c angle1 angle2 angle3')
    d_list = [1e-10] # HKL=000

    for i in range(len(grid)-1):
        peak = grid[i+1]
        d_list.append(
            float(d_f.subs({sym_h: peak[0], sym_k: peak[1], sym_l: peak[2], sym_a: latt[0], sym_b: latt[1],
                            sym_c: latt[2], angle1: latt[3]*np.pi/180, angle2: latt[4]*np.pi/180, angle3:latt[5]*np.pi/180}))
                            )

    # Satisfied the Bragg Law
    # 2theta = 2 * arcsin (lamda / 2 / d)
    bragg_d = cal_wavelength /2/np.array(d_list)
    index0 = np.where(bragg_d > 1)
    # avoid null values
    _d_list = pd.DataFrame(d_list).drop(index0[0])
    _grid = pd.DataFrame(grid).drop(index0[0])

    # recover the index of DataFrame
    for i in [_d_list, _grid]:
        i.index = range(i.shape[0])

    two_theta = 2 * np.arcsin(cal_wavelength /2/np.array(_d_list.iloc[:,0])) * (180 / np.pi)
    index = np.where((two_theta <= two_theta_range[0]) | (two_theta >= two_theta_range[1]))

    d_list = _d_list.drop(index[0])
    grid = _grid.drop(index[0])

    # return all HKL which are satisfied Bragg law
    res_HKL, res_d = de_redundant(grid, d_list, system)
    return res_HKL, res_d

class BraggLawDerivation:
    """Interplanar-spacing and lattice-constant update formulas.

    The interplanar-spacing equations cover the seven crystal systems.
    Reference: https://github.com/Bin-Cao/MPhil_SHU/tree/main/thesis_BInCAO
    table (1.1).
    """

    def d_spcing(self, crystal_system):
        """Return the symbolic d-spacing formula for a crystal-system code.

        Angles in substitutions must be provided in radians.
        """
        sym_h, sym_k, sym_l, sym_a, sym_b, sym_c, angle1, angle2, angle3 = \
            symbols('sym_h sym_k sym_l sym_a sym_b sym_c angle1 angle2 angle3')
        if crystal_system == 1:  # Cubic
            d_hkl = sym_a / sqrt(sym_h ** 2 + sym_k ** 2 + sym_l ** 2)
        elif crystal_system == 2:  # Hexagonal
            d_hkl = sqrt(3) / 2 / sqrt(
                ((sym_h ** 2 + sym_k ** 2 + sym_h * sym_k) / sym_a ** 2) + 3 * sym_l ** 2 / 4 / sym_c ** 2)
        elif crystal_system == 3:  # Tetragonal
            d_hkl = 1 / sqrt((sym_h ** 2 + sym_k ** 2) / sym_a ** 2 + sym_l ** 2 / sym_c ** 2)
        elif crystal_system == 4:  # Orthorhombic
            d_hkl = 1 / sqrt((sym_h / sym_a) ** 2 + (sym_k / sym_b) ** 2 + (sym_l / sym_c) ** 2)
        elif crystal_system == 5:  # Rhombohedral
            d_hkl = sym_a * sqrt((1 - 3 * cos(angle1) ** 2 + 2 * cos(angle1) ** 3) /
                                 ((sym_h ** 2 + sym_k ** 2 + sym_l ** 2) * sin(angle1) ** 2
                                  + 2 * (sym_h * sym_k + sym_k * sym_l + sym_h * sym_l) * cos(angle1) ** 2 - cos(
                                             angle1)))
        elif crystal_system == 6:  # Monoclinic
            d_hkl = sin(angle2) / sqrt(
                sym_h ** 2 / sym_a ** 2 + sym_k ** 2 * sin(angle2) ** 2 / sym_b ** 2 + sym_l ** 2 / sym_c ** 2 -
                2 * sym_h * sym_l * cos(angle2) / (sym_a * sym_c))
        elif crystal_system == 7:  # Triclinic
            __m = sqrt(
                sin(angle3) ** 2 - cos(angle1) ** 2 + cos(angle2) ** 2 + 2 * cos(angle1) * cos(angle2) * cos(angle3))
            __p = cos(angle1) - cos(angle2) * cos(angle3)
            __n = cos(angle2) - cos(angle1) * cos(angle3)
            __q = cos(angle3) - cos(angle1) * cos(angle2)
            d_hkl = __m / sqrt((sym_h / sym_a) ** 2 * sin(angle1) ** 2 + (sym_k / sym_b) ** 2 * sin(angle2) ** 2 +
                               (sym_l / sym_c) ** 2 * sin(angle3) ** 2 - 2 * sym_h * sym_k * __q / (sym_a * sym_b)
                               - 2 * sym_h * sym_l * __n / (sym_a * sym_c) - 2 * sym_k * sym_l * __p / (sym_b * sym_c))
        else:
            d_hkl = -1
        return d_hkl

    def get_angle_sort(self, mui_cal_em_list, mui_abc_list):
        """Sort WPEM peak positions to match the Bragg-law order.

        Args:
            mui_cal_em_list (Sequence[float]): Peak positions calculated by the
                empirical/WPEM branch.
            mui_abc_list (Sequence[float]): Reference Bragg-law peak positions.

        Returns:
            list[float]: ``mui_cal_em_list`` reordered by ascending
            ``mui_abc_list``.
        """
        __mui_abc_copy = copy.deepcopy(  mui_abc_list)
        __mui_abc_copy.sort()
        __mui_cal_em_matching = []
        __j_index = []
        for i in range(len(  mui_cal_em_list)):
            for j in range(len(  mui_cal_em_list)):
                if   mui_abc_list[i] == __mui_abc_copy[j] and j not in __j_index:
                    __mui_cal_em_matching.append(  mui_cal_em_list[j])
                    __j_index.append(j)
                    break
        return __mui_cal_em_matching

    def get_mui_sort(self, mui_1, mui_2, mui_new_list):
        """Sort Bragg-law peak positions to match a reference peak order.

        Args:
            mui_1 (Sequence[float]): Target reference order.
            mui_2 (Sequence[float]): Current order used for matching.
            mui_new_list (Sequence[float]): Values to reorder.

        Returns:
            list[float]: ``mui_new_list`` aligned to ``mui_1``.
        """
        __mui_out = []
        __j_index = []
        for i in range(len(  mui_1)):
            for j in range(len(  mui_1)):
                if   mui_1[i] ==   mui_2[j] and j not in __j_index:
                    __mui_out.append(  mui_new_list[j])
                    __j_index.append(j)
                    break
        return __mui_out

    def get_d_space(self, crystal_sys, h, k, l, a, b, c, la1, la2, la3):
        """Calculate d-spacings and symbolic derivative terms for HKL arrays.

        Args:
            crystal_sys (int): Crystal-system code.
            h, k, l (Sequence[int]): Miller-index components.
            a, b, c (float): Lattice lengths.
            la1, la2, la3 (float): Lattice angles in degrees.

        Returns:
            tuple[list[float], list[list[float]], int]: d-spacings, derivatives
            for each optimizable lattice parameter, and the number of optimized
            parameters for the selected crystal system.
        """
        d_f =  self.d_spcing(crystal_sys)
        sym_h, sym_k, sym_l, sym_a, sym_b, sym_c, angle1, angle2, angle3 = \
            symbols('sym_h sym_k sym_l sym_a sym_b sym_c angle1 angle2 angle3')

        d_f_derivative = [diff(d_f, sym_a, 1)]
        if   crystal_sys == 2 or  crystal_sys == 3:
            d_f_derivative.append(diff(d_f, sym_c, 1))
        elif   crystal_sys == 4:
            d_f_derivative.append(diff(d_f, sym_b, 1))
            d_f_derivative.append(diff(d_f, sym_c, 1))
        elif   crystal_sys == 5:
            d_f_derivative.append(diff(d_f, angle1, 1))
        elif   crystal_sys == 6:
            d_f_derivative.append(diff(d_f, sym_b, 1))
            d_f_derivative.append(diff(d_f, sym_c, 1))
            d_f_derivative.append(diff(d_f, angle2, 1))
        elif   crystal_sys == 7:
            d_f_derivative.append(diff(d_f, sym_b, 1))
            d_f_derivative.append(diff(d_f, sym_c, 1))
            d_f_derivative.append(diff(d_f, angle1, 1))
            d_f_derivative.append(diff(d_f, angle2, 1))
            d_f_derivative.append(diff(d_f, angle3, 1))

        peak_n = len(h)

        # defined a list for storing the interplanar spacing of each combination of diffraction index [HKL --> d]
        d_list = []
        # defined a list for storing the derivations of interplanar spacing of each diffraction index [HKL --> d(spaceing)/d(variables, a,b,c,...]
        d_der_list = []
        for i in range(peak_n):
            d_list.append(
                float(d_f.subs({sym_h: h[i], sym_k: k[i], sym_l: l[i], sym_a: a, sym_b: b,
                                sym_c: c, angle1: la1*np.pi/180, angle2: la2*np.pi/180, angle3: la3*np.pi/180})))
            d_der_term = []
            for j in range(len(d_f_derivative)):
                d_der_term.append(
                    float(d_f_derivative[j].subs({sym_h: h[i], sym_k: k[i], sym_l: l[i], sym_a: a,
                                                  sym_b: b, sym_c: c, angle1: la1*np.pi/180, angle2: la2*np.pi/180,
                                                  angle3: la3*np.pi/180}))
                                )
            d_der_list.append(d_der_term)
        return d_list, d_der_list, len(d_f_derivative)

    def get_d_space_multitask(self, crystal_sys_set, hkl_dat, Lattice_constants):
        """Calculate d-spacings for multiple crystal/lattice tasks.

        Args:
            crystal_sys_set (Sequence[int]): Crystal-system code per task.
            hkl_dat (Sequence): Per-task HKL component arrays.
            Lattice_constants (Sequence): Per-task lattice constants in
                ``a, b, c, alpha, beta, gamma`` order.

        Returns:
            list[list[float]]: d-spacing arrays grouped by task.
        """
        d_list_set = []
        for task in range(len(Lattice_constants)):
            d_f =  self.d_spcing(crystal_sys_set[task])
            sym_h, sym_k, sym_l, sym_a, sym_b, sym_c, angle1, angle2, angle3 = \
                symbols('sym_h sym_k sym_l sym_a sym_b sym_c angle1 angle2 angle3')

            peak_n = len(hkl_dat[task][1])
            d_list = []
            for i in range(peak_n):
                d_list.append(float(d_f.subs({sym_h: hkl_dat[task][0][i],
                                              sym_k: hkl_dat[task][1][i],
                                              sym_l: hkl_dat[task][2][i],
                                              sym_a: Lattice_constants[task][0],
                                              sym_b: Lattice_constants[task][1],
                                              sym_c: Lattice_constants[task][2],
                                              angle1: Lattice_constants[task][3] * np.pi/180,
                                              angle2: Lattice_constants[task][4] * np.pi/180,
                                              angle3: Lattice_constants[task][5] * np.pi/180})))
            d_list_set.append(d_list)  # task * peak

        return d_list_set

    def get_derivative_term(self, d_spac, d_derivative, n_para, wavelength):
        """Build gradient terms for lattice-constant optimization.

        Args:
            d_spac (Sequence[float]): d-spacings for selected reflections.
            d_derivative (Sequence[Sequence[float]]): Derivatives of d-spacing
                with respect to active lattice parameters.
            n_para (int): Number of active lattice parameters.
            wavelength (Sequence[float]): One or two X-ray wavelengths.

        Returns:
            tuple[list[list[float]], list[float]]: Parameter update terms and
            calculated 2-theta positions.
        """
        lmb = len(wavelength)
        peak_n = len(d_spac)
        mui_abc = []
        update_term = []
        for i in range(peak_n):
            for j in range(lmb):
                # f_term --> 2theta, the diffraction angles of Bargg Law
                f_term = 2 * (np.arcsin(wavelength[j] / 2 / d_spac[i]) * 180 / np.pi)
                s_term = 1 / np.sqrt(1 - wavelength[j] ** 2 / d_spac[i] ** 2 / 4)
                t_term =   wavelength[j] / d_spac[i] ** 2
                mui_abc.append(f_term)
                update_para = []
                for up_j in range(n_para):
                    update_para.append(s_term * t_term * d_derivative[i][up_j])
                update_term.append(update_para)
        return update_term, mui_abc

    def update_derivative(self, crystal_sys, total_peak_n, h, k, l, a, b, c, la1, la2, la3, wavelength, mui_cal_em_match):
        """Calculate the first derivative of peak-position error.

        Args:
            crystal_sys (int): Crystal-system code.
            total_peak_n (int): Number of matched peaks used for averaging.
            h, k, l (Sequence[int]): Miller-index components.
            a, b, c (float): Current lattice lengths.
            la1, la2, la3 (float): Current lattice angles in degrees.
            wavelength (Sequence[float]): One or two X-ray wavelengths.
            mui_cal_em_match (Sequence[float]): Observed or target peak
                positions aligned to the calculated HKL sequence.

        Returns:
            list[float]: Average first-derivative term for each active lattice
            parameter.
        """
        d_list, d_der_list, n_para = self.get_d_space(crystal_sys, h, k, l, a, b, c,la1, la2, la3)
        update_term, mui_fit =  self.get_derivative_term(d_list, d_der_list, n_para,wavelength)

        mui_error = []
        for i in range(len(mui_cal_em_match)):
            mui_error.append(mui_cal_em_match[i] - mui_fit[i])

        deri_sum = []
        for j in range(n_para):
            two_der_mui_fit = []
            for i in range(len(mui_error)):
                two_der_mui_fit.append(mui_error[i] * update_term[i][j])
            deri_sum.append(sum(two_der_mui_fit) / total_peak_n)
        return deri_sum

    def two_up_derivative(self, crystal_sys, h, k, l, a, b, c, la1, la2, la3, der_term, wavelength, mui_fit_abc):
        """Calculate a finite-difference derivative term for learning rate.

        Args:
            crystal_sys (int): Crystal-system code.
            h, k, l (Sequence[int]): Miller-index components.
            a, b, c (float): Perturbed lattice lengths.
            la1, la2, la3 (float): Perturbed lattice angles in degrees.
            der_term (int): Parameter selector, from ``1`` for ``a`` through
                ``6`` for ``gamma``.
            wavelength (Sequence[float]): One or two X-ray wavelengths.
            mui_fit_abc (Sequence[float]): Target peak positions.

        Returns:
            float: Averaged derivative contribution used to estimate the local
            learning-rate denominator.
        """
        d_f =  self.d_spcing(crystal_sys)
        sym_h, sym_k, sym_l, sym_a, sym_b, sym_c, angle1, angle2, angle3 = \
            symbols('sym_h sym_k sym_l sym_a sym_b sym_c angle1 angle2 angle3')

        if   der_term == 1:
            __d_f_derivative = diff(d_f, sym_a, 1)
        elif   der_term == 2:
            __d_f_derivative = diff(d_f, sym_b, 1)
        elif   der_term == 3:
            __d_f_derivative = diff(d_f, sym_c, 1)
        elif   der_term == 4:
            __d_f_derivative = diff(d_f, angle1, 1)
        elif   der_term == 5:
            __d_f_derivative = diff(d_f, angle2, 1)
        elif   der_term == 6:
            __d_f_derivative = diff(d_f, angle3, 1)
        else:
            print('please input the derivative variable')

        peak_n = len(h)
        d_list = []
        d_der_list = []
        for i in range(peak_n):
            d_list.append(
                float(d_f.subs({sym_h: h[i], sym_k:k[i], sym_l: l[i], sym_a: a, sym_b:  b,
                                sym_c:c, angle1:la1*np.pi/180, angle2:la2*np.pi/180, angle3:la3*np.pi/180})))

            d_der_list.append(float(__d_f_derivative.subs(
                {sym_h:h[i], sym_k:k[i], sym_l:l[i], sym_a:a, sym_b: b,
                 sym_c: c, angle1:la1*np.pi/180, angle2:la2*np.pi/180, angle3:la3*np.pi/180})))

        lmb = len(wavelength)
        peak_n = len(d_list)
        mui_abc = []
        update_term = []
        for i in range(peak_n):
            for j in range(lmb):
                f_term = 2 * (np.arcsin(wavelength[j] / 2 / d_list[i]) * 180 / np.pi)
                s_term = 1 / np.sqrt(1 - wavelength[j] ** 2 / d_list[i] ** 2 / 4)
                t_term =  wavelength[j] / d_list[i] ** 2
                mui_abc.append(f_term)
                update_term.append(s_term * t_term * d_der_list[i])

        mui_error = []
        for i in range(len( mui_fit_abc)):
            mui_error.append( mui_fit_abc[i] - mui_abc[i])

        two_der_mui_fit = []
        for i in range(len(mui_error)):
            two_der_mui_fit.append(mui_error[i] * update_term[i])
        return sum(two_der_mui_fit) / len(  mui_fit_abc)

    def get_new_mui(self, d_spac, wavelength):
        """Calculate diffraction peak positions from d-spacings.

        Args:
            d_spac (Sequence[float]): d-spacings.
            wavelength (Sequence[float]): One or two X-ray wavelengths.

        Returns:
            list[float]: 2-theta positions for every d-spacing/wavelength pair.
        """
        lmb = len( wavelength)
        peak_n = len( d_spac)
        mui_abc = []
        for i in range(peak_n):
            for j in range(lmb):
                f_term = 2 * (np.arcsin(  wavelength[j] / 2 /   d_spac[i]) * 180 / np.pi)
                mui_abc.append(f_term)
        return mui_abc

    def get_new_mui_multitask(self, d_spac_set, wavelength):
        """Calculate diffraction peak positions for multiple tasks.

        Args:
            d_spac_set (Sequence[Sequence[float]]): d-spacings grouped by task.
            wavelength (Sequence[float]): One or two X-ray wavelengths.

        Returns:
            list[list[float]]: 2-theta positions grouped by task.
        """
        lmb = len(wavelength)
        mui_abc_set = []
        for task in range(len(d_spac_set)):
            peak_n = len(d_spac_set[task])
            mui_abc = []
            for i in range(peak_n):
                for j in range(lmb):
                    f_term = 2 * (np.arcsin(wavelength[j] / 2 / d_spac_set[task][i]) * 180 / np.pi)
                    mui_abc.append(f_term)
            mui_abc_set.append(mui_abc)

        return mui_abc_set  # peak*task

    def OptmialLatticeConstant(self, crystal_sys, old_p1_list, p1_list, subset_number, low_bound, up_bound, lattice_h,
                               lattice_k,lattice_l, ini_a, ini_b, ini_c, ini_la1, ini_la2, ini_la3, wavelength,fixed =False, tao=0.05):
        """Estimate lattice constants by matching observed and calculated peaks.

        The method name preserves the historical spelling for compatibility.

        Args:
            crystal_sys (int): Crystal-system code.
            old_p1_list (Sequence[float]): Previous or observed peak positions.
            p1_list (Sequence[float]): Candidate peak positions.
            subset_number (int): Number of peaks used for fitting.
            low_bound (float): Lower 2-theta bound for candidate peaks.
            up_bound (float): Upper 2-theta bound for candidate peaks.
            lattice_h, lattice_k, lattice_l (Sequence[int]): HKL component
                arrays aligned to candidate peaks.
            ini_a, ini_b, ini_c (float): Initial lattice lengths.
            ini_la1, ini_la2, ini_la3 (float): Initial lattice angles.
            wavelength (Sequence[float]): One or two X-ray wavelengths.
            fixed (bool): Return the initial lattice constants without fitting.
            tao (float): Finite-difference perturbation used for learning-rate
                estimation.

        Returns:
            tuple: Optimized lattice constants in
            ``a, b, c, alpha, beta, gamma`` order.
        """
        if fixed == False:
            # updata latt params by Bragg
            # defined for mixed rays have two wavelength
            if len( wavelength) == 2:
                error_fit = []
                for i in range(int(len( old_p1_list) / 2)):
                    mui_error1 = ( p1_list[2 * i] - old_p1_list[2 * i]) ** 2
                    mui_error2 = ( p1_list[2 * i + 1] - old_p1_list[2 * i + 1]) ** 2
                    if mui_error1 >= mui_error2:
                        error_fit.append(mui_error1)
                    else:
                        error_fit.append(mui_error2)
                error_fit_sort = copy.deepcopy(error_fit)
                error_fit_sort.sort()

                h_fit_abc, k_fit_abc, l_fit_abc = [], [], []
                mui_fit_abc_list = []

                # search index
                for i_err in range(len(error_fit)):
                    for j in range(len(error_fit)):
                        if error_fit_sort[i_err] == error_fit[j] and low_bound <= p1_list[2 * j] <= up_bound:
                            for t in range(len(  wavelength)):
                                mui_fit_abc_list.append(p1_list[2 * j + t])
                            h_fit_abc.append(lattice_h[j])
                            k_fit_abc.append(lattice_k[j])
                            l_fit_abc.append(lattice_l[j])
                    if len(h_fit_abc) ==   subset_number:
                        break
                total_peak_n = len(mui_fit_abc_list)

            elif len(wavelength) == 1:
                error_fit = []
                for i in range(len(old_p1_list)):
                    mui_error = (p1_list[i] - old_p1_list[i]) ** 2
                    error_fit.append(mui_error)

                error_fit_sort = copy.deepcopy(error_fit)
                error_fit_sort.sort()

                h_fit_abc, k_fit_abc, l_fit_abc = [], [], []
                mui_fit_abc_list = []

                # search index
                for i_err in range(len(error_fit)):
                    for j in range(len(error_fit)):
                        if error_fit_sort[i_err] == error_fit[j] and low_bound <= p1_list[j] <= up_bound:
                            mui_fit_abc_list.append(  p1_list[j])
                            h_fit_abc.append(lattice_h[j])
                            k_fit_abc.append(lattice_k[j])
                            l_fit_abc.append(lattice_l[j])
                    if len(h_fit_abc) ==   subset_number:
                        break
                total_peak_n = len(mui_fit_abc_list)

            else:
                print('only for mixed rays with one or two wavelength')

            i_ter = 0
            if   crystal_sys == 1:
                cal_a = copy.deepcopy(ini_a)
                while (True):
                    i_ter += 1
                    ini_a = copy.deepcopy(cal_a)

                    derive_term = self.update_derivative(crystal_sys, total_peak_n, h_fit_abc, k_fit_abc, l_fit_abc,
                                                        ini_a, ini_a, ini_a, ini_la1, ini_la2, ini_la3, wavelength, mui_fit_abc_list)

                    lef_derive = self.two_up_derivative( crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a - tao,
                                                        ini_a, ini_a, ini_la1, ini_la2, ini_la3, 1, wavelength, mui_fit_abc_list)

                    right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a + tao,
                                                        ini_a, ini_a, ini_la1, ini_la2, ini_la3, 1, wavelength, mui_fit_abc_list)

                    learning_rate = 2 * tao / (right_derive - lef_derive)

                    cal_a = ini_a - derive_term[0] * learning_rate

                    if abs(cal_a - ini_a) <= 1e-8:
                        break

                    if i_ter >= 800:
                        break
                d_list, _, _ = self.get_d_space( crystal_sys, lattice_h, lattice_k,
                                                              lattice_l, cal_a, cal_a, cal_a, ini_la1, ini_la2, ini_la3)
                mui_fit =  self.get_new_mui(d_list, wavelength)

                return cal_a, cal_a, cal_a, ini_la1, ini_la2, ini_la3, mui_fit

            elif crystal_sys == 2 or crystal_sys == 3:
                cal_a = copy.deepcopy(ini_a)
                cal_c = copy.deepcopy(ini_c)

                while (True):
                    i_ter += 1
                    ini_a = copy.deepcopy(cal_a)
                    ini_c = copy.deepcopy(cal_c)

                    derive_term = self.update_derivative( crystal_sys, total_peak_n, h_fit_abc, k_fit_abc, l_fit_abc,
                                                        ini_a, ini_a, ini_c, ini_la1, ini_la2, ini_la3, wavelength,
                                                        mui_fit_abc_list)

                    a_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc,
                                                        ini_a - tao, ini_a, ini_c, ini_la1, ini_la2, ini_la3, 1,
                                                          wavelength, mui_fit_abc_list)
                    c_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_a,
                                                        ini_c - tao, ini_la1, ini_la2, ini_la3, 3,wavelength, mui_fit_abc_list)

                    a_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc,
                                                            ini_a + tao, ini_a, ini_c, ini_la1, ini_la2, ini_la3, 1,
                                                              wavelength, mui_fit_abc_list)
                    c_right_derive = self.two_up_derivative( crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_a,
                                                            ini_c + tao, ini_la1, ini_la2, ini_la3, 3, wavelength, mui_fit_abc_list)

                    a_learning_rate = 2 * tao / (a_right_derive - a_lef_derive)
                    c_learning_rate = 2 * tao / (c_right_derive - c_lef_derive)

                    cal_a = ini_a - derive_term[0] * a_learning_rate
                    cal_c = ini_c - derive_term[1] * c_learning_rate

                    if abs(cal_a - ini_a) <= 1e-8 and abs(cal_c - ini_c) <= 1e-8:
                        break

                    if i_ter >= 800:
                        break
                d_list, _, _ = self.get_d_space(crystal_sys, lattice_h,lattice_k,
                                                              lattice_l, cal_a, cal_a, cal_c, ini_la1, ini_la2, ini_la3)
                mui_fit = self.get_new_mui(d_list, wavelength)

                return cal_a, cal_a, cal_c, ini_la1, ini_la2, ini_la3, mui_fit

            elif crystal_sys == 4:
                cal_a = copy.deepcopy(ini_a)
                cal_b = copy.deepcopy(ini_b)
                cal_c = copy.deepcopy(ini_c)

                while (True):
                    i_ter += 1
                    ini_a = copy.deepcopy(cal_a)
                    ini_b = copy.deepcopy(cal_b)
                    ini_c = copy.deepcopy(cal_c)

                    derive_term = self.update_derivative(crystal_sys, total_peak_n, h_fit_abc, k_fit_abc, l_fit_abc,
                                                        ini_a, ini_b, ini_c, ini_la1, ini_la2, ini_la3, wavelength,
                                                        mui_fit_abc_list)

                    a_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a - tao, ini_b,
                                                        ini_c, ini_la1, ini_la2, ini_la3, 1, wavelength, mui_fit_abc_list)
                    b_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a,ini_b - tao,
                                                        ini_c, ini_la1, ini_la2, ini_la3, 2, wavelength, mui_fit_abc_list)
                    c_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_b, ini_c - tao, ini_la1, ini_la2, ini_la3, 3,
                                                          wavelength, mui_fit_abc_list)

                    a_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc,ini_a + tao, ini_b,
                                                            ini_c, ini_la1, ini_la2, ini_la3, 1,wavelength, mui_fit_abc_list)

                    b_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a,
                                                            ini_b + tao, ini_c, ini_la1, ini_la2, ini_la3, 2,wavelength, mui_fit_abc_list)
                    c_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_b,
                                                            ini_c + tao, ini_la1, ini_la2, ini_la3, 3, wavelength, mui_fit_abc_list)

                    a_learning_rate = 2 * tao / (a_right_derive - a_lef_derive)
                    b_learning_rate = 2 * tao / (b_right_derive - b_lef_derive)
                    c_learning_rate = 2 * tao / (c_right_derive - c_lef_derive)

                    cal_a = ini_a - derive_term[0] * a_learning_rate
                    cal_b = ini_b - derive_term[1] * b_learning_rate
                    cal_c = ini_c - derive_term[2] * c_learning_rate

                    if abs(cal_a - ini_a) <= 1e-8 and abs(cal_b - ini_b) <= 1e-8 and abs(cal_c - ini_c) <= 1e-8 or abs(cal_a - ini_a) >= 0.01 *ini_a or abs(cal_b - ini_b) >= 0.01*ini_b or abs(cal_c - ini_c) >= 0.01*ini_c:
                        break

                    if i_ter >= 400:
                        break
                d_list, _, _ = self.get_d_space(crystal_sys, lattice_h, lattice_k,
                                                              lattice_l, cal_a, cal_b, cal_c, ini_la1, ini_la2, ini_la3)
                mui_fit = self.get_new_mui(d_list,wavelength)

                return cal_a, cal_b, cal_c, ini_la1, ini_la2, ini_la3, mui_fit

            elif crystal_sys == 5:
                cal_a = copy.deepcopy(ini_a)
                cal_la1 = copy.deepcopy(ini_la1)

                while (True):
                    i_ter += 1
                    ini_a = copy.deepcopy(cal_a)
                    ini_la1 = copy.deepcopy(cal_la1)

                    derive_term = self.update_derivative(crystal_sys, total_peak_n, h_fit_abc, k_fit_abc, l_fit_abc,
                                                        ini_a, ini_a, ini_a, ini_la1, ini_la1, ini_la1, wavelength,
                                                        mui_fit_abc_list)

                    a_lef_derive = self.two_up_derivative( crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc,
                                                        ini_a - tao, ini_a, ini_a, ini_la1, ini_la1, ini_la1, 1, wavelength,
                                                        mui_fit_abc_list)
                    la1_lef_derive = self.two_up_derivative( crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_a,
                                                            ini_a, ini_la1 - tao, ini_la1, ini_la1, 4, wavelength,
                                                            mui_fit_abc_list)

                    a_right_derive = self.two_up_derivative( crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc,
                                                            ini_a +   tao, ini_a, ini_a, ini_la1, ini_la1, ini_la1, 1, wavelength,
                                                            mui_fit_abc_list)
                    la1_right_derive = self.two_up_derivative( crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a,
                                                            ini_a, ini_a, ini_la1 + tao, ini_la1, ini_la1, 4, wavelength,
                                                            mui_fit_abc_list)

                    a_learning_rate = 2 * tao / (a_right_derive - a_lef_derive)
                    la1_learning_rate = 2 * tao / (la1_right_derive - la1_lef_derive)

                    cal_a = ini_a - derive_term[0] * a_learning_rate
                    cal_la1 = ini_la1 - derive_term[1] * la1_learning_rate

                    if abs(cal_a - ini_a) <= 1e-8 and abs(cal_la1 - ini_la1) <= 1e-8:
                        break

                    if i_ter >= 800:
                        break
                d_list, _, _ = self.get_d_space( crystal_sys, lattice_h, lattice_k, lattice_l, cal_a, cal_a,
                                                            cal_a, cal_la1, cal_la1, cal_la1)
                mui_fit = self.get_new_mui(d_list, wavelength)

                return cal_a, cal_a, cal_a, cal_la1, cal_la1, cal_la1, mui_fit

            elif crystal_sys == 6:
                cal_a = copy.deepcopy(ini_a)
                cal_b = copy.deepcopy(ini_b)
                cal_c = copy.deepcopy(ini_c)
                cal_la2 = copy.deepcopy(ini_la2)

                while (True):
                    i_ter += 1
                    ini_a = copy.deepcopy(cal_a)
                    ini_b = copy.deepcopy(cal_b)
                    ini_c = copy.deepcopy(cal_c)
                    ini_la2 = copy.deepcopy(cal_la2)

                    derive_term = self.update_derivative(crystal_sys, total_peak_n, h_fit_abc, k_fit_abc, l_fit_abc,
                                                        ini_a, ini_b, ini_c, ini_la1, ini_la2, ini_la3, wavelength,
                                                        mui_fit_abc_list)

                    # each time we only update one lattice constant randomly
                    random_number = random.randint(0, 3)
                    if random_number <= 2:
                        a_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a - tao, ini_b,
                                                        ini_c, ini_la1, ini_la2, ini_la3, 1, wavelength, mui_fit_abc_list)
                        b_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a,ini_b - tao,
                                                            ini_c, ini_la1, ini_la2, ini_la3, 2, wavelength, mui_fit_abc_list)
                        c_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_b, ini_c - tao,
                                                            ini_la1, ini_la2, ini_la3, 3, wavelength, mui_fit_abc_list)

                        a_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc,ini_a + tao, ini_b,
                                                                ini_c, ini_la1, ini_la2, ini_la3, 1,wavelength, mui_fit_abc_list)

                        b_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a,
                                                                ini_b + tao, ini_c, ini_la1, ini_la2, ini_la3, 2,wavelength, mui_fit_abc_list)
                        c_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_b,
                                                                ini_c + tao, ini_la1, ini_la2, ini_la3, 3, wavelength, mui_fit_abc_list)

                        a_learning_rate = 2 * tao / (a_right_derive - a_lef_derive)
                        b_learning_rate = 2 * tao / (b_right_derive - b_lef_derive)
                        c_learning_rate = 2 * tao / (c_right_derive - c_lef_derive)

                        cal_a = ini_a - derive_term[0] * a_learning_rate
                        cal_b = ini_b - derive_term[1] * b_learning_rate
                        cal_c = ini_c - derive_term[2] * c_learning_rate

                        if abs(cal_a - ini_a) <= 1e-8 and abs(cal_b - ini_b) <= 1e-8 and abs(cal_c - ini_c) <= 1e-8 or abs(cal_a - ini_a) >= 0.01 *ini_a or abs(cal_b - ini_b) >= 0.01*ini_b or abs(cal_c - ini_c) >= 0.01*ini_c:
                            break

                        if i_ter >= 400:
                            break
                        d_list, _, _ = self.get_d_space(crystal_sys,lattice_h,lattice_k, lattice_l, cal_a, cal_b, cal_c,ini_la1, ini_la2,ini_la3)
                        mui_fit = self.get_new_mui(d_list,wavelength)
                        return cal_a, cal_b, cal_c, ini_la1, ini_la2, ini_la3, mui_fit

                    else:
                        la2_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_b,
                                                                ini_c, ini_la1, ini_la2 - tao, ini_la3, 5, wavelength, mui_fit_abc_list)
                        la2_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a,
                                                                ini_b, ini_c, ini_la1, ini_la2 + tao, ini_la3, 5,wavelength, mui_fit_abc_list)
                        la2_learning_rate = 2 * tao / (la2_right_derive - la2_lef_derive)
                        cal_la2 = ini_la2 - derive_term[3] * la2_learning_rate
                        if abs(cal_la2 - ini_la2) <=  1e-8 or abs(cal_la2 - ini_la2) > 0.01*ini_la2:
                            break
                        if i_ter >= 400:
                            break
                        d_list, _, _ = self.get_d_space(crystal_sys,lattice_h,lattice_k, lattice_l, ini_a, ini_b, ini_c,ini_la1, cal_la2,ini_la3)
                        mui_fit = self.get_new_mui(d_list,wavelength)
                        return ini_a, ini_b, ini_c, ini_la1, cal_la2, ini_la3, mui_fit

            elif crystal_sys == 7:
                # original version Sep 19 2023
                cal_a = copy.deepcopy(ini_a)
                cal_b = copy.deepcopy(ini_b)
                cal_c = copy.deepcopy(ini_c)
                cal_la1 = copy.deepcopy(ini_la1)
                cal_la2 = copy.deepcopy(ini_la2)
                cal_la3 = copy.deepcopy(ini_la3)

                while (True):
                    i_ter += 1
                    ini_a = copy.deepcopy(cal_a)
                    ini_b = copy.deepcopy(cal_b)
                    ini_c = copy.deepcopy(cal_c)
                    ini_la1 = copy.deepcopy(cal_la1)
                    ini_la2 = copy.deepcopy(cal_la2)
                    ini_la3 = copy.deepcopy(cal_la3)

                    derive_term =  self.update_derivative(crystal_sys, total_peak_n, h_fit_abc, k_fit_abc, l_fit_abc,
                                                        ini_a, ini_b, ini_c, ini_la1, ini_la2, ini_la3, wavelength,
                                                        mui_fit_abc_list)

                    a_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc,
                                                        ini_a - tao, ini_b, ini_c, ini_la1, ini_la2, ini_la3, 1, wavelength,
                                                        mui_fit_abc_list)

                    b_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a,
                                                        ini_b - tao, ini_c, ini_la1, ini_la2, ini_la3, 2, wavelength,
                                                        mui_fit_abc_list)

                    c_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_b,
                                                        ini_c - tao, ini_la1, ini_la2, ini_la3, 3, wavelength, mui_fit_abc_list)

                    la1_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_b,
                                                            ini_c, ini_la1 - tao, ini_la2, ini_la3, 4, wavelength,mui_fit_abc_list)

                    la2_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_b,
                                                            ini_c, ini_la1, ini_la2 - tao, ini_la3, 5, wavelength,mui_fit_abc_list)

                    la3_lef_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_b,
                                                            ini_c, ini_la1, ini_la2, ini_la3 - tao, 6, wavelength,mui_fit_abc_list)

                    a_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc,
                                                            ini_a + tao, ini_b, ini_c, ini_la1, ini_la2, ini_la3, 1, wavelength,
                                                            mui_fit_abc_list)

                    b_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a,
                                                            ini_b + tao, ini_c, ini_la1, ini_la2, ini_la3, 2,wavelength,
                                                            mui_fit_abc_list)

                    c_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a, ini_b,
                                                            ini_c + tao, ini_la1, ini_la2, ini_la3, 3, wavelength,
                                                            mui_fit_abc_list)

                    la1_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a,
                                                            ini_b, ini_c, ini_la1 + tao, ini_la2, ini_la3, 4, wavelength,
                                                            mui_fit_abc_list)

                    la2_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a,
                                                            ini_b, ini_c, ini_la1, ini_la2 + tao, ini_la3, 5, wavelength,
                                                            mui_fit_abc_list)

                    la3_right_derive = self.two_up_derivative(crystal_sys, h_fit_abc, k_fit_abc, l_fit_abc, ini_a,
                                                            ini_b, ini_c, ini_la1, ini_la2, ini_la3 + tao, 6, wavelength,
                                                            mui_fit_abc_list)

                    a_learning_rate = 2 * tao / (a_right_derive - a_lef_derive)
                    b_learning_rate = 2 * tao / (b_right_derive - b_lef_derive)
                    c_learning_rate = 2 * tao / (c_right_derive - c_lef_derive)
                    la1_learning_rate = 2 * tao / (la1_right_derive - la1_lef_derive)
                    la2_learning_rate = 2 * tao / (la2_right_derive - la2_lef_derive)
                    la3_learning_rate = 2 * tao / (la3_right_derive - la3_lef_derive)

                    cal_a = ini_a - derive_term[0] * a_learning_rate
                    cal_b = ini_b - derive_term[1] * b_learning_rate
                    cal_c = ini_c - derive_term[2] * c_learning_rate
                    cal_la1 = ini_la1 - derive_term[3] * la1_learning_rate
                    cal_la2 = ini_la2 - derive_term[4] * la2_learning_rate
                    cal_la3 = ini_la3 - derive_term[5] * la3_learning_rate

                    if abs(cal_a - ini_a) <= 1e-8 and abs(cal_b - ini_b) <= 1e-8 and abs(cal_c - ini_c) <= 1e-8 \
                            and abs(cal_la1 - ini_la1) <= 1e-8 and abs(cal_la2 - ini_la2) <= 1e-8 and abs(
                        cal_la3 - ini_la3) <= 1e-8:
                        break

                    if i_ter >= 400:
                        break
                d_list, _, _ = self.get_d_space(crystal_sys, lattice_h, lattice_k,
                                                              lattice_l, cal_a, cal_b,cal_c, cal_la1, cal_la2, cal_la3)
                mui_fit = self.get_new_mui(d_list, wavelength)

                return cal_a, cal_b, cal_c, cal_la1, cal_la2, cal_la3, mui_fit

            else:
                return -1

        if fixed == True:
            return ini_a, ini_b , ini_c , ini_la1 , ini_la2, ini_la3, old_p1_list
