from __future__ import division

"""Features that describe the local environment of a single atom
The `featurize` function takes two arguments:
    struct (Structure): Object representing the structure containing the site 
        of interest
    idx (int): Index of the site to be featurized
We have to use two parameters because the Site object does not hold a pointer 
back to its structure and often information on neighbors is required. To run
:code:`featurize_dataframe`, you must pass the column names for both the site 
index and the structure. For example:
.. code:: python
    f = AGNIFingerprints()
    f.featurize_dataframe(data, ['structure', 'site_idx'])
"""

import numpy as np
import math

from collections import defaultdict

from matminer.featurizers.base import BaseFeaturizer
from pymatgen.analysis.structure_analyzer import VoronoiAnalyzer
from pymatgen.analysis.local_env import LocalStructOrderParas, \
    VoronoiNN, JMolNN, MinimumDistanceNN, MinimumOKeeffeNN, \
    MinimumVIRENN, get_neighbors_of_site_with_index
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder \
    import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies \
   import SimplestChemenvStrategy, MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.core.periodic_table import Specie, get_el_sp

from matminer.featurizers.stats import PropertyStats


class AGNIFingerprints(BaseFeaturizer):
    """Integral of the product of the radial distribution function and a
    Gaussian window function. Originally used by [Botu *et al*]
    (http://pubs.acs.org/doi/abs/10.1021/acs.jpcc.6b10908) to fit empiricial
    potentials. These features come in two forms: atomic fingerprints and
    direction-resolved fingerprints.
    Atomic fingerprints describe the local environment of an atom and are
    computed using the function:
    :math:`A_i(\eta) = \sum\limits_{i \ne j} e^{-(\frac{r_{ij}}{\eta})^2} f(r_{ij})`
    where :math:`i` is the index of the atom, :math:`j` is the index of a neighboring atom, :math:`\eta` is a scaling function,
    :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`, and :math:`f(r)` is a cutoff function where
    :math:`f(r) = 0.5[cos(\frac{\pi r_{ij}}{R_c}) + 1]` if :math:`r < R_c:math:` and 0 otherwise.

    The direction-resolved fingerprints are computed using
    :math:`V_i^k(\eta) = \sum\limits_{i \ne j} \frac{r_{ij}^k}{r_{ij}} e^{-(\frac{r_{ij}}{\eta})^2} f(r_{ij})`
    where :math:`r_{ij}^k` is the :math:`k^{th}` component of :math:`\bold{r}_i - \bold{r}_j`.

    Parameters:

    TODO: Differentiate between different atom types (maybe as another class)
    """

    def __init__(self, directions=(None, 'x', 'y', 'z'), etas=None,
                 cutoff=8):
        """

        Args:
            directions (iterable): List of directions for the fingerprints. Can
                be one or more of 'None`, 'x', 'y', or 'z'
            etas (iterable of floats): List of which window widths to compute
            cutoff (float): Cutoff distance (Angstroms)
        """
        self.directions = directions
        self.etas = etas
        if self.etas is None:
            self.etas = np.logspace(np.log10(0.8), np.log10(16), 8)
        self.cutoff = cutoff

    def featurize(self, struct, idx):
        # Get all neighbors of this site
        my_site = struct[idx]
        sites, dists = zip(*struct.get_neighbors(my_site, self.cutoff))

        # Convert dists to a ndarray
        dists = np.array(dists)

        # If one of the features is direction-dependent, compute the :math:`(r_i - r_j) / r_{ij}`
        if any([x in self.directions for x in ['x', 'y', 'z']]):
            disps = np.array(
                [my_site.coords - s.coords for s in sites]) / dists[:,
                                                              np.newaxis]

        # Compute the cutoff function
        cutoff_func = 0.5 * (np.cos(np.pi * dists / self.cutoff) + 1)

        # Compute "e^(r/eta) * cutoff_func" for each eta
        windowed = np.zeros((len(dists), len(self.etas)))
        for i, eta in enumerate(self.etas):
            windowed[:, i] = np.multiply(
                np.exp(-1 * np.power(np.true_divide(dists, eta), 2)),
                cutoff_func)

        # Compute the fingerprints
        output = []
        for d in self.directions:
            if d is None:
                output.append(np.sum(windowed, axis=0))
            else:
                if d == 'x':
                    proj = [1., 0., 0.]
                elif d == 'y':
                    proj = [0., 1., 0.]
                elif d == 'z':
                    proj = [0., 0., 1.]
                else:
                    raise Exception('Unrecognized direction')
                output.append(
                    np.sum(windowed * np.dot(disps, proj)[:, np.newaxis],
                           axis=0))

        # Return the results
        return np.hstack(output)

    def feature_labels(self):
        labels = []
        for d in self.directions:
            for e in self.etas:
                if d is None:
                    labels.append('AGNI eta=%.2e' % e)
                else:
                    labels.append('AGNI dir=%s eta=%.2e' % (d, e))
        return labels

    def citations(self):
        return ["@article{Botu2015, author = {Botu, Venkatesh and Ramprasad, Rampi},doi = {10.1002/qua.24836}," \
               "journal = {International Journal of Quantum Chemistry},number = {16},pages = {1074--1083}," \
               "title = {{Adaptive machine learning framework to accelerate ab initio molecular dynamics}}," \
               "volume = {115},year = {2015}}"]

    def implementors(self):
        return ['Logan Ward']


class OPSiteFingerprint(BaseFeaturizer):
    """
    Local structure order parameters computed from the neighbor
    environment of a site. For each order parameter, we determine
    the neighbor shell that complies with the expected
    coordination number. For example, we find the 4 nearest
    neighbors for the tetrahedral OP, the 6 nearest for the
    octahedral OP, and the 8 nearest neighbors for the bcc OP.
    If we don't find such a shell, the OP is either set to zero
    or evaluated with the shell of the next largest observed
    coordination number.
    Args:
        dr (float): width for binning neighbors in unit of relative
                    distances (= distance/nearest neighbor
                    distance).  The binning is necessary to make the
                    neighbor-finding step robust against small numerical
                    variations in neighbor distances (default: 0.1).
        ddr (float): variation of width for finding stable OP values.
        ndr (int): number of width variations for each variation direction
                   (e.g., ndr = 0 only uses the input dr, whereas
                   ndr=1 tests dr = dr - ddr, dr, and dr + ddr.
        dop (float): binning width to compute histogram for each OP
                     if ndr > 0.
        dist_exp (boolean): exponent for distance factor to multiply
                            order parameters with that penalizes (large)
                            variations in distances in a given motif.
                            0 will switch the option off
                            (default: 2).
        zero_ops (boolean): set an OP to zero if there is no neighbor
                            shell that complies with the expected
                            coordination number of a given OP
                            (e.g., CN=4 for tetrahedron;
                            default: True).
    """

    def __init__(self, optypes=None, dr=0.1, ddr=0.01, ndr=1, dop=0.001,
                 dist_exp=2, zero_ops=True):
        self.optypes = {
            1: ["sgl_bd"],
            2: ["bent180", "bent45", "bent90", "bent135"],
            3: ["tri_plan", "tet", "T"],
            4: ["sq_plan", "sq", "tet", "see_saw_rect", "tri_pyr"],
            5: ["pent_plan", "sq_pyr", "tri_bipyr"],
            6: ["oct", "pent_pyr"],
            7: ["hex_pyr", "pent_bipyr"],
            8: ["bcc", "hex_bipyr"],
            9: ["q2", "q4", "q6"],
            10: ["q2", "q4", "q6"],
            11: ["q2", "q4", "q6"],
            12: ["cuboct", "q2", "q4", "q6"]} if optypes is None \
            else optypes.copy()
        self.dr = dr
        self.ddr = ddr
        self.ndr = ndr
        self.dop = dop
        self.dist_exp = dist_exp
        self.zero_ops = zero_ops
        self.ops = {}
        for cn, t_list in self.optypes.items():
            self.ops[cn] = []
            for t in t_list:
                if t[:4] == 'bent':
                    self.ops[cn].append(LocalStructOrderParas(
                        [t[:4]], parameters=[{'TA': float(t[4:]) / 180.0, \
                                              'IGW_TA': 1.0 / 0.0667}]))
                else:
                    self.ops[cn].append(LocalStructOrderParas([t]))

    def featurize(self, struct, idx):
        """
        Get OP fingerprint of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            opvals (numpy array): order parameters of target site.
        """
        idop = 1.0 / self.dop
        opvals = {}
        s = struct.sites[idx]
        neigh_dist = []
        r = 6
        while len(neigh_dist) < 12:
            r += 1.0
            neigh_dist = struct.get_neighbors(s, r)

        # Smoothen distance, but use relative distances.
        dmin = min([d for n, d in neigh_dist])
        neigh_dist = [[n, d / dmin] for n, d in neigh_dist]
        neigh_dist_alldrs = {}
        d_sorted_alldrs = {}

        for i in range(-self.ndr, self.ndr + 1):
            opvals[i] = []
            this_dr = self.dr + float(i) * self.ddr
            this_idr = 1.0 / this_dr
            neigh_dist_alldrs[i] = []
            for j in range(len(neigh_dist)):
                neigh_dist_alldrs[i].append([neigh_dist[j][0],
                                             (float(
                                                 int(neigh_dist[j][1] * this_idr \
                                                     + 0.5)) + 0.5) * this_dr])
            d_sorted_alldrs[i] = []
            for n, d in neigh_dist_alldrs[i]:
                if d not in d_sorted_alldrs[i]:
                    d_sorted_alldrs[i].append(d)
            d_sorted_alldrs[i] = sorted(d_sorted_alldrs[i])

        # Do q_sgl_bd separately.
        if self.optypes[1][0] == "sgl_bd":
            for i in range(-self.ndr, self.ndr + 1):
                site_list = [s]
                for n, dn in neigh_dist_alldrs[i]:
                    site_list.append(n)
                opval = self.ops[1][0].get_order_parameters(
                    site_list, 0,
                    indices_neighs=[j for j in range(1, len(site_list))])
                opvals[i].append(opval[0])

        for i in range(-self.ndr, self.ndr + 1):
            prev_cn = 0
            for d in d_sorted_alldrs[i]:
                this_cn = 0
                site_list = [s]
                this_av_inv_drel = 0.0
                for j, [n, dn] in enumerate(neigh_dist_alldrs[i]):
                    if dn <= d:
                        this_cn += 1
                        site_list.append(n)
                        this_av_inv_drel += (1.0 / (neigh_dist[j][1]))
                this_av_inv_drel = this_av_inv_drel / float(this_cn)
                d_fac = this_av_inv_drel ** self.dist_exp
                for cn in range(max(2, prev_cn + 1), min(this_cn + 1, 13)):
                    # Set all OPs of non-CN-complying neighbor environments
                    # to zero if applicable.
                    if self.zero_ops and cn != this_cn:
                        for it in range(len(self.optypes[cn])):
                            opvals[i].append(0)
                        continue

                    # Set all (remaining) OPs.
                    for it in range(len(self.optypes[cn])):
                        opval = self.ops[cn][it].get_order_parameters(
                            site_list, 0,
                            indices_neighs=[j for j in
                                            range(1, len(site_list))])
                        if opval[0] is None:
                            opval[0] = 0
                        else:
                            opval[0] = d_fac * opval[0]
                        opvals[i].append(opval[0])
                prev_cn = this_cn
                if prev_cn >= 12:
                    break

        opvals_out = []

        for j in range(len(opvals[0])):
            # Compute histogram, determine peak, and location
            # of peak value.
            op_tmp = [opvals[i][j] for i in range(-self.ndr, self.ndr + 1)]
            minval = float(int(min(op_tmp) * idop - 1.5)) * self.dop
            # print(minval)
            if minval < 0.0:
                minval = 0.0
            if minval > 1.0:
                minval = 1.0
            # print(minval)
            maxval = float(int(max(op_tmp) * idop + 1.5)) * self.dop
            # print(maxval)
            if maxval < 0.0:
                maxval = 0.0
            if maxval > 1.0:
                maxval = 1.0
            # print(maxval)
            if minval == maxval:
                minval = minval - self.dop
                maxval = maxval + self.dop
            # print(minval)
            # print(maxval)
            nbins = int((maxval - minval) * idop)
            # print('{} {} {}'.format(minval, maxval, nbins))
            hist, bin_edges = np.histogram(
                op_tmp, bins=nbins, range=(minval, maxval),
                normed=False, weights=None, density=False)
            max_hist = max(hist)
            op_peaks = []
            for i, h in enumerate(hist):
                if h == max_hist:
                    op_peaks.append(
                        [i, 0.5 * (bin_edges[i] + bin_edges[i + 1])])
            # Address problem that 2 OP values can be close to a bin edge.
            hist2 = []
            op_peaks2 = []
            i = 0
            while i < len(op_peaks):
                if i < len(op_peaks) - 1:
                    if op_peaks[i + 1][0] - op_peaks[i][0] == 1:
                        op_peaks2.append(
                            0.5 * (op_peaks[i][1] + op_peaks[i + 1][1]))
                        hist2.append(
                            hist[op_peaks[i][0]] + hist[op_peaks[i + 1][0]])
                        i += 1
                    else:
                        op_peaks2.append(op_peaks[i][1])
                        hist2.append(hist[op_peaks[i][0]])
                else:
                    op_peaks2.append(op_peaks[i][1])
                    hist2.append(hist[op_peaks[i][0]])
                i += 1
            opvals_out.append(op_peaks2[list(hist2).index(max(hist2))])
        return np.array(opvals_out)

    def feature_labels(self):
        labels = []
        for cn, li in self.optypes.items():
            for e in li:
                labels.append('{} CN_{}'.format(e, cn))
        return labels

    def citations(self):
        return ['@article{zimmermann_jain_2017, title={Applications of order'
                ' parameter feature vectors}, journal={in progress}, author={'
                'Zimmermann, N. E. R. and Jain, A.}, year={2017}}']

    def implementors(self):
        return ['Nils E. R. Zimmermann']


# TODO: unit tests!!
class CrystalSiteFingerprint(BaseFeaturizer):
    """
    A site fingerprint intended for periodic crystals. The fingerprint represents
    the value of various order parameters for the site; each value is the product
    two quantities: (i) the value of the order parameter itself and (ii) a factor
    that describes how consistent the number of neighbors is with that order
    parameter. Note that we can include only factor (ii) using the "wt" order
    parameter which is always set to 1.
    """

    @staticmethod
    def from_preset(preset, cation_anion=False):
        """
        Use preset parameters to get the fingerprint

        Args:
            preset (str): name of preset ("cn" or "ops")
            cation_anion (bool): whether to only consider cation<->anion bonds
                (bonds with zero charge are also allowed)
        """
        if preset == "cn":
            optypes = dict([(k + 1, ["wt"]) for k in range(16)])
            return CrystalSiteFingerprint(optypes, cation_anion=cation_anion)

        elif preset == "ops":
            optypes = {
                1: ["wt"],
                2: ["wt", "bent180", "bent45", "bent90", "bent135"],
                3: ["wt", "tri_plan", "tet", "T"],
                4: ["wt", "sq_plan", "sq", "tet", "see_saw_rect", "tri_pyr"],
                5: ["wt", "pent_plan", "sq_pyr", "tri_bipyr"],
                6: ["wt", "oct", "pent_pyr"],
                7: ["wt", "hex_pyr", "pent_bipyr"],
                8: ["wt", "bcc", "hex_bipyr"],
                9: ["wt", "q2", "q4", "q6"],
                10: ["wt", "q2", "q4", "q6"],
                11: ["wt", "q2", "q4", "q6"],
                12: ["wt", "cuboct", "q2", "q4", "q6"],
                13: ["wt"],
                14: ["wt"],
                15: ["wt"],
                16: ["wt"]}

            return CrystalSiteFingerprint(optypes, cation_anion=cation_anion)

    def __init__(self, optypes, override_cn1=True, cutoff_radius=8, tol=1E-2,
                 cation_anion=False):
        """
        Initialize the CrystalSiteFingerprint. Use the from_preset() function to
        use default params.

        Args:
            optypes (dict): a dict of coordination number (int) to a list of str
                representing the order parameter types
            override_cn1 (bool): whether to use a special function for the single
                neighbor case. Suggest to keep True.
            cutoff_radius (int): radius in Angstroms for neighbor finding
            tol (float): numerical tolerance (in case your site distances are
                not perfect or to correct for float tolerances)
            cation_anion (bool): whether to only consider cation<->anion bonds
                (bonds with zero charge are also allowed)
        """

        self.optypes = optypes.copy()
        self.override_cn1 = override_cn1
        self.cutoff_radius = cutoff_radius
        self.tol = tol
        self.cation_anion = cation_anion

        if self.override_cn1 and self.optypes.get(1) != ["wt"]:
            raise ValueError(
                "If override_cn1 is True, optypes[1] must be ['wt']!")

        self.ops = {}
        for cn, t_list in self.optypes.items():
            self.ops[cn] = []
            for t in t_list:
                if t == "wt":
                    self.ops[cn].append(t)

                elif t[:4] == 'bent':
                    self.ops[cn].append(LocalStructOrderParas(
                        [t[:4]], parameters=[{'TA': float(t[4:]) / 180.0, \
                                              'IGW_TA': 1.0 / 0.0667}]))
                else:
                    self.ops[cn].append(LocalStructOrderParas([t]))

    def featurize(self, struct, idx):
        """
        Get crystal fingerprint of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            list of weighted order parameters of target site.
        """

        cn_fingerprint_array = defaultdict(
            list)  # dict where key = CN, val is array that contains each OP for that CN
        total_weight = math.pi / 4  # 1/4 unit circle area

        target = None
        if self.cation_anion:
            target = []
            m_oxi = struct[idx].specie.oxi_state
            for site in struct:
                if site.specie.oxi_state * m_oxi <= 0:  # opposite charge
                    target.append(site.specie)
            if not target:
                raise ValueError(
                    "No valid targets for site within cation_anion constraint!")

        vnn = VoronoiNN(cutoff=self.cutoff_radius,
                        target=target)
        n_w = vnn.get_voronoi_polyhedra(idx, struct, use_weights=True)

        dist_sorted = (sorted(n_w.values(), reverse=True))

        if self.override_cn1:
            cn1 = 1
            for d in dist_sorted[1:]:
                cn1 = cn1 * (dist_sorted[0] ** 2 - d ** 2) / dist_sorted[0] ** 2
            cn_fingerprint_array[1] = [round(cn1, 6)]
            dist_sorted[0] = dist_sorted[1]

        dist_norm = [d / dist_sorted[0] for d in dist_sorted if d > 0]

        dist_bins = []  # bin numerical tolerances (~error bar of measurement)
        for d in dist_norm:
            if not dist_bins or (
                    d > self.tol and dist_bins[-1] / (1 + self.tol) > d):
                dist_bins.append(d)

        for dist_idx, dist in enumerate(dist_bins):
            neigh_sites = [n for n, w in n_w.items() if
                           w > 0 and w / dist_sorted[0] >= dist / (
                                   1 + self.tol)]
            cn = len(neigh_sites)
            if cn in self.ops:
                for opidx, op in enumerate(self.ops[cn]):
                    if self.optypes[cn][opidx] == "wt":
                        opval = 1
                    else:
                        opval = \
                        op.get_order_parameters([struct[idx]] + neigh_sites, 0,
                                                indices_neighs=[i for i in
                                                                range(1, len(
                                                                    neigh_sites) + 1)])[
                            0]

                    opval = opval or 0  # handles None

                    # figure out the weight for this opval based on semicircle integration method
                    x1 = 1 - dist
                    x2 = 1 if dist_idx == len(dist_bins) - 1 else \
                        1 - dist_bins[dist_idx + 1]
                    weight = self._semicircle_integral(x2) - \
                             self._semicircle_integral(x1)

                    opval = opval * weight / total_weight

                    cn_fingerprint_array[cn].append(opval)

        # convert dict to list
        cn_fingerprint = []
        for cn in sorted(self.optypes):
            for op_idx, _ in enumerate(self.optypes[cn]):
                try:
                    cn_fingerprint.append(cn_fingerprint_array[cn][op_idx])
                except IndexError:  # no OP value computed
                    cn_fingerprint.append(0)

        return cn_fingerprint

    def feature_labels(self):
        labels = []
        for cn in sorted(self.optypes):
            for op in self.optypes[cn]:
                labels.append("{} CN_{}".format(op, cn))

        return labels

    def citations(self):
        return []

    def implementors(self):
        return ['Anubhav Jain', 'Nils E.R. Zimmermann']

    @staticmethod
    def _semicircle_integral(x, r=1):
        if r == x:
            return 0.25 * math.pi * r ** 2

        return 0.5 * ((x * math.sqrt(r ** 2 - x ** 2)) + (
                r ** 2 * math.atan(x / math.sqrt(r ** 2 - x ** 2))))


class VoronoiIndex(BaseFeaturizer):
    """
    The Voronoi indices n_i and the fractional Voronoi indices n_i/sum(n_i) that
    reflects the i-fold symmetry in the local sites.
    n_i denotes the number of the i-edged faces, and i is in the range of 3-10 here.
    e.g. for bcc lattice, the Voronoi indices are [0,6,0,8,0,0...]
         for fcc/hcp lattice, the Voronoi indices are [0,12,0,0,...]
         for icosahedra, the Voronoi indices are [0,0,12,0,...]
    """

    def __init__(self, cutoff=6.0):
        """
        Args:
            cutoff (float): cutoff distance in determining the potential
                neighbors for Voronoi tessellation analysis
        """
        self.cutoff = cutoff
        self.voronoi_analyzer = VoronoiAnalyzer(cutoff=self.cutoff)

    def featurize(self, struct, idx):
        """
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            list including Voronoi indices, sum of Voronoi indices, and
            fractional Voronoi indices
        """

        voro_index_result = []
        voro_index_list = self.voronoi_analyzer.analyze(struct, n=idx)
        for voro_index in voro_index_list:
            voro_index_result.append(voro_index)
        voro_index_sum = sum(voro_index_list)
        voro_index_result.append(voro_index_sum)

        voro_index_frac_list = voro_index_list / voro_index_sum
        for voro_index_frac in voro_index_frac_list:
            voro_index_result.append(voro_index_frac)

        return voro_index_result

    def feature_labels(self):
        labels = []
        for i in range(3, 11):
            labels.append('voro_index_%d' % i)
        labels.append('voro_index_sum')
        for i in range(3, 11):
            labels.append('voro_index_frac_%d' % i)
        return labels

    def citations(self):
        citation = ['@book{okabe1992spatial,  '
                    'title={Spatial tessellations}, '
                    'author={Okabe, Atsuyuki}, '
                    'year={1992}, '
                    'publisher={Wiley Online Library}}']
        return citation

    def implementors(self):
        return ['Qi Wang']


class EwaldSiteEnergy:
    """Compute site energy from Coulombic interactions

    User notes:
        - This class uses that `charges that are already-defined for the structure`.

        - Ewald summations can be expensive. If you evaluating every site in many
          large structures, run all of the sites for each structure at the same time.
          We cache the Ewald result for the structure that was run last, so looping
          over sites and then structures is faster than structures than sites.

    Features:
        ewald_site_energy - Energy for the site computed from Coulombic interactions"""

    def __init__(self, accuracy=None):
        """
        Args:
            accuracy (int): Accuracy of Ewald summation, number of decimal places
        """
        self.accuracy = accuracy

        # Variables used then caching the Ewald result
        self.__last_structure = None
        self.__last_ewald = None

    def featurize(self, strc, idx):
        """
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            ([float]) - Electrostatic energy of the site
        """

        # Check if the new input is the last
        #  Note: We use 'is' rather than structure comparisons for speed
        #
        #  TODO: Figure out if this implementation is thread-safe! I was debating adding
        #        Locks, but think we are OK
        if strc is self.__last_structure:
            ewald = self.__last_ewald
        else:
            self.__last_structure = strc
            ewald = EwaldSummation(strc, acc_factor=self.accuracy)
            self.__last_ewald = ewald
        return [ewald.get_site_energy(idx)]

    def feature_labels(self):
        return ["ewald_site_energy"]

    def implementors(self):
        return ["Logan Ward"]

    def citations(self):
        return ["@Article{Ewald1921,"
                "author = {Ewald, P. P.},"
                "doi = {10.1002/andp.19213690304},"
                "issn = {00033804},"
                "journal = {Annalen der Physik},"
                "number = {3},"
                "pages = {253--287},"
                "title = {{Die Berechnung optischer und elektrostatischer Gitterpotentiale}},"
                "url = {http://doi.wiley.com/10.1002/andp.19213690304},"
                "volume = {369},"
                "year = {1921}"
                "}"]


class ChemEnvSiteFingerprint(BaseFeaturizer):
    """
    Site fingerprint computed from pymatgen's ChemEnv package
    that provides resemblance percentages of a given site
    to ideal environments.
    Args:
        cetypes ([str]): chemical environments (CEs) to be
            considered.
        strategy (ChemenvStrategy): ChemEnv neighbor-finding strategy.
        geom_finder (LocalGeometryFinder): ChemEnv local geometry finder.
        max_csm (float): maximum continuous symmetry measure (CSM;
            default of 8 taken from chemenv). Note that any CSM
            larger than max_csm will be set to max_csm in order
            to avoid negative values (i.e., all features are
            constrained to be between 0 and 1).
    """

    @staticmethod
    def from_preset(preset):
        """
        Use a standard collection of CE types and
        choose your ChemEnv neighbor-finding strategy.
        Args:
            preset (str): preset types ("simple" or
                          "multi_weights").
        Returns:
            ChemEnvSiteFingerprint object from a preset.
        """
        cetypes = [
            'S:1', 'L:2', 'A:2', 'TL:3', 'TY:3', 'TS:3', 'T:4',
            'S:4', 'SY:4', 'SS:4', 'PP:5', 'S:5', 'T:5', 'O:6',
            'T:6', 'PP:6', 'PB:7', 'ST:7', 'ET:7', 'FO:7', 'C:8',
            'SA:8', 'SBT:8', 'TBT:8', 'DD:8', 'DDPN:8', 'HB:8',
            'BO_1:8', 'BO_2:8', 'BO_3:8', 'TC:9', 'TT_1:9',
            'TT_2:9', 'TT_3:9', 'HD:9', 'TI:9', 'SMA:9', 'SS:9',
            'TO_1:9', 'TO_2:9', 'TO_3:9', 'PP:10', 'PA:10',
            'SBSA:10', 'MI:10', 'S:10', 'H:10', 'BS_1:10',
            'BS_2:10', 'TBSA:10', 'PCPA:11', 'H:11', 'SH:11',
            'CO:11', 'DI:11', 'I:12', 'PBP:12', 'TT:12', 'C:12',
            'AC:12', 'SC:12', 'S:12', 'HP:12', 'HA:12', 'SH:13',
            'DD:20']
        lgf = LocalGeometryFinder()
        lgf.setup_parameters(
            centering_type='centroid',
            include_central_site_in_centroid=True,
            structure_refinement=lgf.STRUCTURE_REFINEMENT_NONE)
        if preset == "simple":
            return ChemEnvSiteFingerprint(
                cetypes,
                SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3),
                lgf)
        elif preset == "multi_weights":
            return ChemEnvSiteFingerprint(
                cetypes,
                MultiWeightsChemenvStrategy.stats_article_weights_parameters(),
                lgf)
        else:
            raise RuntimeError('unknown neighbor-finding strategy preset.')

    def __init__(self, cetypes, strategy, geom_finder, max_csm=8):
        self.cetypes = tuple(cetypes)
        self.strat = strategy
        self.lgf = geom_finder
        self.max_csm = max_csm

    def featurize(self, struct, idx):
        """
        Get ChemEnv fingerprint of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure struct.
        Returns:
            (numpy array): resemblance fraction of target site to ideal
                           local environments.
        """
        cevals = []
        self.lgf.setup_structure(structure=struct)
        se = self.lgf.compute_structure_environments()
        #        maximum_distance_factor=1.41)
        for ce in self.cetypes:
            try:
                tmp = se.get_csms(idx, ce)
                tmp = tmp[0]['symmetry_measure'] if len(tmp) != 0 \
                    else self.max_csm
                tmp = tmp if tmp < self.max_csm else self.max_csm
                cevals.append(1 - tmp / self.max_csm)
            except IndexError:
                cevals.append(0)
        return np.array(cevals)

    def feature_labels(self):
        return list(self.cetypes)

    def citations(self):
        return ['@article{waroquiers_chemmater_2017, '
                'title={Statistical analysis of coordination environments '
                'in oxides}, journal={Chemistry of Materials},'
                'author={Waroquiers, D. and Gonze, X. and Rignanese, G.-M.'
                'and Welker-Nieuwoudt, C. and Rosowski, F. and Goebel, M. '
                'and Schenk, S. and Degelmann, P. and Andre, R. '
                'and Glaum, R. and Hautier, G.}, year={2017}}']

    def implementors(self):
        return ['Nils E. R. Zimmermann']

class CoordinationNumber(BaseFeaturizer):
    """
    Coordination number (CN) computed using one of pymatgen's
    NearNeighbor classes for determination of near neighbors
    contributing to the CN.
    Args:
        nn (NearNeighbor): instance of one of pymatgen's NearNeighbor
                           classes.
    """

    @staticmethod
    def from_preset(preset):
        """
        Use one of the standard instances of a given NearNeighbor
        class.
        Args:
            preset (str): preset type ("VoronoiNN", "JMolNN",
                          "MiniumDistanceNN", "MinimumOKeeffeNN",
                          or "MinimumVIRENN").
        Returns:
            CoordinationNumber from a preset.
        """
        if preset == "VoronoiNN":
            return CoordinationNumber(VoronoiNN())
        elif preset == "JMolNN":
            return CoordinationNumber(JMolNN())
        elif preset == "MinimumDistanceNN":
            return CoordinationNumber(MinimumDistanceNN())
        elif preset == "MinimumOKeeffeNN":
            return CoordinationNumber(MinimumOKeeffeNN())
        elif preset == "MinimumVIRENN":
            return CoordinationNumber(MinimumVIRENN())
        else:
            raise RuntimeError('Unknown preset.')

    def __init__(self, nn, use_weights=False):
        self.nn = nn
        self.use_weights = use_weights

    def featurize(self, struct, idx):
        """
        Get coordintion number of site with given index in input
        structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure struct.
        Returns:
            (float): coordination number.
        """
        return [self.nn.get_cn(struct, idx, use_weights=self.use_weights)]

    def feature_labels(self):
        return ['CN_{}'.format(self.nn.__class__.__name__)]

    def citations(self):
        citations = []
        if self.nn.__class__.__name__ == 'VoronoiNN':
            citations.append('@article{voronoi_jreineangewmath_1908, title={'
                'Nouvelles applications des param\\`{e}tres continus \\`{a} la '
                'th\'{e}orie des formes quadratiques. Sur quelques '
                'propri\'{e}t\'{e}s des formes quadratiques positives'
                ' parfaites}, journal={Journal f\"ur die reine und angewandte '
                'Mathematik}, number={133}, pages={97-178}, year={1908}}')
            citations.append('@article{dirichlet_jreineangewmath_1850, title={'
                '\"{U}ber die Reduction der positiven quadratischen Formen '
                'mit drei unbestimmten ganzen Zahlen}, journal={Journal '
                'f\"ur die reine und angewandte Mathematik}, number={40}, '
                'pages={209-227}, doi={10.1515/crll.1850.40.209}, year={1850}}')
        if self.nn.__class__.__name__ == 'JMolNN':
            citations.append('@misc{jmol, title = {Jmol: an open-source Java '
                'viewer for chemical structures in 3D}, howpublished = {'
                '\\url{http://www.jmol.org/}}}')
        if self.nn.__class__.__name__ == 'MinimumOKeeffeNN':
            citations.append('@article{okeeffe_jamchemsoc_1991, title={Atom '
                'sizes and bond lengths in molecules and crystals}, journal='
                '{Journal of the American Chemical Society}, author={'
                'O\'Keeffe, M. and Brese, N. E.}, number={113}, pages={'
                '3226-3229}, doi={doi:10.1021/ja00009a002}, year={1991}}')
        if self.nn.__class__.__name__ == 'MinimumVIRENN':
            citations.append('@article{shannon_actacryst_1976, title={'
                'Revised effective ionic radii and systematic studies of '
                'interatomic distances in halides and chalcogenides}, '
                'journal={Acta Crystallographica}, author={Shannon, R. D.}, '
                'number={A32}, pages={751-767}, doi={'
                '10.1107/S0567739476001551}, year={1976}')
        if self.nn.__class__.__name__ in [
                'MinimumDistanceNN', 'MinimumOKeeffeNN', 'MinimumVIRENN']:
            citations.append('@article{zimmermann_frontmater_2017, '
                'title={Assessing local structure motifs using order '
                'parameters for motif recognition, interstitial '
                'identification, and diffusion path characterization}, '
                'journal={Frontiers in Materials}, author={Zimmermann, '
                'N. E. R. and Horton, M. K. and Jain, A. and Haranczyk, M.}, '
                'number={4:34}, doi={10.3389/fmats.2017.00034}, year={2017}}')
        return citations

    def implementors(self):
        return ['Nils E. R. Zimmermann']

class JahnTellerActiveSite(BaseFeaturizer):
    """
    Calculates a heuristic as to whether a site could be Jahn-Teller active
    or not. Requires an oxidation-state decorated structure.
    If spin configuration is not known, but magnetic moments are supplied,
    then spin configuration will be guessed from that. If magnetic moments
    are not supplied, the spin configuration will be chosen with the strongest
    J-T effect, so as to reduce false negatives.
    """

    spin_configs = {
        "oct":
            {  # key is number of d electrons
                0: {"high": {"e_g": 0, "t_2g": 0}, "default": "high"},
                1: {"high": {"e_g": 0, "t_2g": 1}, "default": "high"},  # weak J-T
                2: {"high": {"e_g": 0, "t_2g": 2}, "default": "high"},  # weak
                3: {"high": {"e_g": 0, "t_2g": 3}, "default": "high"},  # no J-T
                4: {"high": {"e_g": 1, "t_2g": 3},
                    "low": {"e_g": 0, "t_2g": 4}, "default": "high"},  # strong high, weak low
                5: {"high": {"e_g": 2, "t_2g": 3},
                    "low": {"e_g": 0, "t_2g": 5}, "default": "low"},  # no high, weak low
                6: {"high": {"e_g": 2, "t_2g": 4},
                    "low": {"e_g": 0, "t_2g": 6}, "default": "high"},  # weak high, no low
                7: {"high": {"e_g": 2, "t_2g": 5},
                    "low": {"e_g": 1, "t_2g": 6}, "default": "low"},  # weak high, strong low
                8: {"high": {"e_g": 2, "t_2g": 6}, "default": "high"},  # no
                9: {"high": {"e_g": 3, "t_2g": 6}, "default": "high"},  # strong
                10: {"high": {"e_g": 4, "t_2g": 6}, "default": "high"}
            },
        "tet":
            {  # no low spin observed experimentally in tetrahedral, all weak J-T
                0: {"high": {"e": 0, "t_2": 0}, "default": "high"},
                1: {"high": {"e": 1, "t_2": 0}, "default": "high"},
                2: {"high": {"e": 2, "t_2": 0}, "default": "high"},
                3: {"high": {"e": 2, "t_2": 1}, "default": "high"},
                4: {"high": {"e": 2, "t_2": 2}, "default": "high"},
                5: {"high": {"e": 2, "t_2": 3}, "default": "high"},
                6: {"high": {"e": 3, "t_2": 3}, "default": "high"},
                7: {"high": {"e": 4, "t_2": 3}, "default": "high"},
                8: {"high": {"e": 4, "t_2": 4}, "default": "high"},
                9: {"high": {"e": 4, "t_2": 5}, "default": "high"},
                10: {"high": {"e": 4, "t_2": 6}, "default": "high"}
            }
    }

    def featurize(self, structure, idx):
        """
        Feature is whether site is or could be Jahn-Teller active.

        Args:
            structure: oxi-state decorated Structure
            idx (int): index of site

        Returns (list): 'strong', 'weak', 'none' (not active) or 'unknown'

        """
        jahn_teller_site_analysis = self.jahn_teller_site_analysis(structure, idx)
        return [jahn_teller_site_analysis['magnitude']]

    def feature_labels(self):
        return ['Jahn-Teller active site']

    def citations(self):
        return []

    def implementors(self):
        return ['Matthew Horton']

    def jahn_teller_site_analysis(self, structure, idx, spin_state='default'):
        """
        Perform analysis of a given site as to whether that site is likely to
        be Jahn-Teller active.

        Args:
            structure: oxi-state decorated Structure
            idx (int): index of site
            spin_state (str): 'high', 'low' or 'default' ('default' will try
        to estimate spin config from magnetic moments if present or, if not,
        will choose spin config with strongest J-T effect)

        Returns (dict): summary of analysis for that site
        """

        site = structure[idx]

        if getattr(site.specie, 'oxi_state', None) is None:
            return {'magnitude': 'unknown',
                    'reason': 'Oxidation state of site is not specified.'}

        # get structural motif for site
        op = OPSiteFingerprint()
        op_site = op.featurize(structure, idx)

        tet_idx = op.feature_labels().index('tet CN_4')
        oct_idx = op.feature_labels().index('oct CN_6')

        # threshold for motif to be considered present
        # lower than you might expect since many J-T structures
        # are significantly distorted
        threshold = 0.45

        if op_site[tet_idx] > op_site[oct_idx] \
                and op_site[tet_idx] > threshold:
            motif = 'tet'
            motif_op = op_site[tet_idx]
        elif op_site[oct_idx] > threshold:
            motif = 'oct'
            motif_op = op_site[oct_idx]
        else:
            return {'magnitude': 'none',
                    'reason': 'Site is not in an octahedral or tetrahedral motif.'}

        if spin_state == 'default' and 'magmom' in site.properties:
            magmom = site.properties['magmom']
            spin_state = self._estimate_spin_state(site.specie, motif, magmom)

        magnitude = self._get_magnitude_of_effect_from_species(site.specie, spin_state, motif)

        if magnitude != 'none' and magnitude != 'unknown':


            ligands = get_neighbors_of_site_with_index(structure, idx)
            ligand_bond_lengths = [ligand.distance(structure[idx]) for ligand in ligands]
            ligands_species = list(set([str(ligand.specie) for ligand in ligands]))
            ligand_bond_length_spread = max(ligand_bond_lengths) - min(ligand_bond_lengths)

            def trim(f):
                # avoid displaying unreasonable precision, hurts readability
                return float("{:.4f}".format(f))

            # to be Jahn-Teller active, all ligands have to be the same
            if len(ligands_species) == 1:

                summary = {
                    'magnitude': magnitude,
                    'motif': motif,
                    'motif_order_parameter': trim(motif_op),  # estimate of confidence
                    'given_spin_state': spin_state,
                    'ligand': ligands_species[0],
                    'ligand_bond_lengths': [trim(length) for length in ligand_bond_lengths],
                    'ligand_bond_length_spread': trim(ligand_bond_length_spread)
                }
            else:
                summary =  {
                    'magnitude': 'Not active',
                    'reason': 'Site symmetry is broken by presence of different ligands.'
                }

        else:

            summary = {
                'magnitude': magnitude,
                'reason': 'Species on site is not a Jahn-Teller active species.'
            }

        return summary

    @staticmethod
    def _get_number_of_d_electrons(species):

        # taken from get_crystal_field_spin
        elec = species.full_electronic_structure
        if len(elec) < 4 or elec[-1][1] != "s" or elec[-2][1] != "d":
            raise AttributeError(
                "Invalid element {} for crystal field calculation.".format(species.symbol))
        nelectrons = int(elec[-1][2] + elec[-2][2] - species.oxi_state)
        if nelectrons < 0 or nelectrons > 10:
            raise AttributeError(
                "Invalid oxidation state {} for element {}".format(species.oxi_state,
                                                                   species.symbol))

        return nelectrons

    @staticmethod
    def _get_magnitude_of_effect_from_species(species, spin_state, motif):
        """


        Args:
            species: species string with oxidation state
            spin_state: "high" or "low" or "unknwown"
            motif: "oct" or "tet"

        Returns (str): magnitude of Jahn-Teller effect

        """

        sp = get_el_sp(species)

        # has to be Specie; we need to know the oxidation state
        if isinstance(sp, Specie) and sp.element.is_transition_metal:

            d_electrons = JahnTellerActiveSite._get_number_of_d_electrons(sp)
            spin_configs = JahnTellerActiveSite.spin_configs

            if motif in spin_configs:
                if spin_state not in ('high', 'low'):
                    spin_state = spin_configs[motif][d_electrons]['default']
                spin_config = spin_configs[motif][d_electrons][spin_state]
                print(spin_configs[motif][d_electrons])
                magnitude = JahnTellerActiveSite.get_magnitude_of_effect_from_spin_config(motif,
                                                                                          spin_config)
            else:
                magnitude = "none"
        else:
            magnitude = "unknown"

        return magnitude

    @staticmethod
    def get_magnitude_of_effect_from_spin_config(motif, spin_config):
        """
        Roughly, magnitude of Jahn-Teller distortion will be:
        * in octahedral environments, strong if e_g orbitals
        unevenly occupied but weak if t_2g orbitals unevenly
        occupied
        * in tetrahedral environments always weaker

        Args:
            motif: "oct" or "tet"
            spin_config: dict of 'e' (e_g) and 't' (t2_g)
        with number of electrons in each state

        Returns (str): "none", "weak" or "strong"

        """
        magnitude = "none"
        if motif == "oct":
            e_g = spin_config["e_g"]
            t_2g = spin_config["t_2g"]
            if (e_g % 2 != 0) or (t_2g % 3 != 0):
                magnitude = "weak"
                if e_g % 2 == 1:
                    magnitude = "strong"
        elif motif == "tet":
            e = spin_config["e"]
            t_2 = spin_config["t_2"]
            if (e % 3 != 0) or (t_2 % 2 != 0):
                magnitude = "weak"
        return magnitude

    @staticmethod
    def _estimate_spin_state(species, motif, known_magmom):
        """
        Simple heuristic to estimate spin state. If magnetic moment
        is sufficiently close to that predicted for a given spin state,
        we assign it that state. If we only have data for one spin
        state then that's the one we use (e.g. we assume all tetrahedral
        complexes are high-spin, since this is typically the case).

        Args:
            species: str or Species
            motif: "oct" or "tet"
            known_magmom: known magnetic moment

        Returns (str): "high", "low" or "unknown"

        """
        mu_so_high = JahnTellerActiveSite._mu_so(species, motif=motif, spin_state="high")
        mu_so_low = JahnTellerActiveSite._mu_so(species, motif=motif, spin_state="low")
        if mu_so_high == mu_so_low:
            return "undefined"  # undefined or only one spin state possible
        elif mu_so_high is None:
            return "low"
        elif mu_so_low is None:
            return "high"
        else:
            diff = mu_so_high-mu_so_low
            # WARNING! this heuristic has not been robustly tested or benchmarked
            # using 'diff*0.25' as arbitrary measure, if known magmom is
            # too far away from expected value, we don't try to classify it
            if known_magmom > mu_so_high or abs(mu_so_high-known_magmom) < diff*0.25:
                return "high"
            elif known_magmom < mu_so_low or abs(mu_so_low-known_magmom) < diff*0.25:
                return "low"
            else:
                return "unknown"

    @staticmethod
    def _mu_so(species, motif, spin_state):
        """
        Calculates the spin-only magnetic moment for a
        given species. Only supports transition metals.

        Args:
            species: str or Species
            motif: "oct" or "tet"
            spin_state: "high" or "low"

        Returns: spin-only magnetic moment in Bohr magnetons

        """
        try:
            sp = get_el_sp(species)
            n = sp.get_crystal_field_spin(coordination=motif, spin_config=spin_state)
            # calculation spin-only magnetic moment for this number of unpaired spins
            return np.sqrt(n(n+2))
        except:
            return None
