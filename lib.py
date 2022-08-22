import numpy as np
import pandas as pd
import yaml

BIGNEG = -1e32
MINVAL, MAXVAL = -2, 7

# TOP assay yields of representative n:2 FT and ECF precursors reported in the literature:
# n:2 FT precursors: 4:2 FTS, 5:3 FTCA, 6:2 FTA, 6:2 FTAB, 6:2 diPAP, 8:2 diPAP, 7:3 FTCA, 8:2 FTS, 10:2 FTS
# ECF precursors: FBSA, FHxSA, PFHxSAm, PFHxSAmS, FOSA, MeFOSA, EtFOSA, FOSAA, N-MeFOSAA, N-EtFOSAA,
# PFOAB, PFOSB, PFOANA, PFOSNO, PFOSAmS, PFOSAm, FDSA

# Average of n:2 FTs reported in Martin et al., 2019, Talanta; Houtz & Sedlak, 2012, ES&T;
# Simonnet-Laprade et al., 2019, ESPI; Gockener et al., 2020, JAFC; Wang et al., 2021, ES&T;
# and internal Heidi TOP fish analysis (20210222, 20210303)
# x_ft = [n, n-1, n-2, n-3, n-4, n-5, n-6, n-7]
x_ft = [0.0568, 0.2290, 0.3469, 0.2094, 0.1336, 0.0811, 0.0200, 0.0100]

# Standard deviation of average n:2 FTs reported in Martin et al., 2019, Talanta; Houtz & Sedlak, 2012, ES&T;
# Simonnet-Laprade et al., 2019, ESPI; Gockener et al., 2020, JAFC; Wang et al., 2021, ES&T;
# and internal Heidi TOP fish analysis (20210222, 20210303)
err_ft = [0.0586, 0.1100, 0.0986, 0.0488, 0.0330, 0.0468, 0.0212, 0.0141]

# Average of ECF precursors reported in Martin et al., 2019, Talanta; Houtz & Sedlak, 2012, ES&T;
# Simonnet-Laprade et al., 2019, ESPI; Gockener et al., 2020, JAFC; Wang et al., 2021, ES&T;
# Janda et al., 2019, ESPI; and internal Bridger (PFHxSAm/S) and Heidi TOP fish analyses (20210222, 20210303, 20210323)
# x_ecf = [n, n-1, n-2, n-3, n-4, n-5, n-6, n-7]
x_ecf = [0.0043, 0.8620, 0.0262, 0.0260, 0.0061, 0.0011, 0.0900, 0]

# Standard deviation of average ECF precursors reported in Martin et al., 2019, Talanta; Houtz & Sedlak, 2012, ES&T;
# Simonnet-Laprade et al., 2019, ESPI; Gockener et al., 2020, JAFC; Wang et al., 2021, ES&T;
# Janda et al., 2019, ESPI; and internal Bridger (PFHxSAm/S) and Heidi TOP fish analyses (20210222, 20210303, 20210323)
err_ecf = [0.0126, 0.1600, 0.0338, 0.0645, 0.0226, 0.0058, 0, 0]

# Prior information: the ratio of PFOS to ECF precursors in ECF AFFF
# From Houtz et al. 2013 Table S5 and S6
ECFcomp = pd.read_csv('data/3M_AFFF_Compositions.csv')
chains = [f'C{x}' for x in range(4, 11)]
fmeans = np.array([np.mean(ECFcomp[c]) for c in chains])
fstds = np.array([np.std(ECFcomp[c]) for c in chains])

precursor_order = ['4:2 FT',  '5:3 FT',  '6:2 FT',  '7:3 FT',  '8:2 FT',  '9:3 FT',  '10:2 FT',
                   'C4 ECF',  'C5 ECF',  'C6 ECF',  'C7 ECF',  'C8 ECF',  'C9 ECF',  'C10 ECF']
terminal_order = ['C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']


def makeA(x_ft, err_ft, x_ecf, err_ecf):
    """ TOP assay PFCA yield matrix.

    Takes yields reported Martin et al., 2019, Talanta; Houtz & Sedlak, 2012, ES&T; Simonnet-Laprade et al., 2019, ESPI;
    Gockener et al., 2020, JAFC; Wang et al., 2021, ES&T; Janda et al., 2019, ESPI; and internal Bridger (PFHxSAm/S)
    and Heidi TOP fish analyses (20210222, 20210303, 20210323)
    and returns 8x14 matrices representing A and U from matrix eq (A±U)x=b """
    #       0       1       2       3       4       5       6       7       8       9       10      11      12      13
    #    4:2 FT  5:3 FT  6:2 FT  7:3 FT  8:2 FT  9:3 FT  10:2 FT  C4 ECF  C5 ECF  C6 ECF  C7 ECF  C8 ECF  C9 ECF  C10 ECF
   # A = [x1_FT, x2_FT,  x3_FT,  x4_FT,  x5_FT,  x6_FT,  x7_FT,   x1_ECF, x2_ECF, x3_ECF, x4_ECF, x5_ECF, x6_ECF, x7_ECF] #to C3  (PFBA)
   #     [x0_FT, x1_FT,  x2_FT,  x3_FT,  x4_FT,  x5_FT,  x6_FT,   x0_ECF, x1_ECF, x2_ECF, x3_ECF, x4_ECF, x5_ECF, x6_ECF] #to C4  (PFPeA)
   #     [0    , x0_FT,  x1_FT,  x2_FT,  x3_FT,  x4_FT,  x5_FT,   0     , x0_ECF, x1_ECF, x2_ECF, x3_ECF, x4_ECF, x5_ECF] #to C5  (PFHxA)
   #     [0    , 0    ,  x0_FT,  x1_FT,  x2_FT,  x3_FT,  x4_FT,   0     , 0     , x0_ECF, x1_ECF, x2_ECF, x3_ECF, x4_ECF] #to C6  (PFHpA)
   #     [0    , 0    ,  0    ,  x0_FT,  x1_FT,  x2_FT,  x3_FT,   0     , 0     , 0     , x0_ECF, x1_ECF, x2_ECF, x3_ECF] #to C7  (PFOA)
   #     [0    , 0    ,  0    ,  0    ,  x0_FT,  x1_FT,  x2_FT,   0     , 0     , 0     , 0     , x0_ECF, x1_ECF, x2_ECF] #to C8  (PFNA)
   #     [0    , 0    ,  0    ,  0    ,  0    ,  x0_FT,  x1_FT,   0     , 0     , 0     , 0     , 0     , x0_ECF, x1_ECF] #to C9  (PFDA)
   #     [0    , 0    ,  0    ,  0    ,  0    ,  0    ,  x0_FT,   0     , 0     , 0     , 0     , 0     , 0     , x0_ECF] #to C10 (PFUnDA)

    # U has the same structure as A, but is populated by
    # standard deviation (err) instead of mean

    # Construct A and U separately
    A = np.zeros((8, 14))
    U = np.zeros_like(A)

    # FT
    A[1, 0] = A[2, 1] = A[3, 2] = A[4, 3] = A[5, 4] = A[6, 5] = A[7, 6] = x_ft[0]
    U[1, 0] = U[2, 1] = U[3, 2] = U[4, 3] = U[5, 4] = U[6, 5] = U[7, 6] = err_ft[0]
    A[0, 0] = A[1, 1] = A[2, 2] = A[3, 3] = A[4, 4] = A[5, 5] = A[6, 6] = x_ft[1]
    U[0, 0] = U[1, 1] = U[2, 2] = U[3, 3] = U[4, 4] = U[5, 5] = U[6, 6] = err_ft[1]
    A[0, 1] = A[1, 2] = A[2, 3] = A[3, 4] = A[4, 5] = A[5, 6] = x_ft[2]
    U[0, 1] = U[1, 2] = U[2, 3] = U[3, 4] = U[4, 5] = U[5, 6] = err_ft[2]
    A[0, 2] = A[1, 3] = A[2, 4] = A[3, 5] = A[4, 6] = x_ft[3]
    U[0, 2] = U[1, 3] = U[2, 4] = U[3, 5] = U[4, 6] = err_ft[3]
    A[0, 3] = A[1, 4] = A[2, 5] = A[3, 6] = x_ft[4]
    U[0, 3] = U[1, 4] = U[2, 5] = U[3, 6] = err_ft[4]
    A[0, 4] = A[1, 5] = A[2, 6] = x_ft[5]
    U[0, 4] = U[1, 5] = U[2, 6] = err_ft[5]
    A[0, 5] = A[1, 6] = x_ft[6]
    U[0, 5] = U[1, 6] = err_ft[6]
    A[0, 6] = x_ft[7]
    U[0, 6] = err_ft[7]

    # ECF
    A[1, 7] = A[2, 8] = A[3, 9] = A[4, 10] = A[5,
                                               11] = A[6, 12] = A[7, 13] = x_ecf[0]
    U[1, 7] = U[2, 8] = U[3, 9] = U[4, 10] = U[5,
                                               11] = U[6, 12] = U[7, 13] = err_ecf[0]
    A[0, 7] = A[1, 8] = A[2, 9] = A[3, 10] = A[4,
                                               11] = A[5, 12] = A[6, 13] = x_ecf[1]
    U[0, 7] = U[1, 8] = U[2, 9] = U[3, 10] = U[4,
                                               11] = U[5, 12] = U[6, 13] = err_ecf[1]
    A[0, 8] = A[1, 9] = A[2, 10] = A[3, 11] = A[4, 12] = A[5, 13] = x_ecf[2]
    U[0, 8] = U[1, 9] = U[2, 10] = U[3, 11] = U[4, 12] = U[5, 13] = err_ecf[2]
    A[0, 9] = A[1, 10] = A[2, 11] = A[3, 12] = A[4, 13] = x_ecf[3]
    U[0, 9] = U[1, 10] = U[2, 11] = U[3, 12] = U[4, 13] = err_ecf[3]
    A[0, 10] = A[1, 11] = A[2, 12] = A[3, 13] = x_ecf[4]
    U[0, 10] = U[1, 11] = U[2, 12] = U[3, 13] = err_ecf[4]
    A[0, 11] = A[1, 12] = A[2, 13] = x_ecf[5]
    U[0, 11] = U[1, 12] = U[2, 13] = err_ecf[5]
    A[0, 12] = A[1, 13] = x_ecf[6]
    U[0, 12] = U[1, 13] = err_ecf[6]
    A[0, 13] = x_ecf[7]
    U[0, 13] = err_ecf[7]

    return (A, U)


all_targeted = ['42FTmeas', '53FTmeas', '62FTmeas', '73FTmeas', '82FTmeas',
                '93FTmeas', '102FTmeas', 'C4ECFmeas', 'C5ECFmeas', 'C6ECFmeas',
                'C7ECFmeas', 'C8ECFmeas', 'C9ECFmeas', 'C10ECFmeas']

all_chains = ['C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']

all_keys = all_chains + ['PFOS', 'CXerr',
                         'C3pre', 'C4pre', 'C5pre', 'C6pre', 'C7pre', 'C8pre',
                         'C9pre', 'C10pre', 'PFOSpre',
                         'C3post', 'C4post', 'C5post', 'C6post', 'C7post', 'C8post',
                         'C9post', 'C10post', 'PFOSpost',
                         'C3MDL', 'C4MDL', 'C5MDL', 'C6MDL', 'C7MDL', 'C8MDL',
                         'C9MDL', 'C10MDL', 'PFOSMDL',
                         'C3err', 'C4err', 'C5err', 'C6err', 'C7err', 'C8err',
                         'C9err', 'C10err', 'PFOSerr'] + all_targeted


class Config(object):
    """
    Configuration options and parsed measurement data.
    Build based on rows of provided csv datafile, and the config yaml
    file pointed to therein. 
    """

    def __init__(self, dfrow):

        self.b = None
        self.bpre = None
        self.bpost = None
        self.pfos = None
        self.mdls = None
        self.errs = None
        self.targetprec = None
        self.whats_measured = None
        self.configfile = None
        self.jeffreys_variance = None

        # Get info from csv row
        self.from_row(dfrow)

        # Get info from config file
        ym = yaml.safe_load(open(self.configfile, 'r'))
        input_precursors = ym['possible_precursors']
        prior_name = ym['prior_name']
        self.possible_precursors = input_precursors
        self.prior_name = prior_name

        self.jeffreys_variance = ym.get('jeffreys_variance', None)

        x_ft_in = ym['x_ft']
        err_ft_in = ym['err_ft']
        x_ecf_in = ym['x_ecf']
        err_ecf_in = ym['err_ecf']
        A, U = makeA(x_ft_in, err_ft_in, x_ecf_in, err_ecf_in)
        self.full_model = A
        self.full_uncertainty = U
        self.model = None
        self.uncertainty = None
        self.compmeans = None
        self.compstds = None

        self.setup_model()

    def __str__(self):
        """Give some configuration information for print."""
        a = [f'Configuration source: {self.configfile}',
             f'Using prior: {self.prior_name}.',
             f'Found measured chains: {self.whats_measured}',
             f'Precursors to infer: {self.possible_precursors}'
             ]
        if self.BLANKETERROR:
            a.append('Using CX blanket measurement error.')
        else:
            a.append('Using chain length-specific measurement error.')
        if self.PREPOST:
            a.append('Using both pre- and post-oxidation measurements.')
        return '\n'.join(a)

    def from_row(self, dfrow):
        """Parse csv row for measurement and meta data."""
        measured_values = {}
        whats_measured = []
        targeted_precursors = []
        PREPOST = False
        BLANKETERROR = False
        present_keys = dfrow.keys()
        for key in present_keys:
            if key in all_keys:
                val = dfrow[key]
                if not np.isnan(val):
                    if (not PREPOST) and key.endswith('pre'):
                        PREPOST = True
                    measured_values[key] = val
                    if key in ['CXerr']:
                        BLANKETERROR = True

        for targ in all_targeted:
            try:
                val = dfrow[targ]
            except KeyError:
                val = np.nan
            if not np.isnan(val):
                measured_values[targ] = val
                targeted_precursors.append(targ)

        for chain in all_chains:
            if (chain+'MDL' in measured_values) and ((chain+'err' in measured_values) or BLANKETERROR):
                pass
            else:
                continue
            if PREPOST:
                if (chain+'pre' in measured_values) and (chain+'post' in measured_values):
                    pass
                else:
                    continue
            else:
                if chain in measured_values:
                    pass
                else:
                    continue
            whats_measured.append(chain)

        self.whats_measured = whats_measured

        try:
            cfg = 'config/'+dfrow['config']
        except KeyError:
            raise KeyError

        if PREPOST:
            b = None
            bpre = []
            bpost = []
            for chain in whats_measured:
                bpre.append(dfrow[chain+'pre'])
                bpost.append(dfrow[chain+'post'])
            bpre = np.array(bpre)
            bpost = np.array(bpost)
        else:
            bpre = None
            bpost = None
            b = []
            for chain in whats_measured:
                b.append(dfrow[chain])
            b = np.array(b)
        mdls = []
        berr = []
        for chain in whats_measured:
            mdls.append(dfrow[chain+'MDL'])
            if BLANKETERROR:
                berr.append(dfrow['CXerr'])
            else:
                berr.append(dfrow[chain+'err'])

        PFOS = (dfrow['PFOS'], dfrow['PFOSMDL'])

        self.b = b
        self.bpre = bpre
        self.bpost = bpost
        self.mdls = np.array(mdls)
        self.berr = np.array(berr)
        self.pfos = PFOS
        self.configfile = cfg
        self.targetprec = [measured_values[targ]
                           for targ in targeted_precursors]  # measured values
        self.targeted_precursors = targeted_precursors  # names
        self.BLANKETERROR = BLANKETERROR
        self.PREPOST = PREPOST

    def _subset_model(self):
        """Choose subset of full precursors->terminal products model.
        Based on config-provided precursors to look for and what is
        actually measured according to the datafile."""
        choosep = []
        for p in self.possible_precursors:
            choosep.append(precursor_order.index(p))
        cp = np.array(sorted(choosep))
        choosec = []
        for c in self.whats_measured:
            choosec.append(terminal_order.index(c))
        cc = np.array(sorted(choosec))
        self.model = self.full_model[:, cp][cc, :]
        self.uncertainty = self.full_uncertainty[:, cp][cc, :]

    def _set_ecf_ft_indices(self):
        """Set indices of ECF and FT precursors for some priors."""
        ecf_indices = []
        ft_indices = []
        compmeans = []
        compstds = []
        for i, p in enumerate(self.possible_precursors):
            if 'ECF' in p:
                ecf_indices.append(i)
                chain = p[:-4]
                compmeans.append(fmeans[chains.index(chain)])
                std = fstds[chains.index(chain)]
                if std < 1e-5:
                    std = 1
                compstds.append(std)
            if 'FT' in p:
                ft_indices.append(i)
        self.ecf_indices = np.array(ecf_indices)
        self.ft_indices = np.array(ft_indices)
        self.compmeans = np.array(compmeans)
        self.compstds = np.array(compstds)

    def _set_targeted_indices(self):
        """Set indices of targeted precursor measurements for some priors."""
        calculated_precursors = [p.replace(':', '').replace(' ', '')
                                 for p in self.possible_precursors]
        order = []
        for targ in self.targeted_precursors:
            order.append(calculated_precursors.index(targ[:-4]))
        self.targeted_indices = order

    def setup_model(self, ):
        """Subset and index full model for particular case."""
        self._subset_model()
        self._set_ecf_ft_indices()
        self._set_targeted_indices()
