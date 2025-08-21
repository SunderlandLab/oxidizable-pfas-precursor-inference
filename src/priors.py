import numpy as np
from abc import ABC
import pandas as pd

BIGNEG = -1e32
MINVAL, MAXVAL = -6, 4

ECFcomp = pd.read_csv('data/3M_AFFF_Compositions.csv')
# chains = [f'C{x}' for x in range(4, 11)]
# fmeans = np.array([np.mean(ECFcomp[c]) for c in chains])
# fstds = np.array([np.std(ECFcomp[c]) for c in chains])

class Prior(ABC):
    ecf_indices: np.ndarray
    ft_indices: np.ndarray
    PFOS: float
    PFOS_MDL: float
    measured_sum: float
    composition_means: np.ndarray
    composition_stds: np.ndarray
    jeffreys_variance: float
    targeted_concentrations: np.ndarray
    targeted_indices: np.ndarray

    def __init__(self, precursors, measurements, jeffreys_variance=5, composition_means=None, composition_stds=None):
        self.ecf_indices = np.array([i for i, prec in enumerate(precursors) if prec.endswith("ECF")])
        self.ft_indices = np.array([i for i, prec in enumerate(precursors) if prec.endswith("FT")])
        PFOS_measurement = measurements.PFOS
        self.PFOS = PFOS_measurement.value
        self.PFOS_MDL = PFOS_measurement.MDL
        self.measured_sum = sum([m.value for name,m in measurements.PFCAs.items()])
        if composition_means is None:
            self.composition_means = np.array([np.mean(ECFcomp[c.replace(' ECF','')]) for c in precursors if c.endswith("ECF")])
        else:
            self.composition_means = np.array(composition_means)
        if composition_stds is None:
            self.composition_stds = np.array([np.std(ECFcomp[c.replace(' ECF','')]) for c in precursors if c.endswith("ECF")])
        else:
            self.composition_stds = np.array(composition_stds)
        self.jeffreys_variance = jeffreys_variance
        targ, targ_inds = [], []
        for i, prec in enumerate(precursors):
            t = measurements.targeted_precursors.get(prec, None)
            if t is not None:
                targ.append(t)
                targ_inds.append(i)
        self.targeted_concentrations = np.array(targ)
        self.targeted_indices = np.array(targ_inds)

# include upper/lower bounds in prior

class AFFFPrior(Prior):
    def __call__(self, params: np.ndarray) -> float:
        logprob = 0

        emin, emax = 0, 2
        PFOS, MDL = self.PFOS, self.PFOS_MDL

        x_p = 10**params[:-1]
        e_p = params[-1]
        ecf = np.sum(x_p[self.ecf_indices])  # sum only the ECF precursors

        # the sum of ECF precursors should fall inside of lower and upper bounds
        # of the ratio of TOP assay precursors (corrected for their PFCA yield
        # [88±12%]) to PFOS reported in 3M AFFF in Houtz et al. Tables S5 and S6
        lowratio = 0.84
        highratio = 2.73
        if PFOS < MDL:
            PFOS = MDL / np.sqrt(2)
            lowratio = 0
        pmin, pmax = PFOS * lowratio, PFOS * highratio

        if pmin < ecf < pmax:
            logprob = 0
        else:
            logprob = BIGNEG

        if emin < e_p < emax:
            logprob += 0
        else:
            logprob += BIGNEG


        # Evaluate the composition of ECF precursor proposal against their
        # composition reported in 3M AFFF in Houtz et al. Table S6 assuming that
        # the oxidation yields of ECF precursors do not depend on their
        # perfluorinated chain length (n)
        ecf_comp = x_p[self.ecf_indices] / ecf
        logprob += -(np.sum(np.abs((ecf_comp - self.composition_means) / (2 * self.composition_stds))**2))

        for i, xi in enumerate(params[:-1]):
            if xi < MINVAL:
                # don't let it waste time wandering arbitrarily low
                logprob += BIGNEG
            if xi > MAXVAL:  # or high
                logprob += BIGNEG

        return logprob

class AFFFJeffreysPrior(Prior):
    def __call__(self, params: np.ndarray) -> float:
        logprob = 0

        emin, emax = 0, 2
        PFOS, MDL = self.PFOS, self.PFOS_MDL

        x_p = 10**params[:-1]
        e_p = params[-1]
        ecf = np.sum(x_p[self.ecf_indices])  # sum only the ECF precursors

        var = self.jeffreys_variance
        if var is None:
            raise ValueError('Please define jeffreys_variance in yaml config')

        meassum = self.measured_sum

        jeffreys_min = np.log10(meassum) - var

        # the sum of ECF precursors should fall inside of lower and upper bounds
        # of the ratio of TOP assay precursors (corrected for their PFCA yield
        # [88±12%]) to PFOS reported in 3M AFFF in Houtz et al. Tables S5 and S6
        lowratio = 0.84
        highratio = 2.73
        if PFOS < MDL:
            PFOS = MDL / np.sqrt(2)
            lowratio = 0
        pmin, pmax = PFOS * lowratio, PFOS * highratio

        if pmin < ecf < pmax:
            logprob = 0
        else:
            logprob = BIGNEG

        if emin < e_p < emax:
            logprob += 0
        else:
            logprob += BIGNEG


        # Evaluate the composition of ECF precursor proposal against their
        # composition reported in 3M AFFF in Houtz et al. Table S6 assuming that
        # the oxidation yields of ECF precursors do not depend on their
        # perfluorinated chain length (n)
        ecf_comp = x_p[self.ecf_indices] / ecf
        logprob += -(np.sum(np.abs((ecf_comp - self.composition_means) / (2 * self.composition_stds))**2))

        for i, xi in enumerate(params[:-1]):
            if xi < jeffreys_min:
                # don't let it waste time wandering arbitrarily low
                logprob += BIGNEG
            if xi > MAXVAL:  # or high
                logprob += BIGNEG
        return logprob

class UnknownJeffreysPrior(Prior):
    def __call__(self, params: np.ndarray):
        logprob = 0

        targetprec = self.targeted_concentrations
        targeted_indices = self.targeted_indices

        var = self.jeffreys_variance
        if var is None:
            raise ValueError('Please define jeffreys_variance in yaml config')

        meassum = self.measured_sum

        jeffreys_min = np.log10(meassum) - var

        x_p = 10**params[:-1]
        totp = np.sum(x_p)

        # Prevent inference from infering solutions with more than 2x the
        # measured mass
        if totp/meassum <= 0.7:
            logprob += BIGNEG
        elif totp/meassum >= 2:
            logprob += BIGNEG

        for i, xi in enumerate(params[:-1]):
            xi_lin = 10**xi
            if xi < jeffreys_min:
                # don't let it waste time wandering arbitrarily low
                logprob += BIGNEG
            # if (xi_lin/meassum) >= 10:  # or high
                # logprob += BIGNEG
        if params[-1] < -2:
            logprob += BIGNEG
        if params[-1] > 1:
            logprob += BIGNEG

        # make sure the targeted measurements line up with the right parameters
        for i, tp in zip(targeted_indices, targetprec):
            xi = x_p[i]
            if xi < tp.value*0.7:
                # Prevent solutions where infered concentration < targeted
                # precursor concentrations
                logprob += BIGNEG
        return logprob

class BiotaPrior(Prior):
    def __call__(self, params: np.ndarray):
        logprob = 0

        targetprec = self.targeted_concentrations
        targeted_indices = self.targeted_indices

        var = self.jeffreys_variance
        if var is None:
            raise ValueError('Please define jeffreys_variance in yaml config')

        meassum = self.measured_sum

        jeffreys_min = np.log10(meassum) - var

        x_p = 10**params[:-1]
        totp = np.sum(x_p)

        # Prevent inference from infering solutions with more than 10x the
        # measured mass
        if totp/meassum <= 0.7:
            logprob += BIGNEG
        elif totp/meassum >= 2:
            logprob += BIGNEG

        for i, xi in enumerate(params[:-1]):
            xi_lin = 10**xi
            if xi < jeffreys_min:
                # don't let it waste time wandering arbitrarily low
                logprob += BIGNEG
            if (xi_lin/meassum) >= 2:  # or high
                logprob += BIGNEG
        if params[-1] < MINVAL:
            logprob += BIGNEG
        if params[-1] > 2:
            logprob += BIGNEG

        # make sure the targeted measurements line up with the right parameters
        for i, tp in zip(targeted_indices, targetprec):
            xi = x_p[i]
            if xi < tp*0.7:
                # Prevent solutions where infered concentration < targeted
                # precursor concentrations
                logprob += BIGNEG
        return logprob

class UnknownPrior(Prior):
    def __call__(self, params: np.ndarray):
        logprob = 0

        targetprec = self.targeted_concentrations
        targeted_indices = self.targeted_indices


        meassum = self.measured_sum

        x_p = 10**params[:-1]
        totp = np.sum(x_p)

        # Prevent inference from infering solutions with more than 10x the
        # measured mass
        if totp/meassum <= 0.7:
            logprob += BIGNEG
        elif totp/meassum >= 2:
            logprob += BIGNEG

        for i, xi in enumerate(params[:-1]):
            xi_lin = 10**xi
            if xi < MINVAL:
                # don't let it waste time wandering arbitrarily low
                logprob += BIGNEG
            if (xi_lin/meassum) >= 2:  # or high
                logprob += BIGNEG
        if params[-1] < MINVAL:
            logprob += BIGNEG
        if params[-1] > MAXVAL:
            logprob += BIGNEG

        # make sure the targeted measurements line up with the right parameters
        for i, tp in zip(targeted_indices, targetprec):
            xi = x_p[i]
            if xi < tp*0.7:
                # Prevent solutions where infered concentration < targeted
                # precursor concentrations
                logprob += BIGNEG
        return logprob

class AFFFImpactedPrior(Prior):
    def __call__(self, params):

        logprob = 0
        PFOS, MDL = self.PFOS, self.PFOS_MDL

        emin, emax = 0, 2

        x_p = 10**params[:-1]
        totp = np.sum(x_p)
        meassum = self.measured_sum

        x_p = 10**params[:-1]
        e_p = params[-1]
        ecf = np.sum(x_p[self.ecf_indices])  # sum only the ECF precursors

        # the sum of ECF precursors should not exceed the upper bound
        # of the ratio of TOP assay precursors (corrected for their PFCA yield
        # [88±12%]) to PFOS reported in 3M AFFF in Houtz et al. Tables S5 and S6
        lowratio = 0.84
        highratio = 2.73
        if PFOS < MDL:
            PFOS = MDL / np.sqrt(2)
            lowratio = 0
        pmin, pmax = PFOS * lowratio, PFOS * highratio

        if pmin < ecf < pmax:
            logprob += 0
        else:
            logprob += BIGNEG

        if emin < e_p < emax:
            logprob += 0
        else:
            logprob += BIGNEG

        # Prevent inference from infering solutions with more than 10x the
        # measured mass (i.e. we don't expect a recovery ≤ 10%)
        if totp/meassum <= 0.7:
            logprob += BIGNEG
        elif totp/meassum >= 2:
            logprob += BIGNEG
        else:
            logprob += 0

        # Evaluate the composition of ECF precursor proposal against their
        # composition reported in 3M AFFF in Houtz et al. Table S6 assuming that
        # the oxidation yields of ECF precursors do not depend on their
        # perfluorinated chain length (n)
        ecf_comp = x_p[self.ecf_indices] / ecf
        logprob += - \
            (np.sum(np.abs((ecf_comp - self.composition_means) / (self.composition_stds))**2))

        for i, xi in enumerate(params[:-1]):
            if xi < MINVAL:
                # don't let it waste time wandering arbitrarily low
                logprob += BIGNEG
            if xi > np.log10(meassum):  # or high
                logprob += BIGNEG
        if e_p < MINVAL:
            logprob += BIGNEG

        return logprob

class AFFFImpactedPriorArchival(Prior):
    def __call__(self, params):

        logprob = 0
        PFOS, MDL = self.PFOS, self.PFOS_MDL

        # emin, emax = 0.6, 2.0

        x_p = 10**params[:-1]
        totp = np.sum(x_p)
        meassum = self.measured_sum

        x_p = 10**params[:-1]
        e_p = params[-1]
        ecf = np.sum(x_p[self.ecf_indices])  # sum only the ECF precursors

        # the sum of ECF precursors should not exceed the upper bound
        # of the ratio of TOP assay precursors (corrected for their PFCA yield
        # [88±12%]) to PFOS reported in 3M AFFF in Houtz et al. Tables S5 and S6
        lowratio = 0.1
        highratio = 10
        if PFOS < MDL:
            PFOS = MDL / np.sqrt(2)
            lowratio = 0
        pmin, pmax = PFOS * lowratio, PFOS * highratio

        if pmin < ecf < pmax:
            logprob += 0
        else:
            logprob += BIGNEG

        # if emin < e_p < emax:
        if e_p > 0.6:
            logprob += -np.exp(abs(e_p - 0.6))
        else:
            logprob += BIGNEG

        # Prevent inference from infering solutions with more than 10x the
        # measured mass (i.e. we don't expect a recovery ≤ 10%)
        if totp/meassum <= 0.1:
            logprob += BIGNEG
        elif totp/meassum >= 10:
            logprob += BIGNEG
        else:
            logprob += 0

        # Evaluate the composition of ECF precursor proposal against their
        # composition reported in 3M AFFF in Houtz et al. Table S6 assuming that
        # the oxidation yields of ECF precursors do not depend on their
        # perfluorinated chain length (n)
        ecf_comp = x_p[self.ecf_indices] / ecf
        logprob += - \
            (np.sum(np.abs((ecf_comp - self.composition_means) / (self.composition_stds))**2))

        for i, xi in enumerate(params[:-1]):
            if xi < MINVAL:
                # don't let it waste time wandering arbitrarily low
                logprob += BIGNEG
            if xi > np.log10(meassum):  # or high
                logprob += BIGNEG
        if e_p < MINVAL:
            logprob += BIGNEG

        return logprob

class AFFFImpactedJeffreysPrior(Prior):
    def __call__(self, params):

        logprob = 0
        PFOS, MDL = self.PFOS, self.PFOS_MDL

        emin, emax = 0, 2

        x_p = 10**params[:-1]
        totp = np.sum(x_p)
        meassum = self.measured_sum

        var = self.jeffreys_variance
        if var is None:
            raise ValueError('Please define jeffreys_variance in yaml config')
        jeffreys_min = np.log10(meassum) - var

        x_p = 10**params[:-1]
        e_p = params[-1]
        ecf = np.sum(x_p[self.ecf_indices])  # sum only the ECF precursors

        # the sum of ECF precursors should not exceed the upper bound
        # of the ratio of TOP assay precursors (corrected for their PFCA yield
        # [88±12%]) to PFOS reported in 3M AFFF in Houtz et al. Tables S5 and S6
        lowratio = 0.84
        highratio = 2.73
        if PFOS < MDL:
            PFOS = MDL / np.sqrt(2)
            lowratio = 0
        pmin, pmax = PFOS * lowratio, PFOS * highratio

        if pmin < ecf < pmax:
            logprob += 0
        else:
            logprob += BIGNEG

        if emin < e_p < emax:
            logprob += 0
        else:
            logprob += BIGNEG

        # Prevent inference from infering solutions with more than 10x the
        # measured mass (i.e. we don't expect a recovery ≤ 10%)
        if totp/meassum <= 0.7:
            logprob += BIGNEG
        elif totp/meassum >= 2:
            logprob += BIGNEG
        else:
            logprob += 0

        # Evaluate the composition of ECF precursor proposal against their
        # composition reported in 3M AFFF in Houtz et al. Table S6 assuming that
        # the oxidation yields of ECF precursors do not depend on their
        # perfluorinated chain length (n)
        ecf_comp = x_p[self.ecf_indices] / ecf
        logprob += - \
            (np.sum(np.abs((ecf_comp - self.composition_means) / (self.composition_stds))**2))

        for i, xi in enumerate(params[:-1]):
            if xi < jeffreys_min:
                # don't let it waste time wandering arbitrarily low
                logprob += BIGNEG
            if xi > np.log10(meassum):  # or high
                logprob += BIGNEG
        if e_p < MINVAL:
            logprob += BIGNEG

        return logprob

prior_lookup = {"unknown_jeffreys": UnknownJeffreysPrior,
                "unknown": UnknownPrior,
                "AFFF": AFFFPrior,
                "AFFF_impacted": AFFFImpactedPrior,
                "AFFF_impacted_archival": AFFFImpactedPriorArchival,
                "AFFF_impacted_jeffreys": AFFFImpactedJeffreysPrior,
                "biota": BiotaPrior,
                }