import numpy as np
import pandas as pd
from lib import Config

BIGNEG = -1e32
MINVAL, MAXVAL = -6, 4
INITMIN = {'AFFF_impacted':-4, 'AFFF_impacted_targeted':-1, 'unknown_jeffreys': -1}
INITMAX = {'AFFF_impacted':1, 'AFFF_impacted_targeted':0, 'unknown_jeffreys': 1}

def likelihood(x, config: Config) -> float:
    """log-likelihood of x given b.

    Calculates the log-probability of x given observations b,
    with additional considerations based on the MDLs and
    whether C8 (PFNA) was measured.

    Arguments:
    x (array) simulated proposal of precursors
    config (Config object) configuration info for sample

    Returns:
    (float) log-likelihood
    """
    b = config.b
    bpre = config.bpre
    bpost = config.bpost

    # print(bpre, bpost)
    mdls = config.mdls
    berr = config.berr

    x_p = 10**x[:-1]  # transform from log space
    # (last value is an error term)
    logprob = 0

    A, U = config.model, config.uncertainty

    mm = np.dot(A, x_p)
    uu = np.dot(U, x_p)
    mod = np.log10(mm)

    # errors
    e_p = x[-1]  # error parameter
    moderr = (uu / mm)  # fractional error
    obserr = berr
    toterr = (moderr**2 + obserr**2 + e_p**2)**0.5

    # concentrations by difference
    if ((bpre is not None) and (bpost is not None)):
        for i in range(len(bpre)):
            delta = bpost[i] - bpre[i]
            if bpre[i] > mdls[i]:
                rsd = bpre[i]*berr[i]
            else:
                rsd = mdls[i]*berr[i]
            if delta <= rsd:
                # change in PFCA concentration is indistinguishable from experimental error
                obsmin = MINVAL  # don't want -inf
                obsmax = np.log10(rsd)
                if obsmin <= mod[i] < obsmax:
                    logprob += 0
                elif mod[i] > obsmax:
                    logprob += -((mod[i] - obsmax) / toterr[i])**2
                else:
                    logprob += BIGNEG
            else:
                obs = np.log10(delta)
                logprob += -((mod[i] - obs) / toterr[i])**2

    else:
        for i in range(len(b)):
            if b[i] <= mdls[i]:
                # non-detect
                obsmin = MINVAL  # don't want -inf
                obsmax = np.log10(mdls[i] / np.sqrt(2))
                if obsmin <= mod[i] < obsmax:
                    logprob += 0
                elif mod[i] > obsmax:
                    logprob += -((mod[i] - obsmax) / toterr[i])**2
                else:
                    logprob += BIGNEG
            else:
                obs = np.log10(b[i])
                logprob += -((mod[i] - obs) / toterr[i])**2

    # if np.isnan(logprob):
        # print(x_p, A)
    return logprob

def prior_AFFF(x, config: Config) -> float:
    """Prior log-probability of proposal x.
    Prior used for AFFF samples.

    Arguments:
    x (array) proposal precursors
    config (Config object) configuration info for sample

    Returns:
    (float) log-prior
    """
    PFOS2 = config.pfos
    logprob = 0
    PFOS, MDL = PFOS2

    emin, emax = 0, 2

    x_p = 10**x[:-1]
    e_p = x[-1]
    ecf = np.sum(x_p[3:])  # sum only the ECF precursors

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
    ecf_comp = x_p[3:] / ecf
    logprob += -(np.sum(np.abs((ecf_comp - config.compmeans) / (2 * config.compstds))**2))

    for i, xi in enumerate(x):
        if xi < MINVAL:
            # don't let it waste time wandering arbitrarily low
            logprob += BIGNEG
        if xi > MAXVAL:  # or high
            logprob += BIGNEG
    if e_p < MINVAL:
        logprob += BIGNEG

    return logprob

def prior_AFFF_jeffreys(x, config: Config) -> float:
    """Prior log-probability of proposal x.

    Prior used for AFFF samples.

    Arguments:
    x (array) proposal precursors
    config (Config object) configuration info for sample

    Returns:
    (float) log-prior
    """
    var = config.jeffreys_variance
    if var is None:
        raise ValueError('Please define jeffreys_variance in yaml config')

    PFOS2 = config.pfos
    logprob = 0
    PFOS, MDL = PFOS2

    emin, emax = 0, 2

    x_p = 10**x[:-1]
    e_p = x[-1]
    ecf = np.sum(x_p[3:])  # sum only the ECF precursors

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
        logprob += 0
    else:
        logprob += BIGNEG

    if emin < e_p < emax:
        logprob += 0
    else:
        logprob += BIGNEG

    # Evaluate the composition of ECF precursor proposal against their
    # composition reported in 3M AFFF in Houtz et al. Table S6 assuming that
    # the oxidation yields of ECF precursors do not depend on their
    # perfluorinated chain length (n)
    ecf_comp = x_p[3:] / ecf
    logprob += - \
        (np.sum(np.abs((ecf_comp - config.compmeans) / (2 * config.compstds))**2))

    jeffreys_min = np.log10(pmax) - var

    for i, xi in enumerate(x):
        if xi < jeffreys_min:
            # don't let it waste time wandering arbitrarily low
            logprob += BIGNEG
        if xi > np.log10(pmax):  # or high
            logprob += BIGNEG

    return logprob


def prior_AFFF_impacted(x, config: Config) -> float:
    """Prior log-probability of proposal x.

    Prior used for environmental samples where predominant PFAS source
    is thought to be AFFF.

    Arguments:
    x (array) proposal precursors
    config (Config object) configuration info for sample

    Returns:
    (float) log-prior
    """
    PFOS2 = config.pfos
    b = config.b
    bpre = config.bpre
    bpost = config.bpost

    logprob = 0
    PFOS, MDL = PFOS2

    emin, emax = 0, 2

    x_p = 10**x[:-1]
    totp = np.sum(x_p)
    if (bpre is not None) and (bpost is not None):
        meassum = (bpost-bpre).sum()
    else:
        meassum = b.sum()

    x_p = 10**x[:-1]
    e_p = x[-1]
    ecf = np.sum(x_p[config.ecf_indices])  # sum only the ECF precursors

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
    if totp/meassum <= 1:
        logprob += BIGNEG
    elif totp/meassum >= 10:
        logprob += BIGNEG
    else:
        logprob += 0

    # Evaluate the composition of ECF precursor proposal against their
    # composition reported in 3M AFFF in Houtz et al. Table S6 assuming that
    # the oxidation yields of ECF precursors do not depend on their
    # perfluorinated chain length (n)
    ecf_comp = x_p[config.ecf_indices] / ecf
    logprob += - \
        (np.sum(np.abs((ecf_comp - config.compmeans) / (config.compstds))**2))

    for i, xi in enumerate(x):
        if xi < MINVAL:
            # don't let it waste time wandering arbitrarily low
            logprob += BIGNEG
        if xi > np.log10(meassum):  # or high
            logprob += BIGNEG
    if e_p < MINVAL:
        logprob += BIGNEG

    return logprob

def prior_AFFF_impacted_jeffreys(x, config: Config) -> float:
    """Prior log-probability of proposal x.

    Prior used for environmental samples where predominant PFAS source
    is thought to be AFFF.

    Arguments:
    x (array) proposal precursors
    config (Config object) configuration info for sample

    Returns:
    (float) log-prior
    """
    PFOS2 = config.pfos
    targetprec = config.targetprec
    targeted_indices = config.targeted_indices
    b = config.b
    bpre = config.bpre
    bpost = config.bpost

    var = config.jeffreys_variance
    if var is None:
        raise ValueError('Please define jeffreys_variance in yaml config')

    logprob = 0
    PFOS, MDL = PFOS2

    emin, emax = 0, 2

    x_p = 10**x[:-1]
    totp = np.sum(x_p)
    if (bpre is not None) and (bpost is not None):
        meassum = (bpost-bpre).sum()
    else:
        meassum = b.sum()

    x_p = 10**x[:-1]
    e_p = x[-1]
    ecf = np.sum(x_p[config.ecf_indices])  # sum only the ECF precursors

    # the sum of ECF precursors should not exceed the upper bound
    # of the ratio of TOP assay precursors (corrected for their PFCA yield
    # [88±12%]) to PFOS reported in 3M AFFF in Houtz et al. Tables S5 and S6
    highratio = 2.73
    if PFOS < MDL:
        PFOS = MDL / np.sqrt(2)
    pmax = PFOS * highratio

    if ecf < pmax:
        logprob += 0
    else:
        logprob += BIGNEG

    if emin < e_p < emax:
        logprob += 0
    else:
        logprob += BIGNEG

    # make sure the targeted measurements line up with the right parameters
    for i, tp in zip(targeted_indices, targetprec):
        xi = x_p[i]
        if xi < tp:
            # Prevent solutions where infered concentration < targeted
            # precursor concentrations
            logprob += BIGNEG

    # Prevent inference from infering solutions with more than 10x the
    # measured mass (i.e. we don't expect a recovery ≤ 10%)
    if totp/meassum <= 1:
        logprob += BIGNEG
    elif totp/meassum >= 10:
        logprob += BIGNEG
    else:
        logprob += 0

    # Evaluate the composition of ECF precursor proposal against their
    # composition reported in 3M AFFF in Houtz et al. Table S6 assuming that
    # the oxidation yields of ECF precursors do not depend on their
    # perfluorinated chain length (n)
    ecf_comp = x_p[config.ecf_indices] / ecf
    logprob += - \
        (np.sum(np.abs((ecf_comp - config.compmeans) / (config.compstds))**2))

    jeffreys_min = np.log10(meassum) - var
    for i, xi in enumerate(x):
        if xi < jeffreys_min:
            # don't let it waste time wandering arbitrarily low
            logprob += BIGNEG
        if xi > np.log10(meassum):  # or high
            logprob += BIGNEG

    return logprob


def prior_unknown(x, config: Config) -> float:
    """Prior log-probability of proposal x.

    Prior used for environmental samples where predominant PFAS source
    is unknown.

    Arguments:
    x (array) proposal precursors
    config (Config object) configuration info for sample

    Returns:
    (float) log-prior
    """

    cost = 0

    b = config.b
    bpre = config.bpre
    bpost = config.bpost
    targetprec = config.targetprec
    targeted_indices = config.targeted_indices

    if (bpre is not None) and (bpost is not None):
        meassum = (bpost-bpre).sum()
    else:
        meassum = b.sum()

    x_p = 10**x[:-1]
    totp = np.sum(x_p)

    # Prevent inference from infering solutions with more than 10x the
    # measured mass
    if totp/meassum <= 1:
        cost += BIGNEG
    elif totp/meassum >= 10:
        cost += BIGNEG
    else:
        cost += 0

    for i, xi in enumerate(x):
        if xi < MINVAL:
            # don't let it waste time wandering arbitrarily low
            cost += BIGNEG
        if xi > MAXVAL:  # or high
            cost += BIGNEG

    # make sure the targeted measurements line up with the right parameters
    for i, tp in zip(targeted_indices, targetprec):
        xi = x_p[i]
        if xi < tp:
            # Prevent solutions where infered concentration < targeted
            # precursor concentrations
            cost += BIGNEG
    return cost

def prior_unknown_jeffreys(x, config: Config) -> float:
    """Prior log-probability of proposal x.

    Prior used for environmental samples where predominant PFAS source
    is unknown.

    Arguments:
    x (array) proposal precursors
    config (Config object) configuration info for sample

    Returns:
    (float) log-prior
    """

    cost = 0

    b = config.b
    bpre = config.bpre
    bpost = config.bpost
    targetprec = config.targetprec
    targeted_indices = config.targeted_indices

    var = config.jeffreys_variance
    if var is None:
        raise ValueError('Please define jeffreys_variance in yaml config')

    if (bpre is not None) and (bpost is not None):
        meassum = (bpost-bpre).sum()
    else:
        meassum = b.sum()

    jeffreys_min = np.log10(meassum) - var

    x_p = 10**x[:-1]
    totp = np.sum(x_p)

    # Prevent inference from infering solutions with more than 10x the
    # measured mass
    if totp/meassum <= 1:
        cost += BIGNEG
    elif totp/meassum >= 10:
        cost += BIGNEG
    else:
        cost += 0

    for i, xi in enumerate(x[:-1]):
        xi_lin = 10**xi
        if xi < jeffreys_min:
            # don't let it waste time wandering arbitrarily low
            cost += BIGNEG
        if (xi_lin/meassum) >= 10:  # or high
            cost += BIGNEG
    if x[-1] < MINVAL:
        cost += BIGNEG
    if x[-1] > MAXVAL:
        cost += BIGNEG

    # make sure the targeted measurements line up with the right parameters
    for i, tp in zip(targeted_indices, targetprec):
        xi = x_p[i]
        if xi < tp:
            # Prevent solutions where infered concentration < targeted
            # precursor concentrations
            cost += BIGNEG
    return cost

# Collect priors by name for easy lookup
priors = {'AFFF': prior_AFFF,
          'AFFF_jeffreys': prior_AFFF_jeffreys,
          'AFFF_impacted': prior_AFFF_impacted,
          'AFFF_impacted_jeffreys': prior_AFFF_impacted_jeffreys,
          'unknown': prior_unknown,
          'unknown_jeffreys': prior_unknown_jeffreys}
