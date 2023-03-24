# build the forward model/likelihood and state vector based on Config and Measurements
# forward model should take state vector and compute a likelihood using its __call__ method to work for inferengine
# record state vector meta as well

from .measurements import Measurement, Measurements
from .config import Config
from .core import Problem, Distribution
from .yields import YieldGenerator
from typing import Tuple, List
from dataclasses import dataclass
import numpy as np

MINVAL = -6
BIGNEG = -1e32
# might have to generate priors based on decided state vector?

class ModelLikelihood:
    forward_matrix: np.ndarray
    error_matrix: np.ndarray
    obs: np.ndarray
    error_obs: np.ndarray
    mdls: np.ndarray
    minval: float

    def __init__(self, measurements: Measurements,
                        config: Config, minval=MINVAL):
        # build matrix forward model and U matrix
        # set up likelihood function to use state vector and matrices
        measured_pfcas = sorted(measurements.PFCAs.keys())
        precursors = config.possible_precursors
        m = np.zeros((len(measured_pfcas), len(precursors)))
        u = np.zeros_like(m)

        yields = YieldGenerator(config=config)
        for i, prec in enumerate(precursors):
            for j, pfca in enumerate(measured_pfcas):
                y, unc = yields.get_yield(prec, pfca)
                m[j,i] = y
                u[j,i] = unc
                # print(prec, pfca, y, unc)

        self.forward_matrix = m
        self.error_matrix = u
        self.obs = np.array([measurements.PFCAs[m].value for m in measured_pfcas])
        self.error_obs = np.array([measurements.PFCAs[m].error for m in measured_pfcas])
        self.mdls = np.array([measurements.PFCAs[m].MDL for m in measured_pfcas])
        self.minval = minval

    def __call__(self, params: np.ndarray) -> float:
        """Log-probability for set of parameter values."""
        pre = 10**params[:-1]
        pfcas = np.dot(self.forward_matrix, pre)
        u_pfcas = np.dot(self.error_matrix, pre)
        log_pfcas = np.log10(pfcas)

        e_p = params[-1]  # error parameter
        moderr = (u_pfcas / pfcas)  # fractional error
        obserr = self.error_obs
        toterr = (moderr**2 + obserr**2 + e_p**2)**0.5
        mdls = self.mdls
        b = self.obs

        # print('.',log_pfcas)
        # print(b)
        # print(moderr)
        # print(toterr)

        logprob = 0

        # split up detects and non-detects into different matrices in advance
        for i in range(len(obserr)):
            if b[i] <= mdls[i]:
                # non-detect
                obsmin = self.minval  # don't want -inf
                obsmax = np.log10(mdls[i] / np.sqrt(2))
                if obsmin <= log_pfcas[i] < obsmax:
                    logprob += 0
                elif log_pfcas[i] > obsmax:
                    logprob += -((log_pfcas[i] - obsmax) / toterr[i])**2
                else:
                    # logprob += BIGNEG
                    # give an on-ramp to reasonable values
                    logprob += -((log_pfcas[i] - obsmin) / toterr[i] /100)**2
            else:
                obs = np.log10(b[i])
                logprob += -((log_pfcas[i] - obs) / toterr[i])**2

        return logprob


