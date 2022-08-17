import numpy as np
from functions import priors, likelihood
from functions import BIGNEG, MINVAL, MAXVAL, INITMIN, INITMAX
import emcee
from emcee.autocorr import AutocorrError
from emcee.moves import DESnookerMove
from lib import Config
from typing import Tuple


class Tuner(object):
    """Decide which stretch parameter is best to use.
    Tracks previous trials and generates new trials
    along with when to stop.
    """

    def __init__(self, max_depth: int=3) -> None:
        """max_depth determines how granular a search for
        the optimal parameter value is conducted.
        """
        self.trial_queue = [.55]
        self.alphas = []
        self.acceptances = []
        self.good_alpha = None
        self.depth = 0
        self.max_depth = max_depth

    def update(self, alpha: float, f_accept: float) -> None:
        """Update record of trials and results."""
        self.alphas.append(alpha)
        self.acceptances.append(f_accept)
        if 0.2 < f_accept < 0.8:
            self.good_alpha = alpha

        self.acceptances = [x for _, x in sorted(zip(self.alphas,
                                                     self.acceptances))]
        self.alphas = sorted(self.alphas)

    def get_trial(self) -> Tuple[float,bool]:
        """What parameter value to try next?

        Returns: alpha, stopcode
        alpha (float) value of parameter for next trial.
        stopcode (boolean) whether to stop trials
        """

        if self.good_alpha is not None:
            return self.good_alpha, True

        if self.depth >= self.max_depth:
            return self.get_consolation(), True

        if len(self.trial_queue) < 1:
            self.update_queue()

        tri = self.trial_queue.pop(0)

        return tri, False

    def update_queue(self) -> None:
        """Add further trials to the queue."""
        alps, accs = self.alphas, self.acceptances
        best = np.argmax(accs)
        EDGE = False
        if best == 0:
            left = 0.
            EDGE = True
        else:
            left = alps[best-1]
        if best == len(accs)-1:
            right = alps[-1] * 2
            EDGE = True
        else:
            right = alps[best+1]

        if not EDGE:
            self.depth += 1

        self.trial_queue.append((alps[best]+left)/2)
        self.trial_queue.append((alps[best]+right)/2)

    def get_consolation(self):
        """Get most value of most successful trial."""
        best = np.argmax(self.acceptances)
        return self.alphas[best]


def prob_func(x, config: Config, prior: str) -> float:
    """ log-probability of posterior.

    Takes current state vector and obs vector
    along with additional information
    and returns posterior log-probability of
    sample based on defined likelihood and prior.

    Arguments:
    x (array) proposal
    config (Config object) configuration info
    prior (str) prior name

    Returns:
    (float) log-probability of proposal given observations
    """

    # log-posterior, so sum prior and likelihood
    lp = priors[prior](x, config)

    if not np.isfinite(lp):
        return BIGNEG
    ll = likelihood(x, config)
    if not np.isfinite(ll):
        return BIGNEG
#    print(ll,lp)
    return ll + lp


def sample_measurement(config: Config, prior:str ='AFFF',
                       nwalkers=32,
                       Nincrement=2000, TARGET_EFFECTIVE_STEPS=2500,
                       MAX_STEPS=100000, MAX_DEPTH=3):
    """For a given measurement b, sample the posterior.

    Some MCMC options are set here to do ensemble
    MCMC sampling in the 8 state vector dimensions
    using 32 walkers doing Snooker moves.

    Arguments:
    config (Config object): contains the sample-specific configuration
    
    Keyword arguments:
    prior (string; default 'AFFF') name of prior function to use
    nwalkers (int; default 32) number of samplers in ensemble
    Nincrement (int; default 2000) interations between status checks
    TARGET_EFFECTIVE_STEPS (int; default 2500) effective sample size goal
    MAX_STEPS (int; default 100000) cap on number of interations
    MAX_DEPTH (int; default 3) level of granularity in
               windowed meta-parameter search

    Returns:
    (emcee.EnsembleSampler) ensemble of samplers with results
    """

    ndim = len(config.possible_precursors)+1
    WEGOOD = False

    tuner = Tuner(max_depth=MAX_DEPTH)
    print('-' * 50)
    print(f'Number of walkers moving on each iteration: {nwalkers}')
    print('Doing burn-in initialization and parameter tuning...')
    while not WEGOOD:

        alpha, WEGOOD = tuner.get_trial()
        if WEGOOD:
            print(f'alpha of {alpha} selected.')
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        prob_func,
                                        args=(config, prior),
                                        moves=[(DESnookerMove(alpha),
                                                1.0)])
        
        # Make sure the sampling is initialized in reasonable values:
        pfos_measured = config.pfos[0]
        if (prior in ['unknown', 'unknown_jeffreys']) or (pfos_measured <= 0):
            init_min0, init_max0 = INITMIN.get(prior,-1), INITMAX.get(prior,1)
        else:
            init_min0, init_max0 = np.log10(pfos_measured)+INITMIN.get(prior,-1), np.log10(pfos_measured)+INITMAX.get(prior,1)
        INITIALIZED = False
        # The sequence of definitions of "reasonable" to try
        init_sequence = [(-1,-1),(-2,-2),(-1,1),(-1,0),(-2,0),(-4,-2)]
        init_min, init_max, init_counter = init_min0 + 0, init_max0 + 0, 0
        while not INITIALIZED:
            try:
                init = np.random.rand(nwalkers, ndim)*(init_max - init_min) + init_min
                state = sampler.run_mcmc(init, Nincrement*5)
                INITIALIZED = True
            except ValueError: # sometimes an unlucky init will cause infinite values
                # Change the space of initial values you consider reasonable if it doesn't work
                if init_counter > len(init_sequence):
                    raise ValueError("Can't seem to find initial conditions allowed by the prior...")
                init_min += init_sequence[init_counter][0]
                init_max += init_sequence[init_counter][1]
                init_counter += 1
        sampler.reset()
        S = 1
        state = sampler.run_mcmc(
            state, Nincrement, skip_initial_state_check=True)

        f_accept = np.mean(sampler.acceptance_fraction)
        print(
            f'acceptance rate is {np.mean(f_accept):.2f} when alpha is {alpha}')
        tuner.update(alpha, f_accept)

    print(f'Sampling posterior in {Nincrement}-iteration increments.')
    WEGOOD = False
    count = 0
    prev_Nindep = 0
    Nindep = 1
    sampler.reset()
    while (not WEGOOD) and (count < MAX_STEPS):
        state = sampler.run_mcmc(
            state, Nincrement, skip_initial_state_check=True)
        f_accept = np.mean(sampler.acceptance_fraction)
        count += Nincrement
        try:
            tac = sampler.get_autocorr_time()
            mtac = np.nanmax(tac)  # go by the slowest-sampling dim or mean??
            if np.isnan(mtac):
                WEGOOD = False
            else:
                WEGOOD = True
        except AutocorrError:
            mtac = 'unavailable'
            WEGOOD = False
        print(f'After {count} iterations, autocorr time: {mtac}')
    WEGOOD = False
    while (not WEGOOD) and (count < MAX_STEPS):
        if Nindep < prev_Nindep:
            print("WARNING: Number of independent samples decreasing!")

        state = sampler.run_mcmc(
            state, Nincrement, skip_initial_state_check=True)
        f_accept = np.mean(sampler.acceptance_fraction)
        count += Nincrement
        try:
            tac = sampler.get_autocorr_time()
            mtac = np.nanmax(tac)
        except AutocorrError:
            pass
        prev_Nindep = Nindep
        Nindep = count * nwalkers / mtac
        print(
            f'After {count} iterations, effective number of samples:\
                 {int(Nindep)}'
        )
        if Nindep > TARGET_EFFECTIVE_STEPS:
            WEGOOD = True
    if MAX_STEPS <= count:
        print("WARNING: maximum number of iterations reached! Terminating.")
    print('SAMPLE DONE')
    return sampler
