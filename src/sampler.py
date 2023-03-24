from dataclasses import dataclass
from typing import Optional, Protocol
from .core import Problem, UniformBounded

import numpy as np
from numpy.typing import ArrayLike

import emcee
from emcee.autocorr import AutocorrError
from emcee.moves import DESnookerMove, GaussianMove

import matplotlib.pyplot as plt


class Tuner(Protocol):
    """Form of tuner objects."""

    def is_tuned(self):
        """Is the tuning finished?"""

    def get_trial(self):
        """What parameter value to try next?"""


@dataclass
class Posterior:
    n_dimensions: int
    samples: np.ndarray

    def show_trace(self, **kwargs):
        plt.figure()
        for n in range(self.n_dimensions):
            plt.subplot(2, 3, n+1)
            plt.plot(self.samples[:, n], **kwargs)

    def show_hist(self, **kwargs):
        plt.figure()
        for n in range(self.n_dimensions):
            plt.subplot(2, 3, n+1)
            plt.hist(self.samples[:, n], **kwargs)

    def save(self, filename: str):
        """Save posterior to file."""
        ...

    @classmethod
    def from_emcee(cls, emcee_sampler: emcee.EnsembleSampler):
        return Posterior(n_dimensions=emcee_sampler.ndim, samples=emcee_sampler.flatchain)

    @classmethod
    def from_npy(cls, filename: str):
        samples = np.load(filename)
        assert len(
            samples.shape) == 2, f'File {filename} wrong shape for posterior'
        return Posterior(n_dimensions=samples.shape[1], samples=samples)


class Sampler(Protocol):
    """Form of sampler objects."""

    def sample(self, problem: Problem) -> Posterior:
        """Get sample of posterior distribution."""
        ...


@dataclass
class MCMCSampler:
    """Markov chain Monte Carlo."""

    Nwalkers: int = 2
    Nincrement: int = 2000
    target_effective_steps: int = 2500
    max_steps: int = 50000
    default_alpha: float = 0.3

    def sample(self, problem: Problem, alpha: float = -1, movetype='DESnooker') -> Posterior:
        """For a given inference Problem, sample the posterior."""
        if alpha <= 0.0:
            alpha = self.default_alpha

        total_walkers = self.Nwalkers*problem.n_dimensions
        if movetype in ['DESnooker']:
            Mover = DESnookerMove
        elif movetype in ['Gaussian']:
            Mover = GaussianMove
        else:
            raise ValueError(f"movetype '{movetype} unknown...")
        mcsampler = emcee.EnsembleSampler(total_walkers,
                                          problem.n_dimensions,
                                          problem.posterior,
                                          moves=[(Mover(alpha),
                                                  1.0)])
        MINVAL, MAXVAL = problem.get_bounds()
        init = np.random.rand(total_walkers,
                              problem.n_dimensions)\
            * (MAXVAL-MINVAL)+MINVAL

        state = mcsampler.run_mcmc(init, self.Nincrement*5)
        mcsampler.reset()
        S = 1
        state = mcsampler.run_mcmc(
            state, self.Nincrement, skip_initial_state_check=True)
        f_accept = np.mean(mcsampler.acceptance_fraction)
        print(
            f'acceptance rate is {np.mean(f_accept):.2f} when alpha is {alpha}')
        print(f'Sampling posterior in {self.Nincrement}-iteration increments.')
        WEGOOD = False
        count = 0
        prev_Nindep = 0
        Nindep = 1
        mcsampler.reset()
        while (not WEGOOD) and (count < self.max_steps):
            state = mcsampler.run_mcmc(
                state, self.Nincrement, skip_initial_state_check=True)
            f_accept = np.mean(mcsampler.acceptance_fraction)
            count += self.Nincrement
            try:
                tac = mcsampler.get_autocorr_time()
                # go by the slowest-sampling dim or mean??
                mtac = np.nanmax(tac)
                if np.isnan(mtac):
                    WEGOOD = False
                else:
                    WEGOOD = True
            except AutocorrError:
                mtac = 'unavailable'
                WEGOOD = False
            print(f'After {count} iterations, autocorr time: {mtac}')
        WEGOOD = False
        while (not WEGOOD) and (count < self.max_steps):
            if Nindep < prev_Nindep:
                print("WARNING: Number of independent samples decreasing!")

            state = mcsampler.run_mcmc(
                state, self.Nincrement, skip_initial_state_check=True)
            f_accept = np.mean(mcsampler.acceptance_fraction)
            count += self.Nincrement
            try:
                tac = mcsampler.get_autocorr_time()
                mtac = np.nanmax(tac)
            except AutocorrError:
                pass
            prev_Nindep = Nindep
            Nindep = count * total_walkers / mtac
            print(
                f'After {count} iterations, effective number of samples:\
                    {int(Nindep)}'
            )
            if Nindep > self.target_effective_steps:
                WEGOOD = True
        if self.max_steps <= count:
            print("WARNING: maximum number of iterations reached! Terminating.")
        print('SAMPLE DONE')
        return Posterior.from_emcee(mcsampler)

    def tune_alpha(self, problem: Problem, tuner: Tuner) -> float:
        """Tune hyperparameter for snooker move for given problem."""
        return self.default_alpha


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    sampler = MCMCSampler(max_steps=50000, Nwalkers=4,
                          target_effective_steps=3500)

    class MyProblem(Problem):

        def __init__(self, n_dimensions, lower_bounds, upper_bounds):
            self.n_dimensions = n_dimensions
            self.lower_bounds = np.array(lower_bounds)
            self.upper_bounds = np.array(upper_bounds)
            self.likelihood = UniformBounded(
                lower_bounds=self.lower_bounds, upper_bounds=self.upper_bounds)

        def get_bounds(self):
            return self.lower_bounds, self.upper_bounds

    problem = MyProblem(2, [-3, 2], [3, 7])
    posterior = sampler.sample(problem=problem)
    posterior.show_trace()
    posterior.show_hist(bins=np.linspace(-5, 10, 31))
    plt.show()
