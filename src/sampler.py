from dataclasses import dataclass
from typing import Optional, Protocol, Tuple
from .core import Problem, UniformBounded
from .posterior import Posterior

import numpy as np
from numpy.typing import ArrayLike

import emcee
from emcee.autocorr import AutocorrError
from emcee.moves import DESnookerMove, GaussianMove



# class Tuner(Protocol):
#     """Form of tuner objects."""

#     def is_tuned(self):
#         """Is the tuning finished?"""

#     def get_trial(self):
#         """What parameter value to try next?"""

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


class Sampler(Protocol):
    """Form of sampler objects."""

    def sample(self, problem: Problem) -> Posterior:
        """Get sample of posterior distribution."""
        ...


@dataclass
class MCMCSampler:
    """Markov chain Monte Carlo."""

    Nwalkers: int = 2
    Nincrement: int = 5000
    target_effective_steps: int = 2500
    max_steps: int = 150000
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
            print("WARNING: maximum number of iterations reached! Terminating. Something might have gone wrong.")
        if np.isinf(np.mean(mcsampler.flatchain)):
            print("WARNING: one or more of the samplers ran off to inifinity! Something must have gone wrong.")
        print('SAMPLE DONE')
        return Posterior.from_emcee(mcsampler, labels=problem.state_vector_labels, model=problem.likelihood)

    def tune_alpha(self, problem: Problem, tuner: Tuner, 
                    movetype='DESnooker') -> float:
        """Tune hyperparameter for snooker move for given problem."""
        print('Doing burn-in initialization and parameter tuning...')
        WEGOOD = False
        while not WEGOOD:

            alpha, WEGOOD = tuner.get_trial()
            if WEGOOD:
                print(f'alpha of {alpha} selected.')
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
            
            state = mcsampler.run_mcmc(
                state, self.Nincrement, skip_initial_state_check=True)

            f_accept = np.mean(mcsampler.acceptance_fraction)
            print(
                f'acceptance rate is {np.mean(f_accept):.2f} when alpha is {alpha}')
            tuner.update(alpha, f_accept)
        return alpha


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
