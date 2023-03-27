from .config import Config
from .measurements import Measurements
from .sampler import MCMCSampler, Tuner
from .model import ModelLikelihood
from .core import Problem, Distribution
from .priors import prior_lookup

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

@dataclass
class PrecursorProblem(Problem):
    n_dimensions: int
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    state_vector_labels: List[str]
    prior: Distribution
    likelihood: Distribution
    

    def get_bounds(self) -> Tuple:
        return (self.lower_bounds, self.upper_bounds)

def sample_measurement(config: Config, measurments, prior_name, Nincrement, 
                        TARGET_EFFECTIVE_STEPS, MAX_STEPS, MAX_DEPTH, alpha=-1):
    precursors = config.possible_precursors
    print(config)
    print(measurments)
    likelihood_model = ModelLikelihood(measurements=measurments, config=config)
    prior = prior_lookup[prior_name](precursors=precursors, measurements=measurments,
                                    jeffreys_variance=config.jeffreys_variance)
    n_dimensions = len(precursors) + 1
    lower_bounds, upper_bounds = np.zeros(n_dimensions)-4, np.zeros(n_dimensions)+4
    problem = PrecursorProblem(n_dimensions=n_dimensions,
                                prior=prior,
                                likelihood=likelihood_model,
                                lower_bounds=lower_bounds,
                                upper_bounds=upper_bounds,
                                state_vector_labels=precursors)

    sampler = MCMCSampler(max_steps=MAX_STEPS, Nwalkers=2,
                          target_effective_steps=TARGET_EFFECTIVE_STEPS,
                          Nincrement=Nincrement)
    tuner = Tuner(max_depth=MAX_DEPTH)
    if alpha < 0:
        alpha = sampler.tune_alpha(problem=problem, tuner=tuner)
    posterior = sampler.sample(problem=problem, alpha=alpha)

    return posterior