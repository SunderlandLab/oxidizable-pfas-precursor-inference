"""Core classes for general form of inference problem."""

from abc import ABC, abstractmethod

from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple

import numpy as np


class Distribution(Protocol):
    """Generic distribution on some parameters."""

    def __call__(self, params: np.ndarray) -> float:
        """Log-probability for set of parameter values."""
        ...


class Uninformed:
    """Distribution which adds no information."""

    def __call__(self, params: np.ndarray) -> float:
        """Everything equally likely."""
        return 0.


# class LeftCensored

# class Etc

@dataclass
class UniformBounded:
    """Uniform distribution with bounds."""

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    def __call__(self, params: np.ndarray) -> float:
        """Everything equally likely within bounds."""
        if (params > self.lower_bounds).all() and (params < self.upper_bounds).all():
            return 0.
        else:
            return -50000000


class Problem(ABC):
    """Form of inference problems."""

    n_dimensions: int
    prior: Distribution = Uninformed()
    likelihood: Distribution = Uninformed()

    def posterior(self, params: np.ndarray) -> float:
        """Log-probability of the posterior at given parameter values."""
        lp = self.prior(params)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.likelihood(params)
        if not np.isfinite(ll):
            return -np.inf
        return ll + lp

    @abstractmethod
    def get_bounds(self) -> Tuple:
        """Get bounds of parameter space to sample."""
