import numpy as np
from emcee.autocorr import integrated_time, AutocorrError
from typing import Union, Callable

def rhat(samples: np.ndarray,) -> float:
    """
    Calculate the R-hat statistic for convergence diagnostics.
    Args:
        samples (np.ndarray): Samples from the posterior
    Returns:
        float: R-hat statistic
    """
    # Calculate the mean and variance of the samples
    N,m,d = samples.shape
    n = N//2
    samples = np.hstack((samples[:n,:,:], samples[n:,:,:]))
    theta_dotm = np.mean(samples, axis=0)
    theta_dotdot = np.mean(samples, axis=(0,1))
    B = n/(m-1) * np.sum((theta_dotm - theta_dotdot)**2, axis=0)
    sm2 = 1/(n-1) * np.sum((samples - theta_dotm)**2, axis=0)
    W = np.mean(sm2, axis=0)
    
    # Calculate the R-hat statistic
    v_thetay = (n-1)/n * W + B/n
    rhat = np.sqrt(v_thetay/W)
    
    return np.max(rhat)

def ESS(samples: np.ndarray) -> float:
    """
    Calculate the effective sample size (ESS) for convergence diagnostics.
    Args:
        samples (np.ndarray): Samples from the posterior
    Returns:
        float: Effective sample size
    """
    try:
        autoco = np.max(integrated_time(samples))
        ess = len(samples)/autoco
    except AutocorrError:
        ess = 0.0
    return ess

class StopCondition:
    stop: bool
    rhat_threshold: Union[float, None]
    iterations_threshold: Union[int, None]
    max_iterations: int
    ESS_threshold: Union[int, None]
    condition_type: Callable
    
    def __bool__(self) -> bool:
        """
        Check if the stop condition is met.
        Returns:
            bool: True if the stop condition is met, False otherwise.
        """
        return bool(self.stop)
    
    def __init__(self,
                 rhat_threshold: Union[float, None] = None,
                 iterations_threshold: Union[int, None] = None,
                 max_iterations: int = 150000,
                 ESS_threshold: Union[int, None] = None,
                 ) -> None:
        """
        Initialize the stop condition.
        Args:
            rhat_threshold (float, optional): R-hat threshold for convergence. Defaults to None.
            iterations_threshold (int, optional): Iterations threshold for convergence. Defaults to None.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 150000.
            ESS_threshold (int, optional): Effective sample size threshold for convergence. Defaults to None.
            condition_type (Callable, optional): Type of stop condition. Defaults to None.
        """
        self.rhat_threshold = rhat_threshold
        self.iterations_threshold = iterations_threshold
        self.max_iterations = max_iterations
        self.ESS_threshold = ESS_threshold
        self.stop = False
    
    def update(self, samples: np.ndarray, current_iterations: int) -> None:
        """
        Update the stop condition based on the samples and iterations.
        Args:
            samples (np.ndarray): Samples from the posterior
            iterations (int): Number of iterations
        """
        current_rhat = rhat(samples)
        current_ESS = ESS(samples)
        if current_iterations > self.max_iterations:
            self.stop = True
        else:
            self.stop = np.all([(self.rhat_threshold is None or current_rhat < self.rhat_threshold),
                    (self.iterations_threshold is None or current_iterations > self.iterations_threshold),
                    (self.ESS_threshold is None or current_ESS > self.ESS_threshold)])
        return current_rhat, current_ESS
    
    def reset(self) -> None:
        """
        Reset the stop condition.
        """
        self.stop = False