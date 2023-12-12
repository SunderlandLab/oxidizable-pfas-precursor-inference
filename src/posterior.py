from dataclasses import dataclass
from typing import List
import os
import yaml
import emcee
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Posterior:
    def __init__(self, n_dimensions: int, samples: np.ndarray, labels: List[str] =[], model = None):
        self.n_dimensions = n_dimensions
        self.logprec = samples[:,:-1]
        self.model = model
        assert len(labels) == self.logprec.shape[1], f'Specifify {self.logprec.shape[1]} precursor labels'
        
        prec = 10**self.logprec
        #add a column for the sum of all precursors in the last column:
        self.prec = np.append(prec, np.sum(prec, axis = 1).reshape(len(prec), 1), axis = 1)
        self.logprec = np.append(self.logprec, np.log10(np.sum(prec, axis = 1).reshape(len(prec), 1)), axis = 1) 
        self.precursor_labels = labels + ['Total precursors']

    def trace(self, label, ax):
        """
        Plot the posterior trace

            Parameters:
                label (str) : A named precursor
                ax (matplotlib.ax) : a matplotlib axis
        """
        assert label in self.precursor_labels, f'Input a label in precursor_labels'
        idx = self.precursor_labels.index(label)
        ax.plot(self.logprec[:, idx], color = 'k')

    def show_traces(self, **kwargs):
        plt.figure(figsize=(9,9))
        n = int(np.floor(np.sqrt(self.n_dimensions)))+1
        for i, prec in enumerate(self.precursor_labels):
            ax = plt.subplot(n, n, i+1)
            ax.title.set_text(prec)
            self.trace(prec, ax)

    def mean(self):
        """
        Calculate the mean estimate of each precursor

            Returns:
                (array of floats) : mean of each precursor and total precursors
        """
        return np.mean(self.prec, axis = 0)
    
    def geomean(self):
        """
        Calculate the geometric mean estimate of each precursor

            Returns:
                (array of floats) : geometric mean of each precursor and total precursors
        """
        
        return 10**np.mean(self.logprec, axis = 0)
    
    def quantile(self, quantile):
        """
        Calculate the quantile of each precursor

            Parameters:
                quantile (float) : A number between 0 and 1 exclusive
            Returns:
                (array of floats) : quantile estimate of each precursor and total precusors
        """
        assert 0 < quantile < 1, f'Input a quantile between 0 and 1'
        return np.quantile(self.prec, quantile, axis = 0)

    def histogram(self, label, ax, **kwargs):
        """
        Plot the histogram of posterior

            Parameters:
                label (str) : A named precursor
                ax (matplotlib.ax) : a matplotlib axis
        """
        assert label in self.precursor_labels, f'Input a label in precursor_labels'
        idx = self.precursor_labels.index(label)
        ax.hist(self.logprec[:, idx], **kwargs)

    def show_hists(self, **kwargs):
        plt.figure(figsize=(9,9))
        n = int(np.floor(np.sqrt(self.n_dimensions)))+1
        for i, prec in enumerate(self.precursor_labels):
            ax = plt.subplot(n, n, i+1)
            ax.title.set_text(prec)
            self.histogram(prec, ax, **kwargs)

    def kde(self, label, ax, **kwargs):
        """
        Plot the kernel density estimator

            Parameters:
                label (str) : A named precursor
                ax (matplotlib.ax) : a matplotlib axis
                color : line color for kernel density
        """
        assert label in self.precursor_labels, f'Input a label in precursor_labels'
        idx = self.precursor_labels.index(label)
        sns.kdeplot(self.logprec[:, idx], ax = ax, label = label, **kwargs)

    def show_kdes(self, **kwargs):
        plt.figure(figsize=(9,9))
        n = int(np.floor(np.sqrt(self.n_dimensions)))+1
        for i, prec in enumerate(self.precursor_labels):
            ax = plt.subplot(n, n, i+1)
            ax.title.set_text(prec)
            self.kde(prec, ax, **kwargs)

    def boxplot(self, ax=None, confidence_interval=(5,95)):
        """
        Plot the kernel density estimator

            Parameters:
                label (str) : A named precursor
                ax (matplotlib.ax) : a matplotlib axis
                confidence_interval (tuple) : whisker bounds
        """
        if ax is None:
            ax = plt.subplot(1,1,1)
        ax.boxplot(self.prec, whis = confidence_interval, showfliers = False,
                   labels=self.precursor_labels, )
        ax.semilogy()
        plt.xticks(rotation=90)

    def fluorotelomer_fraction(self):
        """
        Calculate the fraction of precursors that are fluorotelomers

            Returns:
                (tuple) : fluorotelomer fraction mean and standard deviation
        """
        fluorotelomer_concentration = np.sum(self.prec[:,['FT' in p for p in self.precursor_labels]], axis = 1)
        total_concentration = np.sum(self.prec[:,:-1], axis = 1)
        ft_fraction = fluorotelomer_concentration / total_concentration
        return(np.mean(ft_fraction), np.std(ft_fraction))

    def __organofluorine(self, n_F):
        """
        Convert from molar units to fluorine equivalents

            Paramets:
                n_F (array) : number of fluorines for each precursor
            Returns:
                (array) : precursor concentrations in fluorine equivalents
        """
        individual_precursors = self.prec[:, :-1]
        assert len(n_F) == individual_precursors.shape[1], 'specify the number of fluorines for each precursor'
        precursor_F = np.multiply(individual_precursors, n_F)
        return(precursor_F)

    def eof_mean(self, n_F):
        """
        Calculates mean fluorine equivalents of precursors

            Paramets:
                n_F (array) : number of fluorines for each precursor class
            Returns:
                (float) : mean precursor concentration in fluorine equivalents
        """
        prec_F = self.__organofluorine(n_F)
        return(np.mean(np.sum(prec_F, axis = 1)))

    def eof_stdev(self, n_F):
        """
        Calculates mean fluorine equivalents of precursors

            Paramets:
                n_F (array) : number of fluorines for each precursor class
            Returns:
                (float) : standard deviation of precursor concentration in fluorine equivalents
        """
        prec_F = self.__organofluorine(n_F)
        return(np.std(np.sum(prec_F, axis = 1)))

    def eof_quantile(self, n_F, quantile):
        """
        Calculates mean fluorine equivalents of precursors

            Paramets:
                n_F (array) : number of fluorines for each precursor class
                quantile (float) : A number between 0 and 1 exclusive
            Returns:
                (float) : standard deviation of precursor concentration in fluorine equivalents
        """
        assert 0 < quantile < 1, f'Input a quantile between 0 and 1'
        prec_F = self.__organofluorine(n_F)
        return(np.quantile(np.sum(prec_F, axis = 1), quantile))

    def summary_table(self):
        rows = []
        d = {'Statistic': 'Mean'}
        for p, s in zip(self.precursor_labels, self.mean()):
            d[p] = s
        rows.append(d)
        d = {'Statistic': 'Geometric mean'}
        for p, s in zip(self.precursor_labels, self.geomean()):
            d[p] = s
        rows.append(d)
        for q in [0.05,0.25,0.5,0.75,0.95]:
            d = {'Statistic': f'{q*100}th percentile'}
            for p, s in zip(self.precursor_labels, self.quantile(q)):
                d[p] = s
            rows.append(d)
        return pd.DataFrame(rows)
    
    def posterior_predictive(self, top_delta, n_posterior):
        """
        Plot posterior predictive against the measured delta of the TOP assay

            Parameters:
                top_delta (array) : measured delta in the TOP assay
                n_posterior (int) : number of samples from the posterior to compare
                infered_columns (list) : column indices of inferred columns (see makeA.py for indices)
                measured_pfca (list) : row indices of the measured PFCA (see makeA.py for indices)
            Returns:
                fig, ax (matplotlib figure) : figure of the posterior predictive
        """


        assert self.model is not None, 'Forward model unavailable for posterior predictive'
        A, U = self.model.forward_matrix, self.model.error_matrix
    

        random_indices = np.random.choice(range(self.n_dimensions), size=n_posterior)
        post_predictive = []
        for i in random_indices:
            post_predictive.append(np.dot(A, self.prec[i, :-1]))

        fig, ax = plt.subplots()
        for idx, pp in enumerate(post_predictive):
            if idx == 0:
                ax.scatter(range(len(top_delta)), pp, label = 'posterior predictive', color = 'k', alpha = 0.5)
            else:
                ax.scatter(range(len(top_delta)), pp, color = 'k', alpha = 0.5)
        ax.scatter(range(len(top_delta)), top_delta, label = 'measured delta', color = 'red')
        ax.set_xticks(range(len(top_delta)))
        ax.set_ylabel('Concentration')
        ax.legend()
        return(fig, ax)

    def save(self, foldername: str):
        """Save posterior to file."""
        try:
            os.mkdir(foldername)
        except FileExistsError:
            pass
        np.save(os.path.join(foldername, 'samples.npy'), self.logprec)
        with open(os.path.join(foldername, 'metadata.yaml'),'w') as f:
            f.writelines(['precursor_labels:\n'] + [f' - "{label}"\n' for label in self.precursor_labels[:-1]])

    @classmethod
    def from_emcee(cls, emcee_sampler: emcee.EnsembleSampler, labels: List[str], model=None):
        return Posterior(n_dimensions=emcee_sampler.ndim, samples=emcee_sampler.flatchain,
                         labels=labels, model=model)

    @classmethod
    def from_npy(cls, filename: str, labels):
        samples = np.load(filename)
        assert len(
            samples.shape) == 2, f'File {filename} wrong shape for posterior'
        return Posterior(n_dimensions=samples.shape[1], samples=samples, labels=labels)
    
    @classmethod
    def from_save(cls, foldername: str):
        sample_filename = os.path.join(foldername, 'samples.npy')
        samples = np.load(sample_filename)
        metadata = yaml.safe_load(open(os.path.join(foldername, 'metadata.yaml'), 'r'))
        labels = metadata.get('precursor_labels', [])
        assert len(
            samples.shape) == 2, f'File {sample_filename} wrong shape for posterior'
        return Posterior(n_dimensions=samples.shape[1], samples=samples, labels=labels)
