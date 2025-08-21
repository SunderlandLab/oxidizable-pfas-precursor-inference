from dataclasses import dataclass
from typing import List
import os
import yaml
import emcee
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

@dataclass
class Posterior:
    logprec_df: pd.DataFrame
    prec_df: pd.DataFrame
    prior_df: pd.DataFrame
    posterior_predictive_df: pd.DataFrame
    prior_predictive_df: pd.DataFrame
    measurements_df: pd.DataFrame
    
    def remove_burnin(self, burnin: int):
        """
        Remove the burnin samples from the posterior

            Parameters:
                burnin (int) : number of samples to remove
        """
        self.logprec_df = self.logprec_df.iloc[burnin:]
        self.prec_df = self.prec_df.iloc[burnin:]
        self.posterior_predictive_df = self.posterior_predictive_df.iloc[burnin:]
        if self.prior_predictive_df is not None:
            self.prior_predictive_df = self.prior_predictive_df.iloc[burnin:]

    def trace(self, label, ax):
        """
        Plot the posterior trace

            Parameters:
                label (str) : A named precursor
                ax (matplotlib.ax) : a matplotlib axis
        """
        assert label in self.prec_df.columns, 'Input a label in precursor_labels'
        ax.plot(self.logprec_df.loc[:, label], color = 'k')

    def show_traces(self, **kwargs):
        plt.figure(figsize=(9,9))
        cols = self.prec_df.columns
        n = int(np.floor(np.sqrt(len(cols))))+1
        for i, prec in enumerate(cols):
            ax = plt.subplot(n, n, i+1)
            ax.title.set_text(prec)
            self.trace(prec, ax)
        plt.tight_layout()

    def mean(self):
        """
        Calculate the mean estimate of each precursor

            Returns:
                (array of floats) : mean of each precursor and total precursors
        """
        return np.mean(self.prec_df.values, axis = 0)
    
    def geomean(self):
        """
        Calculate the geometric mean estimate of each precursor

            Returns:
                (array of floats) : geometric mean of each precursor and total precursors
        """
        
        return 10**np.mean(self.logprec_df.values, axis = 0)
    
    def quantile(self, quantile):
        """
        Calculate the quantile of each precursor

            Parameters:
                quantile (float) : A number between 0 and 1 exclusive
            Returns:
                (array of floats) : quantile estimate of each precursor and total precusors
        """
        assert 0 < quantile < 1, 'Input a quantile between 0 and 1'
        return np.quantile(self.prec_df.values, quantile, axis = 0)

    def histogram(self, label, ax, **kwargs):
        """
        Plot the histogram of posterior

            Parameters:
                label (str) : A named precursor
                ax (matplotlib.ax) : a matplotlib axis
        """
        assert label in self.prec_df.columns, 'Input a label in precursor_labels'
        ax.hist(self.logprec_df.loc[:, label], **kwargs)

    def show_hists(self, **kwargs):
        plt.figure(figsize=(9,9))
        cols = self.prec_df.columns
        n = int(np.floor(np.sqrt(len(cols))))+1
        for i, prec in enumerate(cols):
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
        assert label in self.prec_df.columns, 'Input a label in precursor_labels'
        sns.kdeplot(self.logprec_df.loc[:, label], ax = ax, label = label, **kwargs)

    def show_kdes(self, **kwargs):
        plt.figure(figsize=(9,9))
        cols = self.prec_df.columns
        n = int(np.floor(np.sqrt(len(cols))))+1
        for i, prec in enumerate(cols):
            ax = plt.subplot(n, n, i+1)
            ax.title.set_text(prec)
            self.kde(prec, ax, **kwargs)
        plt.tight_layout()

    def marginalplot(self, ax=None, confidence_interval=(5,95)):
        """
        Plot the kernel density estimator

            Parameters:
                label (str) : A named precursor
                ax (matplotlib.ax) : a matplotlib axis
                confidence_interval (tuple) : whisker bounds
        """
        wide_posterior = self.logprec_df.melt(var_name='Precursor', value_name='log$_{10}$Concentration')
        wide_posterior['type'] = 'Posterior'
        logprior_df = self.prior_df.apply(lambda x: np.log10(x))
        wide_prior = logprior_df.melt(var_name='Precursor', value_name='log$_{10}$Concentration')
        wide_prior['type'] = 'Prior'
        wide = pd.concat([wide_posterior, wide_prior])
        # sns.boxplot(data=wide, x='precursor', y='concentration', ax=ax, showfliers=False,
        #             whis=confidence_interval)
        if ax is None:
             ax = plt.subplot(1,1,1)
        sns.violinplot(data=wide, x='Precursor', y='log$_{10}$Concentration', ax=ax, inner='quart', hue='type',split=True, density_norm='width')
        # plt.semilogy()
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
        assert 0 < quantile < 1, 'Input a quantile between 0 and 1'
        prec_F = self.__organofluorine(n_F)
        return(np.quantile(np.sum(prec_F, axis = 1), quantile))

    def summary_table(self):
        rows = []
        d = {'Statistic': 'Mean'}
        for p, s in zip(self.prec_df.columns, self.mean()):
            d[p] = s
        rows.append(d)
        d = {'Statistic': 'Geometric mean'}
        for p, s in zip(self.prec_df.columns, self.geomean()):
            d[p] = s
        rows.append(d)
        for q in [0.05,0.25,0.5,0.75,0.95]:
            d = {'Statistic': f'{q*100}th percentile'}
            for p, s in zip(self.prec_df.columns, self.quantile(q)):
                d[p] = s
            rows.append(d)
        return pd.DataFrame(rows)
    
    def posterior_predictive(self):
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

        post_predictive = self.posterior_predictive_df.melt(var_name='Product', value_name='Concentration')
        prior_predictive = self.prior_predictive_df.melt(var_name='Product', value_name='Concentration')
        post_predictive['type'] = 'Posterior predictive'
        prior_predictive['type'] = 'Prior predictive'
        post_predictive = pd.concat([post_predictive, prior_predictive])
        
        measurements = {p['name']: (p['value'], p['error'], p['MDL']) for i, p in self.measurements_df.iterrows()}
        sns.violinplot(data=post_predictive, x='Product', y='Concentration', hue='type',inner='quart', split=True, density_norm='width')
        counter = 0
        MDL_IS_SET = False
        MEAS_IS_SET = False
        for p, (value, error, MDL) in measurements.items():
            if value > MDL:
                
                if not MEAS_IS_SET:
                    plt.errorbar(p, value, yerr=error*value, fmt='o', color='k', label='Measured')
                    MEAS_IS_SET = True
                else:
                    plt.errorbar(p, value, yerr=error*value, fmt='o', color='k')
            else:
                if not MDL_IS_SET:
                    plt.plot([counter-0.5,counter+0.5],[MDL,MDL],linestyle='--',lw=3, color='tab:red', label='Non-detect MDL')
                    MDL_IS_SET = True
                else:
                    plt.plot([counter-0.5,counter+0.5],[MDL,MDL],linestyle='--',lw=3, color='tab:red')
            counter += 1
        plt.xticks(rotation=90)
        plt.semilogy()
        plt.legend(bbox_to_anchor=(1.2, 1.01))
            
    def save(self, foldername: str):
        """Save posterior info to files."""
        try:
            os.mkdir(foldername)
        except FileExistsError:
            pass
        for name, df in zip(['log_posterior','posterior','prior','posterior_predictive','observations'],
                            [self.logprec_df, self.prec_df, self.prior_df, self.posterior_predictive_df, self.measurements_df]):
            df.to_csv(os.path.join(foldername, f'{name}.csv'), index=False)
        with open(os.path.join(foldername, 'metadata.yaml'),'w') as f:
            f.writelines(['run at: ' + str(pd.Timestamp.now()) + '\n'])
        if self.prior_predictive_df is not None:
            self.prior_predictive_df.to_csv(os.path.join(foldername, 'prior_predictive.csv'), index=False)

    @classmethod
    def from_emcee(cls, emcee_sampler: emcee.EnsembleSampler, labels: List[str], model=None, nprune=1, product_names: List[str] = None,
                   emcee_sampler_prior: emcee.EnsembleSampler = None, nprune_prior=1):
        n_dimensions = emcee_sampler.ndim
        samples = emcee_sampler.flatchain
        logprec = samples[::nprune,:-1]
        model
        assert len(labels) == logprec.shape[1], f'Specify {logprec.shape[1]} precursor labels'
        logprec_df = pd.DataFrame(logprec, columns=labels)
        prec = 10**logprec
        prec_df = pd.DataFrame(prec, columns=labels)
        #add a column for the sum of all precursors in the last column:
        prec = np.append(prec, np.sum(prec, axis = 1).reshape(len(prec), 1), axis = 1)
        prec_df['Total precursors'] = np.sum(prec, axis = 1)
        logprec = np.append(logprec, np.log10(np.sum(prec, axis = 1).reshape(len(prec), 1)), axis = 1) 
        logprec_df['Total precursors'] = np.log10(np.sum(prec, axis = 1))
        posterior_predictive = np.array([np.dot(model.forward_matrix, prec[i,:-1]) for i in range(len(prec))])
        posterior_predictive_df = pd.DataFrame(posterior_predictive, columns=product_names)
        measurements_dict = {'name':[],'value':[],'error':[],'MDL':[]}
        for i, p in enumerate(model.product_names):
            measurements_dict['name'].append(p)
            measurements_dict['value'].append(model.obs[i])
            measurements_dict['error'].append(model.error_obs[i])
            measurements_dict['MDL'].append(model.mdls[i])
        measurements_df = pd.DataFrame(measurements_dict)
        print(measurements_df)
        if emcee_sampler_prior is not None:
            samples = 10**emcee_sampler_prior.flatchain[::nprune_prior,:-1]
            prior_df = pd.DataFrame(samples, columns=labels)
            #add a column for the sum of all precursors in the last column:
            prior_df['Total precursors'] = np.sum(samples, axis = 1)
            prior_predictive = np.array([np.dot(model.forward_matrix, samples[i,:]) for i in range(len(samples))])
            prior_predictive_df = pd.DataFrame(prior_predictive, columns=product_names)
        else:
            prior_predictive_df = None
        return cls(logprec_df=logprec_df,
                   prec_df=prec_df,
                   prior_df=prior_df,
                   posterior_predictive_df=posterior_predictive_df,
                   prior_predictive_df=prior_predictive_df,
                   measurements_df = measurements_df)

    @classmethod
    def from_saved(cls, foldername: str):
        dataframes = {key: pd.read_csv(os.path.join(foldername, f'{key}.csv')) for key in 
                      ['log_posterior','posterior','prior','posterior_predictive','observations']}
        try:
            prior_predictive = pd.read_csv(os.path.join(foldername, 'prior_predictive.csv'))
        except FileNotFoundError:
            prior_predictive = None
        return cls(
            logprec_df=dataframes['log_posterior'],
            prec_df=dataframes['posterior'],
            prior_df=dataframes['prior'],
            posterior_predictive_df=dataframes['posterior_predictive'],
            prior_predictive_df=prior_predictive,
            measurements_df=dataframes['observations']
        )
