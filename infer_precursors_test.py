""" Generate samples of the posterior for PFAA precursors from TOP assay
measurements in aqueous matrices.

Authors:
Colin Thackray (thackray@seas.harvard.edu)
Bridger Ruyle (bruyle@g.harvard.edu)
"""
import argparse
import numpy as np
import pandas as pd
from src.measurements import Measurements
from src.run import sample_measurement
from src.config import Config
import warnings

warnings.filterwarnings('ignore')

# Command line arguments
parser = argparse.ArgumentParser(
    description='Sample posterior for precursors.')
parser.add_argument('ISTART', metavar='istart', type=int,
                    help='first sample index (first index is 0)', default=0)
parser.add_argument('IEND', metavar='iend', type=int, nargs='?',
                    help='last sample index (first index is 0)', default=None)
parser.add_argument('-d', '--datafile', dest='FILENAME', action='store',
                    default='data/measurements.csv',
                    help='location of measurements file')
parser.add_argument('-o', '--outfile', dest='OUTFILE_STEM', action='store',
                    default='infer_out/mcmcout_',
                    help='Stem for output filename. Will gain suffix \
                    N (sample index)')
parser.add_argument('-t', '--target-steps', dest='TARGET',
                    action='store', default=2500, type=int,
                    help='Effective sample size to attain')
parser.add_argument('-m', '--max-steps', dest='MAX_STEPS',
                    action='store', default=50000, type=int,
                    help='Maximum number of steps before quiting.')
parser.add_argument('-D', '--max-depth', dest='MAX_DEPTH',
                    action='store', default=3, type=int,
                    help='Maximum depth of windowing in sampler tuning.')
parser.add_argument('-a', '--alpha', dest='alpha',
                    action='store', default=-1, type=float,
                    help='Alpha for sampler.')

args = parser.parse_args()
if args.IEND is None:
    args.IEND = args.ISTART

# Load input data from disk
df = pd.read_csv(args.FILENAME)
names = df['Sample'].values


# Do sampling for requested measurements
for bi in range(args.ISTART, args.IEND+1):
    print('Calculating for sample ' + df['Sample'][bi])

    measurements = Measurements.from_row(df.iloc[bi])
    config = Config.from_yaml(measurements.associated_config)
    prior_name = config.prior_name
    # print(config)

    # Run MCMC ensemble to sample posterior
    posterior = sample_measurement(config, measurements,
                                 prior_name=prior_name,
                                 Nincrement=5000,
                                 TARGET_EFFECTIVE_STEPS=args.TARGET,
                                 MAX_STEPS=args.MAX_STEPS,
                                 MAX_DEPTH=args.MAX_DEPTH,
                                 alpha=args.alpha)

    # Save sampling output to disk
    trajectory = posterior.samples[:, :-1]
    outfile = f'{args.OUTFILE_STEM}{bi}'
    np.save(outfile, trajectory)
