NOTICE: <em>We are transitioning to a more user-friendly and interactive formulation of this code. The updated routines and demonstration (see [demo_workbook.ipynb](./demo_workbook.ipynb)) fix some undesirable behavior under some conditions, help to problem solve common issues, and make understanding your inference easier. Presently, we are working on transitioning previously published sample inferences to the new format for archival consistency. If you have any questions or issues, please let us know (thackray@seas.harvard.edu) </em>

# Description
oxidizable-pfas-precursor-inference is a tool to infer concentrations of oxidizable
precursors aggregated by perfluorinated chain length (n) and manufacturing
origin (electrochemical fluorination: ECF vs fluorotelomerization: FT) based
on changes in perfluoroalkyl carboxylates (PFCA) in the total oxidizable
precursor (TOP) assay.

The workflow found in [demo_workbook.ipynb](demo_workbook.ipynb) can be used to analyze TOP assay results for any aqueous
sample with the appropriate choice of a prior. This package
provides several built in priors for samples described below. In the "config" folder, you can find two
versions of each prior (with either uniform uninformed priors or Jeffrey's
uninformed priors for precursor chain lengths without more information
available).
  * prior_AFFF – used for AFFF stocks ([Ruyle et al. 2021a](http://dx.doi.org/10.1021/acs.estlett.0c00798))
  * prior_AFFF_jeffreys – used for AFFF stocks
  * prior_AFFF_impacted – used for AFFF impacted water ([Ruyle et al. 2021b](http://dx.doi.org/10.1021/acs.est.0c07296))
  * prior_AFFF_impacted_jeffreys – used for AFFF impacted water
  * prior_unknown – used for water with unknown PFAS point sources ([Ruyle et al. 2021b](http://dx.doi.org/10.1021/acs.est.0c07296))  
  * prior_unknown_jeffreys – used for water with unknown PFAS point sources ([Ruyle et al. 2021b](http://dx.doi.org/10.1021/acs.est.0c07296))  
  * prior_unknown_fish - used for fish with unknown PFAS sources ([Pickard et al. 2022](http://dx.doi.org/10.1021/acs.est.2c03734))

These priors can be used as a template and adapted for other specific purposes.
Current precursor chainlength that can be inferred include n:2 FT (n = 4,6,8,10),
n:3 FT (n = 5,7,9), and Cn ECF (n = 4-10). If you are interested in adding additional
chain lengths, please contact the model creators.

### Credit:
  * Co-authorship is appropriate if your paper benefits significantly from use
  of this model/code  
  * Citation is appropriate if use of this model/code has only a marginal impact
  on your work or if the work is a second generation application of the model/code.

This model was created by
[Colin P. Thackray](https://scholar.harvard.edu/thackray/about) and
[Bridger J. Ruyle](https://scholar.harvard.edu/ruyle) and originally
presented in [Ruyle et al. 2021a](http://dx.doi.org/10.1021/acs.estlett.0c00798)

### Citation for code:

Ruyle, B. J.; Thackray, C. P.; McCord, J. P.; Strynar, M. J.; Mauge-Lewis, K. A.; Fenton, S. E.; Sunderland, E. M. Reconstructing the Composition of Per- and Polyfluoroalkyl Substances (PFAS) in Contemporary Aqueous Film Forming Foams. Environ. Sci. Technol. Lett. 2021, 8(1), 59-65. [https://doi.org/10.1021/acs.estlett.0c00798](http://dx.doi.org/10.1021/acs.estlett.0c00798).

# Input format
All measurement data should be contained in a csv file. There are two acceptable
input formats for the concentration data.

Format 1 (shown in the 'Ruyle_Cape_Cod_rivers.csv' and 'Ruyle_Houtz_AFFF_stocks.csv'
files) should include column headers for measured changes in concentrations of
Cn PFCA, where n=number of per fluorinated carbons after the TOP assay.
`C3, C4, C5, C6, C7, C8`  
(C3=PFBA, C4=PFPeA, C5=PFHxA, C6=PFHpA, C7=PFOA, C8=PFNA)  

Format 2 (shown in the 'Tokranov_Ashumet_Pond_lake_shadow.csv'
files) should include column headers for the concenrtations of Cn PFCA before
and after the TOP assay.
`C3pre, C4pre, C5pre, C6pre, C7pre, C8pre`
and
`C3post, C4post, C5post, C6post, C7post, C8post`

In both cases, csv file should include the name of the prior to use:  
`prior_name`
the detection limits (LOD, MDL, MQL, etc.):  
`C3MDL, C4MDL, C5MDL, C6MDL, C7MDL, C8MDL`  
and associated measurement errors (ex: relative percent difference of replicate analyses):  
`C3err, C4err, C5err, C6err, C7err, C8err`  

Method errors can be assessed in multiple ways including by the relative percent difference of replicate analyses of the same sample. You can use total method error by setting all err columns equal to the same value or compound specific method errors by
setting C3-C8err to their own values.

In both cases, csv files should include column headers for PFOS:
`PFOS, PFOSMDL, PFOSerr`
If you are using an AFFF or AFFF_impacted prior, these columns should be
populated with data. If you are not using priors that rely on the measurement of
PFOS, then they can be lefts empty.

If a concentration was measured below the MDL, you can either fill it with the
value of the MDL or any number between 0 and the MDL.

# Output
The output samples are saved as a collection of tables in csv format
and contain samples of the posterior precursor concentrations and log10 precursor concentrations, prior precursor concentrations, along with the posterior and prior predictive distributions. This output can be read by your data analysis and plotting pipeline of choice (e.g. pandas + seaborn).

# Example data
Three datasets are provided as examples <em>Note: lining up the priors used in these papers with the current code is in progress.</em>
* 'Ruyle_Houts_AFFF_stocks.csv' which contains TOP assay data from ECF and fluorotelomer AFFF
reported by [Ruyle et al. 2021a](http://dx.doi.org/10.1021/acs.estlett.0c00798) and
[Houtz et al. 2013](https://doi.org/10.1021/es4018877).
* 'Ruyle_Cape_Cod_rivers.csv' which contains TOP assay data from surface water on Cape
Cod, Massachusetts, USA reported by [Ruyle et al. 2021b](http://dx.doi.org/10.1021/acs.est.0c07296)
* 'Tokranov_Ashumet_Pond_lake_shadow.csv' which contains TOP assay data from Ashumet Pond and downgradient groundwater on Cape
Cod, Massachusetts, USA reported by [Tokranov et al. 2022](http://xlink.rsc.org/?DOI=D1EM00329A)
* 'Pickard_NH_fish.csv' which contains TOP assay data from fish from New Hampshire surface waters 
reported by [Pickard et al. 2022](http://dx.doi.org/10.1021/acs.est.2c03734)
# Python dependencies
Python >= 3.7  
[numpy](https://numpy.org/doc/stable/user/install.html)  
[pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)  
[emcee](https://emcee.readthedocs.io/en/stable/user/install/)  
[pyyaml](https://pypi.org/project/PyYAML/)
