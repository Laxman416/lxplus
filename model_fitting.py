"""
model_fitting.py

This code is used to fit the data in one of the bims. It then returns the relevant plots of the best fit to the data and a .txt file containing the values and errors on the normalization constant of both signal and background, the mean and standard deviation of the pull distribution and the reduced chi squared value. It can both fit the data using a binned approach or an unbinned one. The model used consists of a Gaussian function and a Crystal Ball function for the signal, and a Chevychev polynomial for the background. Some of the parameters are fixed to be the same as the best-fit values obtained during the global fit in order to obtain better convergence.
The year of interest, size of the data, meson of interest and polarity to be analysed must be specified using the required flags --year --size --meson --polarity. It is also required to specify the bin to be analyzed using the flag --bin, and if the fit should be done on the binned data or the unbinned data using the flag --binned_fit. There also are the flags --input --parameteers_path and --path, which are not required. These are used to specify the directory where the input data is located, where the global best-fit parameters can be found and where the output should be written, respectively. By default it is set to be the current working directory.

Author: Marc Oriol Pérez (marc.oriolperez@student.manchester.ac.uk)
Last edited: 16th September 2023
"""

# - - - - - - IMPORT STATEMENTS - - - - - - #

import argparse
import os
from utils import plot
import numpy as np
from ROOT import TChain, RooRealVar, RooDataSet, RooGaussian, RooCrystalBall, RooBifurGauss, RooChebychev, RooAddPdf, RooArgList, RooFit, RooArgSet, RooGenericPdf, RooJohnson, RooUnblindUniform

# - - - - - - - FUNCTIONS - - - - - - - #
def dir_path(string):
    '''
    Checks if a given string is the path to a directory.
    If affirmative, returns the string. If negative, gives an error.
    '''
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
        
def parse_arguments():
    '''
    Parses the arguments needed along the code. Arguments:
    
    --year      Used to specify the year at which the data was taken the user is interested in.
                The argument must be one of: [16, 17, 18]. These referr to 2016, 2017 & 2018, respectively.
    --size      Used to specify the amount of events the user is interested in analysing.
                The argument must be one of: [large, small, medium, 1-8]. The integers specify the number of root
                files to be read in. Large is equivalent to 8. Medium is equivalent to 4. Small takes 200000 events.
    --polarity  Used to specify the polarity of the magnet the user is interested in.
                The argument must be one of: [up, down].
    --meson     Used to specify the meson the user is interested in.
                The argument must be one of: [D0, D0bar, both].
    --input     Used to specify the directory in which the input data should be found. It is not required,
                in the case it is not specified, the default path is the current working directory.
    --path      Used to specify the directory in which the output files should be written. It is not required,
                in the case it is not specified, the default path is the current working directory.
    --parameters_path
                Used to specify the directory in which the global best-fit parameters should be found. It is not required,
                in the case it is not specified, the default path is the current working directory.
    --bin       Used to select the bin to be analysed.
                The argument must be an integer between 00 and 99 (note that two characters must be entered for all integers)
    --binned_fit
                Used to specify if the data should be binned before performing the fit or an unbinned fit should be performed.
                Type either y or Y for a binned fit. Type n or N for an unbinned fit.
                
    Returns the parsed arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        type=int,
        choices=[16,17,18],
        required=True,
        help="flag to set the data taking year."
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["large", "medium", "small", "1", "2", "3", "4", "5", "6", "7", "8"],
        required=True,
        help="flag to set the data taking year."
    )
    parser.add_argument(
        "--polarity",
        type=str,
        choices=["up","down"],
        required=True,
        help="flag to set the data taking polarity."
    )
    parser.add_argument(
        "--meson",
        type=str,
        choices=["D0","D0bar","both"],
        required=True,
        help="flag to set the D0 meson flavour."
    )    
    parser.add_argument(
        "--path",
        type=dir_path,
        required=False,
        default=os.getcwd(),
        help="flag to set the path where the output files should be written to"
    )
    parser.add_argument(
        "--parameters_path",
        type=dir_path,
        required=False,
        default=os.getcwd(),
        help="flag to set the path where the global best fit parameters are found"
    )
    parser.add_argument(
        "--input",
        type=dir_path,
        required=False,
        default=os.getcwd(),
        help="flag to set the path where the input files should be taken from"
    )
    parser.add_argument(
        "--bin",
        type=str,
        required=True,
        help="flag to set the path where the output files should be written to"
    )
    parser.add_argument(
        "--binned_fit",
        type=str,
        choices=["y", "Y", "n", "N"],
        required=True,
        help="flag to set whether a binned or an unbinned should be performed (y/n)"
    )
    
    return parser.parse_args()

# - - - - - - - MAIN BODY - - - - - - - #

options = parse_arguments()
numbins = 100
if options.binned_fit=="y" or options.binned_fit=="Y":
    binned = True
else:
    binned = False
    
parameters = np.loadtxt(f"{options.parameters_path}/fit_parameters.txt", delimiter=',')

# Read data
ttree = TChain("D02Kpi_Tuple/DecayTree")
ttree.Add(f"{options.input}/{options.meson}_{options.polarity}_{options.year}_{options.size}_bin{options.bin}.root")

ttree.SetBranchStatus("*", 0)
ttree.SetBranchStatus("D0_MM", 1)
x = RooRealVar("D0_MM", "D0 mass / [MeV]", 1810, 1910) # D0_MM - invariant mass
data = RooDataSet("data", "Data", ttree, RooArgSet(x))

if binned:
    x.setBins(numbins)

# Define variables
mu = RooRealVar("mu", "mu", 1865, 1862, 1868)
Gsig = RooRealVar("sigma", "sigma", 6.59, 0, 100)
Gauss = RooGaussian("Gauss", "Gaussian", x, mu, Gsig)

Csig = RooRealVar("Csig", "Csig", 10.65, 0, 100)
aL = RooRealVar("aL", "aL", 1.77, -10, 10)
nL = RooRealVar("nL", "nL", 9.5, -10, 10)
aR = RooRealVar("aR", "aR", parameters[6])
nR = RooRealVar("nR", "nR", parameters[7])
Crystal = RooCrystalBall("Crystal", "Crystal Ball", x, mu, Csig, aL, nL, aR, nR)

frac = RooRealVar("frac", "frac", 0.575, 0, 1)

a = RooRealVar("a", "a", parameters[8])
chebychev = RooChebychev("Chebychev", "Chebychev", x, RooArgList(a))

Nsig = RooRealVar("Nsig", "Nsig", 0.95*ttree.GetEntries(), 0, ttree.GetEntries())
Nbkg = RooRealVar("Nbkg", "Nbkg", 0.05*ttree.GetEntries(), 0, ttree.GetEntries())

# Create model
signal = RooAddPdf("signal", "signal", RooArgList(Gauss, Crystal), RooArgList(frac))
model = {
    "total": RooAddPdf("total", "Total", RooArgList(signal, chebychev), RooArgList(Nsig, Nbkg)), # extended likelihood
    "signals": {
        Gauss.GetName(): Gauss.GetTitle(),
        Crystal.GetName(): Crystal.GetTitle(),
    },
    "backgrounds": {
        chebychev.GetName(): chebychev.GetTitle()
    }
}

# Fit data
if binned:
    hdata = data.binnedClone()
    model["total"].fitTo(hdata, RooFit.Save(), RooFit.IntegrateBins(1e-3), RooFit.Extended(1), RooFit.Minos(0))
    data = hdata
else:
    model["total"].fitTo(data, RooFit.Save(), RooFit.Extended(1), RooFit.Minos(0))

# Generate plots
chi2, pull_mean, pull_std = plot(x, data, model, nbins=numbins, setlogy=False, save_to=f'{options.path}/{options.meson}_{options.polarity}_{options.year}_{options.size}_bin{options.bin}', plot_type=f"20{options.year} Mag{(options.polarity).title()}", meson=options.meson)
# Write out results
file = open(f"{options.path}/yields_{options.meson}_{options.polarity}_{options.year}_{options.size}_bin{options.bin}.txt", "w")
text = str(Nsig.getValV()) + ', ' + str(Nsig.getError()) + ', ' + str(Nbkg.getValV()) + ', ' + str(Nbkg.getError()) + ', ' + str(chi2) + ', ' + str(pull_mean) + ', ' + str(pull_std)
file.write(text)
file.close

print(ttree.GetEntries())