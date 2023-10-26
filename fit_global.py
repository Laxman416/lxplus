"""
fit_global.py

This code is used to perform a global fit on the selected data. In order to do so a simulatenous fit is done on the four datasets (with different mesons and polarities). This simulatenous fit keeps all variables constant across the four fits except for the normalization constants which are allowed to vary independently. The model used consists of a Crystal Ball function and a Gaussian distribution to model the signal and a Chebychev polynomial to model the background.
The year of interest and size of the data to be analysed must be specified using the required flags --year --size. It is necessary to specify if the fit should be performed on the binned data or the unbinned data using the flag --binned_fit. There is a flag --path, which is not required. This one is used to specify the directory where the input data is located, and where the output file should be written. By default it is set to be the current working directory.
It outputs the value of the constants shared in the simultaneous fit to a text file.

Author: Marc Oriol Pérez (marc.oriolperez@student.manchester.ac.uk)
Last edited: 15th September 2023
"""

import ROOT
import numpy as np
import uproot
import argparse
import os
from ROOT import TChain, RooRealVar, RooDataSet, RooGaussian, RooCrystalBall, RooBifurGauss, RooChebychev, RooAddPdf, RooArgList, RooFit, RooArgSet, RooGenericPdf, RooJohnson, RooUnblindUniform

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
                in the case it is not specified, the default path is the current working directory.
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
        "--path",
        type=dir_path,
        required=False,
        default=os.getcwd(),
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
args = parse_arguments()
numbins = 100
if args.binned_fit=="y" or args.binned_fit=="Y":
    binned = True
else:
    binned = False
ROOT.gROOT.SetBatch(True)

ttree_D0_up = TChain("D02Kpi_Tuple/DecayTree")
ttree_D0_up.Add(f"{args.path}/D0_up_data_{args.year}_{args.size}_clean.root")
ttree_D0_up.SetBranchStatus("*", 0)
ttree_D0_up.SetBranchStatus("D0_MM", 1)

ttree_D0_down = TChain("D02Kpi_Tuple/DecayTree")
ttree_D0_down.Add(f"{args.path}/D0_down_data_{args.year}_{args.size}_clean.root")
ttree_D0_down.SetBranchStatus("*", 0)
ttree_D0_down.SetBranchStatus("D0_MM", 1)

ttree_D0bar_up = TChain("D02Kpi_Tuple/DecayTree")
ttree_D0bar_up.Add(f"{args.path}/D0bar_up_data_{args.year}_{args.size}_clean.root")
ttree_D0bar_up.SetBranchStatus("*", 0)
ttree_D0bar_up.SetBranchStatus("D0_MM", 1)

ttree_D0bar_down = TChain("D02Kpi_Tuple/DecayTree")
ttree_D0bar_down.Add(f"{args.path}/D0bar_down_data_{args.year}_{args.size}_clean.root")
ttree_D0bar_down.SetBranchStatus("*", 0)
ttree_D0bar_down.SetBranchStatus("D0_MM", 1)

D0_M = ROOT.RooRealVar("D0_MM", "D0 mass / [MeV/c*c]", 1810, 1910)
data_D0_up = RooDataSet("data_D0_up", "Data_D0_up", ttree_D0_up, RooArgSet(D0_M))
data_D0bar_up = RooDataSet("data_D0bar_up", "Data_D0bar_up", ttree_D0bar_up, RooArgSet(D0_M))
data_D0_down = RooDataSet("data_D0_down", "Data_D0_down", ttree_D0_down, RooArgSet(D0_M))
data_D0bar_down = RooDataSet("data_D0bar_down", "Data_D0bar_down", ttree_D0bar_down, RooArgSet(D0_M))


# Model Gaussian
mean = ROOT.RooRealVar("mean", "mean", 1865, 1850, 1880)
sigma = ROOT.RooRealVar("sigma", "sigma", 6.59, 0, 100)
gaussian = ROOT.RooGaussian("gauss", "gauss", D0_M, mean, sigma)

# Model CrystalBall
Cmu = RooRealVar("Cmu", "Cmu", 1865.07, 1855, 1875)
Csig = RooRealVar("Csig", "Csig", 10.65, 0, 100)
aL = RooRealVar("aL", "aL", 1.77, -10, 10)
nL = RooRealVar("nL", "nL", 9.5, -10, 10)
aR = RooRealVar("aR", "aR", 3.73, -10, 10)
nR = RooRealVar("nR", "nR", 4.34, -10, 10)
crystal = RooCrystalBall("Crystal", "Crystal Ball", D0_M, Cmu, Csig, aL, nL, aR, nR)

# Model Signal
frac_D0_up = RooRealVar("frac_D0_up", "frac_D0_up", 0.567, 0, 1)
frac_D0_down = RooRealVar("frac_D0_down", "frac_D0_down", 0.567, 0, 1)
frac_D0bar_up = RooRealVar("frac_D0bar_up", "frac_D0bar_up", 0.567, 0, 1)
frac_D0bar_down = RooRealVar("frac_D0bar_down", "frac_D0bar_down", 0.567, 0, 1)

signal_D0_up = RooAddPdf("signal_D0_up", "signal_D0_up", RooArgList(gaussian, crystal), RooArgList(frac_D0_up))
signal_D0_down = RooAddPdf("signal_D0_down", "signal_D0_down", RooArgList(gaussian, crystal), RooArgList(frac_D0_down))
signal_D0bar_up = RooAddPdf("signal_D0bar_up", "signal_D0bar_up", RooArgList(gaussian, crystal), RooArgList(frac_D0bar_up))
signal_D0bar_down = RooAddPdf("signal_D0bar_down", "signal_D0bar_down", RooArgList(gaussian, crystal), RooArgList(frac_D0bar_down))

# Model Chebyshev
a0 = ROOT.RooRealVar("a0", "a0", -0.4, -5, 5)
chebyshev = RooChebychev("chebyshev", "chebyshev", D0_M, RooArgList(a0))

# Generate normalization variables
Nsig_D0_up = ROOT.RooRealVar("Nsig_D0_up", "Nsig_D0_up", 0.95*ttree_D0_up.GetEntries(), 0, ttree_D0_up.GetEntries())
Nsig_D0bar_up = ROOT.RooRealVar("Nsig_D0bar_up", "Nsig_D0bar_up", 0.95*ttree_D0bar_up.GetEntries(), 0, ttree_D0bar_up.GetEntries())
Nbkg_D0_up = ROOT.RooRealVar("Nbkg_D0_up", "Nbkg_D0_up", 0.05*ttree_D0_up.GetEntries(), 0, ttree_D0_up.GetEntries())
Nbkg_D0bar_up = ROOT.RooRealVar("Nbkg_D0bar_up", "Nbkg_D0bar_up", 0.05*ttree_D0bar_up.GetEntries(), 0, ttree_D0bar_up.GetEntries())
Nsig_D0_down = ROOT.RooRealVar("Nsig_D0_down", "Nsig_D0_down", 0.95*ttree_D0_down.GetEntries(), 0, ttree_D0_down.GetEntries())
Nsig_D0bar_down = ROOT.RooRealVar("Nsig_D0bar_down", "Nsig_D0bar_down", 0.95*ttree_D0bar_down.GetEntries(), 0, ttree_D0bar_down.GetEntries())
Nbkg_D0_down = ROOT.RooRealVar("Nbkg_D0_down", "Nbkg_D0_down", 0.05*ttree_D0_down.GetEntries(), 0, ttree_D0_down.GetEntries())
Nbkg_D0bar_down = ROOT.RooRealVar("Nbkg_D0bar_down", "Nbkg_D0bar_down", 0.05*ttree_D0bar_down.GetEntries(), 0, ttree_D0bar_down.GetEntries())

# Generate models
model_D0_up = ROOT.RooAddPdf("model_D0_up", "model_D0_up", [signal_D0_up, chebyshev], [Nsig_D0_up, Nbkg_D0_up])
model_D0bar_up = ROOT.RooAddPdf("model_D0bar_up", "model_D0bar_up", [signal_D0bar_up, chebyshev], [Nsig_D0bar_up, Nbkg_D0bar_up])
model_D0_down = ROOT.RooAddPdf("model_D0_down", "model_D0_down", [signal_D0_down, chebyshev], [Nsig_D0_down, Nbkg_D0_down])
model_D0bar_down = ROOT.RooAddPdf("model_D0bar_down", "model_D0bar_down", [signal_D0bar_down, chebyshev], [Nsig_D0bar_down, Nbkg_D0bar_down])

sample = ROOT.RooCategory("sample", "sample")
sample.defineType("D0_up")
sample.defineType("D0_down")
sample.defineType("D0bar_up")
sample.defineType("D0bar_down")

# Combine all models in order to perform a simultaneous fit for all polarities aand mesons
combData = ROOT.RooDataSet(
    "combData",
    "combined data",
    {D0_M},
    Index=sample,
    Import={"D0_up": data_D0_up, "D0_down": data_D0_down, "D0bar_up": data_D0bar_up, "D0bar_down": data_D0bar_down},
)

simPdf = ROOT.RooSimultaneous("simPdf", "simultaneous pdf", {"D0_up": model_D0_up, "D0_down": model_D0_down, "D0bar_up": model_D0bar_up, "D0bar_down": model_D0bar_down}, sample)

# Perform the fit
if binned:
    hcombData = combData.binnedClone()
    fitResult = simPdf.fitTo(hcombData, IntegrateBins=1e-3, PrintLevel=-1, Save=True)
else:
    fitResult = simPdf.fitTo(combData, PrintLevel=-1, Save=True)

fitResult.Print()

# Get results
parameters = np.array([mean.getValV(), sigma.getValV(), Cmu.getValV(), Csig.getValV(), aL.getValV(), nL.getValV(), aR.getValV(), nR.getValV(), a0.getValV()])
np.savetxt(f"{args.path}/fit_parameters.txt", parameters, delimiter=',')