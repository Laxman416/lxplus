import ROOT
from ROOT import TChain, RooRealVar, RooDataSet, RooGaussian, RooCrystalBall, RooBifurGauss, RooChebychev, RooAddPdf, RooArgList, RooFit, RooArgSet, RooGenericPdf, RooJohnson, RooUnblindUniform
import uproot
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from pylab import cm
from utils import plot
import argparse

# -- parsing -- #

parser = argparse.ArgumentParser()
parser.add_argument(
    "--year",
    type=int,
    choices=[16,17,18],
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
    choices=["D0","D0bar"],
    required=True,
    help="flag to set the D0 meson flavour."
)
parser.add_argument(
    "--model",
    type=str,
    choices=["2","4","5","14"],
    required=True,
    help="flag to set the model."
)
parser.add_argument(
    "--size",
    type=str,
    choices=["large", "medium", "small", "1", "2", "3", "4", "5", "6", "7", "8"],
    required=True,
    help="flag to set the model."
)
ttree = TChain("D02Kpi_Tuple/DecayTree")
ttree.Add(f"/afs/cern.ch/work/l/lseelan/{options.meson}_{options.polarity}_data_{options.year}_{options.size}_clean.root")

ttree.SetBranchStatus("*", 0)
ttree.SetBranchStatus("D0_MM", 1)
x = RooRealVar("D0_MM", "D0 mass / [MeV]", 1810, 1910) # D0_MM - invariant mass
data = RooDataSet("data", "Data", ttree, RooArgSet(x))

def gauss_crystal_chebychev(x,data,ttree):
    """Model 14:
    Signal - gauss and crystall ball function
    Background - chebychev

    Args:
        x (RooRealVar): invariant mass data of D0_MM
        data (RooDataSet): Assigning data to variable
        ttree (tree): data in the file
    """
    
    #Gaussian parameters
    Gmu14 = RooRealVar("Gmu14", "Gmu14", 1855, 1855, 1875)
    Gsig14 = RooRealVar("Gsig14", "Gsig14", 6.59, 0, 100)
    Gauss14 = RooGaussian("Gauss14", "Gaussian", x, Gmu14, Gsig14)

    #Crystal Ball parameters
    Cmu14 = RooRealVar("Cmu", "Cmu", 1865.07, 1855, 1875)
    Csig14 = RooRealVar("Csig", "Csig", 10.65, 0, 100)
    aL14 = RooRealVar("aL", "aL", 1.77, -10, 10)
    nL14 = RooRealVar("nL", "nL", 9.5, -10, 10)
    aR14 = RooRealVar("aR", "aR", 3.73, -10, 10)
    nR14 = RooRealVar("nR", "nR", 4.34, -10, 10)
    Crystal14 = RooCrystalBall("Crystal", "Crystal Ball", x, Cmu14, Csig14, aL14, nL14, aR14, nR14)

    frac14 = RooRealVar("frac14", "frac14", 0.567, 0, 1)

    #Chebychev parameters
    a14 = RooRealVar("a14", "a14", -0.4, -5, 5)
    chebychev14 = RooChebychev("Chebychev14", "Chebychev", x, RooArgList(a14))

    #Normalisation of Background and Signal
    Nsig14 = RooRealVar("Nsig14", "Nsig14", 0.95*ttree.GetEntries(), 0, ttree.GetEntries())
    Nbkg14 = RooRealVar("Nbkg14", "Nbkg14", 0.05*ttree.GetEntries(), 0, ttree.GetEntries())

    signal = RooAddPdf("signal", "signal", RooArgList(Gauss14, Crystal14), RooArgList(frac14))
    model_14 = {
        "total": RooAddPdf("total", "Total", RooArgList(signal, chebychev14), RooArgList(Nsig14, Nbkg14)), # extended likelihood
        "signals": {
            Gauss14.GetName(): Gauss14.GetTitle(),
            Crystal14.GetName(): Crystal14.GetTitle(),
        },
        "backgrounds": {
            chebychev14.GetName(): chebychev14.GetTitle()
        }
    }
    model_14["total"].fitTo(data, RooFit.Save(), RooFit.Extended(1), RooFit.Minos(0))
    chi2, pull_mean, pull_std = plot(x, data, model_14, nbins=100, setlogy=False, save_to= f"fit_{options.meson}_{options.polarity}_data_{options.year}_{options.size}_{options.model}")
    Nsig = Nsig14.getValV()
    Nsig_err = Nsig14.getError()
    Nbkg = Nbkg14.getValV()
    Nbkg_err = Nbkg14.getError()

    return

def gauss_crystal_exp(x,data,ttree):
    """Model 15:
    Signal - gauss and crystall ball function
    Background - exponential

    Args:
        x (RooRealVar): invariant mass data of D0_MM
        data (RooDataSet): Assigning data to variable
        ttree (tree): data in the file
    """
    
    #Gaussian parameters
    Gmu15 = RooRealVar("Gmu15", "Gmu15", 1855, 1855, 1875)
    Gsig15 = RooRealVar("Gsig15", "Gsig15", 6.59, 0, 100)
    Gauss15 = RooGaussian("Gauss15", "Gaussian", x, Gmu15, Gsig15)

    #Crystal Ball parameters
    Cmu15 = RooRealVar("Cmu", "Cmu", 1865.07, 1855, 1875)
    Csig15 = RooRealVar("Csig", "Csig", 10.65, 0, 100)
    aL15 = RooRealVar("aL", "aL", 1.77, -10, 10)
    nL15 = RooRealVar("nL", "nL", 9.5, -10, 10)
    aR15 = RooRealVar("aR", "aR", 3.73, -10, 10)
    nR15 = RooRealVar("nR", "nR", 4.34, -10, 10)
    Crystal15 = RooCrystalBall("Crystal", "Crystal Ball", x, Cmu15, Csig15, aL15, nL15, aR15, nR15)

    frac15 = RooRealVar("frac15", "frac15", 0.567, 0, 1)

    #Chebychev parameters
    a15 = RooRealVar("a15", "a15", -0.4, -5, 5)
    chebychev14 = RooChebychev("Chebychev14", "Chebychev", x, RooArgList(a15))

    #Normalisation of Background and Signal
    Nsig15 = RooRealVar("Nsig15", "Nsig15", 0.95*ttree.GetEntries(), 0, ttree.GetEntries())
    Nbkg15 = RooRealVar("Nbkg15", "Nbkg15", 0.05*ttree.GetEntries(), 0, ttree.GetEntries())

    signal = RooAddPdf("signal", "signal", RooArgList(Gauss15, Crystal15), RooArgList(frac15))
    model_15 = {
        "total": RooAddPdf("total", "Total", RooArgList(signal, chebychev15), RooArgList(Nsig15, Nbkg15)), # extended likelihood
        "signals": {
            Gauss14.GetName(): Gauss14.GetTitle(),
            Crystal14.GetName(): Crystal14.GetTitle(),
        },
        "backgrounds": {
            chebychev15.GetName(): chebychev15.GetTitle()
        }
    }
    model_15["total"].fitTo(data, RooFit.Save(), RooFit.Extended(1), RooFit.Minos(0))
    chi2, pull_mean, pull_std = plot(x, data, model_15, nbins=100, setlogy=False, save_to= f"fit_{options.meson}_{options.polarity}_data_{options.year}_{options.size}_{options.model}")
    Nsig = Nsig15.getValV()
    Nsig_err = Nsig15.getError()
    Nbkg = Nbkg15.getValV()
    Nbkg_err = Nbkg15.getError()

    return





if options.model==14:
    gauss_crystal_chebychev(x,data,ttree)
elif options.model==15:
    gauss_crystal_exp(x,data,ttree)

def file_writer():
    file = open(f"tightcuts_{options.model}_{options.meson}_{options.polarity}_{options.year}.txt", "w")
    text = 'N_sig: ' + str(Nsig) + ', N_sig_err: ' + str(Nsig_err) + ', Chi2: ' + str(chi2) + ', pull mean: ' + str(pull_mean) + ', pull std dev: ' + str(pull_std) + 'N_bkg: ' + str(Nbkg) + 'N_bkg_err: ' + str(Nbkg_err)
    file.write(text)
    file.close

print(ttree.GetEntries())
    