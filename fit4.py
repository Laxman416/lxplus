import ROOT
from ROOT import TChain, RooRealVar, RooDataSet, RooGaussian, RooCrystalBall, RooBifurGauss, RooChebychev, RooAddPdf, RooArgList, RooFit, RooArgSet, RooGenericPdf, RooJohnson, RooUnblindUniform, RooExponential, RooExpPoly
import uproot
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from pylab import cm
from utils2 import plot
import argparse
import os

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
    type=int,
    choices=[14,15,16,17,18],
    required=True,
    help="flag to set the model."
)
parser.add_argument(
    "--size",
    type=str,
    choices=["large", "medium", "small", "1", "2", "3", "4", "5", "6", "7", "8"],
    required=True,
    help="flag to set the size."
)
parser.add_argument(
    "--binned",
    type=str,
    choices=["y", "n"],
    required=True,
    help="flag to set the data will be binned."
)
parser.add_argument(
        "--path",
        type=str,
        required=False,
        default=os.getcwd(),
        help="flag to set the path where the output files should be written to"
    )

options = parser.parse_args()

ttree = TChain("D02Kpi_Tuple/DecayTree")
ttree.Add(f"/afs/cern.ch/work/l/lseelan/lxplus/{options.path}/{options.meson}_{options.polarity}_data_{options.year}_{options.size}_clean.root")

ttree.SetBranchStatus("*", 0)
ttree.SetBranchStatus("D0_MM", 1)
x = RooRealVar("D0_MM", "D0 mass / [MeV]", 1820, 1910) # D0_MM - invariant mass
data = RooDataSet("data", "Data", ttree, RooArgSet(x))

def dir_path(string):
    '''
    Checks if a given string is the path to a directory.
    If affirmative, returns the string. If negative, gives an error.
    '''
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
        





def gauss_crystal_chebychev(x,data,ttree,meson=options.meson,polarity=options.polarity,year=options.year,size=options.size,model=options.model,binned=options.binned):
    """Model 14:
    Signal - gauss and crystall ball function
    Background - chebychev

    Args:
        x (RooRealVar): invariant mass data of D0_MM
        data (RooDataSet): Assigning data to variable
        ttree (tree): data in the file
    """
    
    #Gaussian parameters
    Gmu14 = RooRealVar("Gmu14", "Gmu14", 1855, 1835, 1875)
    Gsig14 = RooRealVar("Gsig14", "Gsig14", 6.59, 0, 100)
    Gauss14 = RooGaussian("Gauss14", "Gaussian", x, Gmu14, Gsig14)

    #Crystal Ball parameters
    Cmu14 = RooRealVar("Cmu", "Cmu", 1865.07, 1855, 1875)
    Csig14 = RooRealVar("Csig", "Csig", 10.65, 0, 100)
    aL14 = RooRealVar("aL", "aL", 1.77, -10, 10)
    nL14 = RooRealVar("nL", "nL", 9.5, 0, 30)
    aR14 = RooRealVar("aR", "aR", 3.73, -10, 10)
    nR14 = RooRealVar("nR", "nR", 4.34, 0, 30)
    Crystal14 = RooCrystalBall("Crystal", "Crystal Ball", x, Cmu14, Csig14, aL14, nL14, aR14, nR14)

    frac14 = RooRealVar("frac14", "frac14", 0.567, 0, 1)

    #Chebychev parameters
    a14 = RooRealVar("a14", "a14", -0.4, -5, 5)
    chebychev14 = RooChebychev("Chebychev14", "Chebychev", x, RooArgList(a14))

    #Normalisation of Background and Signal
    Nsig14 = RooRealVar("Nsig14", "Nsig14", 0.95*ttree.GetEntries(), 0, ttree.GetEntries())
    Nbkg14 = RooRealVar("Nbkg14", "Nbkg14", 0.05*ttree.GetEntries(), 0, ttree.GetEntries())

    if binned=='n':
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
        chi2, pull_mean, pull_std = plot(x, data, model_14, nbins=100, setlogy=False, save_to= f"fit_model{model}_{meson}_{polarity}_{year}_{size}")
        Nsig = Nsig14.getValV()
        Nsig_err = Nsig14.getError()
        Nbkg = Nbkg14.getValV()
        Nbkg_err = Nbkg14.getError()
        
        #Saving file
        file = open(f"tightcuts_{model}_{meson}_{polarity}_{year}_{size}.txt", "w")
        text = 'N_sig: ' + str(Nsig) + ', N_sig_err: ' + str(Nsig_err) + ', Chi2: ' + str(chi2) + ', pull mean: ' + str(pull_mean) + ', pull std dev: ' + str(pull_std) + 'N_bkg: ' + str(Nbkg) + 'N_bkg_err: ' + str(Nbkg_err)
        file.write(text)
        file.close
        return

    elif binned=='y':
        print()

def gauss_crystal_exp(x,data,ttree,meson=options.meson,polarity=options.polarity,year=options.year,size=options.size,model=options.model,binned=options.binned):
    """Model 15:
    Signal - gauss and crystall ball function
    Background - exponential

    Args:
        x (RooRealVar): invariant mass data of D0_MM
        data (RooDataSet): Assigning data to variable
        ttree (tree): data in the file
    """
    
    #Gaussian parameters
    Gmu15 = RooRealVar("Gmu15", "Gmu15", 1865, 1835, 1875)
    Gsig15 = RooRealVar("Gsig15", "Gsig15", 6.42, 0, 100)
    Gauss15 = RooGaussian("Gauss15", "Gaussian", x, Gmu15, Gsig15)

    #Crystal Ball parameters
    Cmu15 = RooRealVar("Cmu", "Cmu", 1865.07, 1855, 1875)
    Csig15 = RooRealVar("Csig", "Csig", 10.24, 0, 20)
    aL15 = RooRealVar("aL", "aL", 1.70, -10, 10)
    nL15 = RooRealVar("nL", "nL", 10, 14, 30)
    aR15 = RooRealVar("aR", "aR", 2.34, -10, 10)
    nR15 = RooRealVar("nR", "nR", 0, 25, 100)
    Crystal15 = RooCrystalBall("Crystal", "Crystal Ball", x, Cmu15, Csig15, aL15, nL15, aR15, nR15)

    frac15 = RooRealVar("frac15", "frac15", 0.567, 0, 1)

    #exponential parameters
    c15 = RooRealVar("c15","c15",-0.008,-1,0)
    exponential15 = RooExponential("Expo15", "Exponential",x,c15)

    #Normalisation of Background and Signal
    Nsig15 = RooRealVar("Nsig15", "Nsig15", 0.95*ttree.GetEntries(), 0, ttree.GetEntries())
    Nbkg15 = RooRealVar("Nbkg15", "Nbkg15", 0.05*ttree.GetEntries(), 0, ttree.GetEntries())

    if binned=='n':
        signal = RooAddPdf("signal", "signal", RooArgList(Gauss15, Crystal15), RooArgList(frac15))
        model_15 = {
            "total": RooAddPdf("total", "Total", RooArgList(signal, exponential15), RooArgList(Nsig15, Nbkg15)), # extended likelihood
            "signals": {
                Gauss15.GetName(): Gauss15.GetTitle(),
                Crystal15.GetName(): Crystal15.GetTitle(),
            },
            "backgrounds": {
                exponential15.GetName(): exponential15.GetTitle()
            }
        }
        model_15["total"].fitTo(data, RooFit.Save(), RooFit.Extended(1), RooFit.Minos(0))
        chi2, pull_mean, pull_std = plot(x, data, model_15, nbins=100, setlogy=False, save_to= f"fit_model{model}_{meson}_{polarity}_{year}_{size}")
        Nsig = Nsig15.getValV()
        Nsig_err = Nsig15.getError()
        Nbkg = Nbkg15.getValV()
        Nbkg_err = Nbkg15.getError()

        #Saving File
        file = open(f"tightcuts_{model}_{meson}_{polarity}_{year}_{size}.txt", "w")
        text = 'N_sig: ' + str(Nsig) + ', N_sig_err: ' + str(Nsig_err) + ', Chi2: ' + str(chi2) + ', pull mean: ' + str(pull_mean) + ', pull std dev: ' + str(pull_std) + 'N_bkg: ' + str(Nbkg) + 'N_bkg_err: ' + str(Nbkg_err)
        file.write(text)
        file.close
    elif binned=='y':
        print()
    return

def gauss_johnson_exp(x,data,ttree,meson=options.meson,polarity=options.polarity,year=options.year,size=options.size,model=options.model,binned=options.binned):

    #gaussian
    Gmu16 = RooRealVar("Gmu16", "Gmu16", 1865, 1835, 1875)
    Gsig16 = RooRealVar("Gsig16", "Gsig16", 6.42, 0, 100)
    Gauss16 = RooGaussian("Gauss16", "Gaussian", x, Gmu16, Gsig16)

    #Johnson
    Jmu16 = RooRealVar("Jmu16","Jmu16",1866, 1835, 1875)
    Jlambda16 = RooRealVar("Jlambda16","Jlambda16",5, 0, 40) 
    Jgamma16 = RooRealVar("Jgamma16","Jgamma16",5, 0, 20)
    Jdelta16 = RooRealVar("Jdelta16","Jdelta16",4, 0, 20)
    Johnson16 = RooJohnson("Johnson16","Johnson",x,Jmu16,Jlambda16,Jgamma16,Jdelta16)

    frac16 = RooRealVar("frac16", "frac16", 0.5, 0.2, 0.8)

    #exponential parameters
    c16 = RooRealVar("c16","c16",-0.008,-1,0)
    exponential16 = RooExponential("Expo16", "Exponential",x,c16)

    #Normalisation of Background and Signal
    Nsig16 = RooRealVar("Nsig16", "Nsig16", 0.95*ttree.GetEntries(), 0, ttree.GetEntries())
    Nbkg16 = RooRealVar("Nbkg16", "Nbkg16", 0.05*ttree.GetEntries(), 0, ttree.GetEntries())

    if binned=='n':
        signal = RooAddPdf("signal", "signal", RooArgList(Gauss16, Johnson16), RooArgList(frac16))
        model_16 = {
            "total": RooAddPdf("total", "Total", RooArgList(signal, exponential16), RooArgList(Nsig16, Nbkg16)), # extended likelihood
            "signals": {
                Gauss16.GetName(): Gauss16.GetTitle(),
                Johnson16.GetName(): Johnson16.GetTitle(),
            },
            "backgrounds": {
                exponential16.GetName(): exponential16.GetTitle()
            }
        }
        model_16["total"].fitTo(data, RooFit.Save(), RooFit.Extended(1), RooFit.Minos(0))
        chi2, pull_mean, pull_std = plot(x, data, model_16, nbins=100, setlogy=False, save_to= f"fit_model{model}_{meson}_{polarity}_{year}_{size}")
        Nsig = Nsig16.getValV()
        Nsig_err = Nsig16.getError()
        Nbkg = Nbkg16.getValV()
        Nbkg_err = Nbkg16.getError()

        #Saving File
        file = open(f"tightcuts_{model}_{meson}_{polarity}_{year}_{size}.txt", "w")
        text = 'N_sig: ' + str(Nsig) + ', N_sig_err: ' + str(Nsig_err) + ', Chi2: ' + str(chi2) + ', pull mean: ' + str(pull_mean) + ', pull std dev: ' + str(pull_std) + 'N_bkg: ' + str(Nbkg) + 'N_bkg_err: ' + str(Nbkg_err)
        file.write(text)
        file.close
    elif binned=='y':
        print()
    return



def crystal_crystal_exp(x,data,ttree,meson=options.meson,polarity=options.polarity,year=options.year,size=options.size,model=options.model,binned=options.binned):
    """Model 15:
    Signal - gauss and crystall ball function
    Background - exponential

    Args:
        x (RooRealVar): invariant mass data of D0_MM
        data (RooDataSet): Assigning data to variable
        ttree (tree): data in the file
    """
    
    #Crystal Ball 1 parameters
    Cmu17_1 = RooRealVar("Cmu1", "Cmu1", 1865.07, 1855, 1875)
    Csig17_1 = RooRealVar("Csig1", "Csig1", 10.24, 0, 100)
    aL17_1 = RooRealVar("aL1", "aL1", 1.70, -10, 10)
    nL17_1 = RooRealVar("nL1", "nL1", 8.4, -10, 10)
    aR17_1 = RooRealVar("aR1", "aR1", 2.34, -10, 10)
    nR17_1 = RooRealVar("nR1", "nR1", 8.16, -10, 10)
    Crystal17_1 = RooCrystalBall("Crystal1", "Crystal Ball1", x, Cmu17_1, Csig17_1, aL17_1, nL17_1, aR17_1, nR17_1)

    #Crystal Ball 2 parameters
    Cmu17_2 = RooRealVar("Cmu2", "Cmu2", 1865, 1855, 1875)
    Csig17_2 = RooRealVar("Csig2", "Csig2", 10, 0, 100)
    aL17_2 = RooRealVar("aL2", "aL2", 2, -10, 10)
    nL17_2 = RooRealVar("nL2", "nL2", 8, -10, 10)
    aR17_2 = RooRealVar("aR2", "aR2", 2, -10, 10)
    nR17_2 = RooRealVar("nR2", "nR2", 8, -10, 10)
    Crystal17_2 = RooCrystalBall("Crystal2", "Crystal Ball2", x, Cmu17_2, Csig17_2, aL17_2, nL17_2, aR17_2, nR17_2)

    frac17 = RooRealVar("frac17", "frac17", 0.567, 0.3, 0.7)

    #exponential parameters
    c17 = RooRealVar("c17","c17",-0.008,-1,0)
    exponential17 = RooExponential("Expo17", "Exponential",x,c17)

    #Normalisation of Background and Signal
    Nsig17 = RooRealVar("Nsig17", "Nsig17", 0.95*ttree.GetEntries(), 0, ttree.GetEntries())
    Nbkg17 = RooRealVar("Nbkg17", "Nbkg17", 0.05*ttree.GetEntries(), 0, ttree.GetEntries())

    if binned=='n':
        signal = RooAddPdf("signal", "signal", RooArgList(Crystal17_1, Crystal17_2), RooArgList(frac17))
        model_17 = {
            "total": RooAddPdf("total", "Total", RooArgList(signal, exponential17), RooArgList(Nsig17, Nbkg17)), # extended likelihood
            "signals": {
                Crystal17_1.GetName(): Crystal17_1.GetTitle(),
                Crystal17_2.GetName(): Crystal17_2.GetTitle(),
            },
            "backgrounds": {
                exponential17.GetName(): exponential17.GetTitle()
            }
        }
        model_17["total"].fitTo(data, RooFit.Save(), RooFit.Extended(1), RooFit.Minos(0))
        chi2, pull_mean, pull_std = plot(x, data, model_17, nbins=100, setlogy=False, save_to= f"fit_model{model}_{meson}_{polarity}_{year}_{size}")
        Nsig = Nsig17.getValV()
        Nsig_err = Nsig17.getError()
        Nbkg = Nbkg17.getValV()
        Nbkg_err = Nbkg17.getError()

        #Saving File
        file = open(f"tightcuts_{model}_{meson}_{polarity}_{year}_{size}.txt", "w")
        text = 'N_sig: ' + str(Nsig) + ', N_sig_err: ' + str(Nsig_err) + ', Chi2: ' + str(chi2) + ', pull mean: ' + str(pull_mean) + ', pull std dev: ' + str(pull_std) + 'N_bkg: ' + str(Nbkg) + 'N_bkg_err: ' + str(Nbkg_err)
        file.write(text)
        file.close
    elif binned=='y':
        print()
    return

def gauss_gauss_crystal_exp(x,data,ttree,meson=options.meson,polarity=options.polarity,year=options.year,size=options.size,model=options.model,binned=options.binned):
    
    #gaussian_1
    Gmu18 = RooRealVar("Gmu18", "Gmu18", 1865, 1835, 1875)
    Gsig18 = RooRealVar("Gsig18", "Gsig18", 10, 2, 100)
    Gauss18 = RooGaussian("Gauss18", "Gaussian", x, Gmu18, Gsig18)

    #gaussian_2
    Gsig18_2 = RooRealVar("Gsig18_2", "Gsig18_1", 4, 0, 100)
    Gauss18_2 = RooGaussian("Gauss18_2", "Gaussian_1", x, Gmu18, Gsig18_2)

    #Crystal Ball 1 parameters
    Csig18_1 = RooRealVar("Csig1", "Csig1", 10.24, 0, 100)
    aL18_1 = RooRealVar("aL1", "aL1", 1.70, -10, 10)
    nL18_1 = RooRealVar("nL1", "nL1", 8.4, -10, 10)
    aR18_1 = RooRealVar("aR1", "aR1", 2.34, -10, 10)
    nR18_1 = RooRealVar("nR1", "nR1", 8.16, -10, 10)
    Crystal18_1 = RooCrystalBall("Crystal1", "Crystal Ball1", x, Gmu18, Csig18_1, aL18_1, nL18_1, aR18_1, nR18_1)

    #exponential parameters
    c18 = RooRealVar("c87","c18",-0.008,-1,0)
    exponential18 = RooExponential("Expo18", "Exponential",x,c18)

    #Normalisation of Background and Signal
    Nsig18 = RooRealVar("Nsig18", "Nsig18", 0.95*ttree.GetEntries(), 0, ttree.GetEntries())
    Nbkg18 = RooRealVar("Nbkg18", "Nbkg18", 0.05*ttree.GetEntries(), 0, ttree.GetEntries())

    
    frac18 = RooRealVar("frac18", "frac18", 0.3, 0, 1)
    frac18_1 = RooRealVar("frac18", "frac18", 0.34, 0, 1)

    if binned=='n':
        signal = RooAddPdf("signal", "signal", RooArgList(Gauss18, Gauss18_2, Crystal18_1), RooArgList(frac18,frac18_1))
        model_18 = {
            "total": RooAddPdf("total", "Total", RooArgList(signal, exponential18), RooArgList(Nsig18, Nbkg18)), # extended likelihood
            "signals": {
                Gauss18.GetName(): Gauss18.GetTitle(),
                Gauss18_2.GetName(): Gauss18_2.GetTitle(),
                Crystal18_1.GetName(): Crystal18_1.GetTitle(),

            },
            "backgrounds": {
                exponential18.GetName(): exponential18.GetTitle()
            }
        }
        model_18["total"].fitTo(data, RooFit.Save(), RooFit.Extended(1), RooFit.Minos(0))
        chi2, pull_mean, pull_std = plot(x, data, model_18, nbins=100, setlogy=False, save_to= f"fit_model{model}_{meson}_{polarity}_{year}_{size}")
        Nsig = Nsig18.getValV()
        Nsig_err = Nsig18.getError()
        Nbkg = Nbkg18.getValV()
        Nbkg_err = Nbkg18.getError()

        #Saving File
        file = open(f"tightcuts_{model}_{meson}_{polarity}_{year}_{size}.txt", "w")
        text = 'N_sig: ' + str(Nsig) + ', N_sig_err: ' + str(Nsig_err) + ', Chi2: ' + str(chi2) + ', pull mean: ' + str(pull_mean) + ', pull std dev: ' + str(pull_std) + 'N_bkg: ' + str(Nbkg) + 'N_bkg_err: ' + str(Nbkg_err)
        file.write(text)
        file.close
    elif binned=='y':
        print()
    return

    





if options.model==14:
    gauss_crystal_chebychev(x,data,ttree)
elif options.model==15:
    gauss_crystal_exp(x,data,ttree)
elif options.model==16:
    gauss_johnson_exp(x,data,ttree)
elif options.model==17:
    crystal_crystal_exp(x,data,ttree)
elif options.model==18:
    gauss_gauss_crystal_exp(x,data,ttree)




print(ttree.GetEntries())
    
