import ROOT
import numpy as np
import uproot
import os
from ROOT import TChain, RooRealVar, RooDataSet, RooGaussian, RooCrystalBall, RooBifurGauss, RooChebychev, RooAddPdf, RooArgList, RooFit, RooArgSet, RooGenericPdf, RooJohnson, RooUnblindUniform, RooDataHist
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR) # mute warnings
ROOT.gROOT.SetBatch(True)

tree = "D02Kpi_Tuple/DecayTree"
filename_D0_up = 'D0_up_data_16_8_clean'
filename_D0bar_up = 'D0bar_up_data_16_8_clean'
numbins = 50
lower_boundary = 1800
upper_boundary = 1900

# Selects invariant mass (D0_MM) of DO
ttree_D0_up = TChain(tree)
ttree_D0_up.Add(f"{filename_D0_up}.root")
ttree_D0_up.SetBranchStatus("*", 0)
ttree_D0_up.SetBranchStatus("D0_MM", 1)

# Selects invariant mass (D0_MM) of DObar
ttree_D0bar_up = TChain(tree)
ttree_D0bar_up.Add(f"{filename_D0bar_up}.root")
ttree_D0bar_up.SetBranchStatus("*", 0)
ttree_D0bar_up.SetBranchStatus("D0_MM", 1)

D0_M = RooRealVar("D0_MM", "D0 mass / [MeV/c*c]", 1810, 1910)

# Creating the histograms for D0 and D0bar by converting the TTree D0_MM data inside the TChain to a TH1(base class of ROOT histograms)
# TTree.Draw plots a histogram with name D0_Up_Hist and given bin parameters and saves it to memory using: >>
ttree_D0_up.Draw(f"D0_MM>>D0_Up_Hist({numbins},{lower_boundary},{upper_boundary})")
# D0_Up_Hist recalled from memory and saved to the local variable
D0_Up_Hist = ROOT.gPad.GetPrimitive("D0_Up_Hist")
ttree_D0bar_up.Draw(f"D0_MM>>D0bar_Up_Hist({numbins},{lower_boundary},{upper_boundary})")
D0bar_Up_Hist = ROOT.gPad.GetPrimitive("D0bar_Up_Hist")

# Creating Binned container sets using RooDataHist
Binned_D0_up = RooDataHist("Binned_D0_up", "Binned D0 Up Data", RooArgList(D0_M), D0_Up_Hist)
Binned_D0bar_up = RooDataHist("Binned_D0bar_up", "Binned D0bar Up Data", RooArgList(D0_M), D0bar_Up_Hist)

# Model Gaussian
mean = RooRealVar("mean", "mean", 1865, 1850, 1880)
sigma = RooRealVar("sigma", "sigma", 6.59, 0, 100)
gaussian = RooGaussian("gauss", "gauss", D0_M, mean, sigma)

# Model CrystalBall
Cmu = RooRealVar("Cmu", "Cmu", 1865.07, 1855, 1875)
Csig = RooRealVar("Csig", "Csig", 10.65, 0, 100)
aL = RooRealVar("aL", "aL", 1.77, -10, 10)
nL = RooRealVar("nL", "nL", 9.5, -10, 10)
aR = RooRealVar("aR", "aR", 3.73, -10, 10)
nR = RooRealVar("nR", "nR", 4.34, -10, 10)
crystal = RooCrystalBall("Crystal", "Crystal Ball", D0_M, Cmu, Csig, aL, nL, aR, nR)




binned_sample = ROOT.RooCategory("binned_sample", "binned_sample")
simultaneous_pdf = ROOT.RooSimultaneous("simultaneous", "simultaneous", binned_sample)




binned_sample.defineType("Binned_D0_up_sample")
# Model Signal
frac_D0_up = RooRealVar("frac_D0_up", "frac D0 up", 0.567, 0, 1)
signal_D0_up = RooAddPdf("signal_D0_up", "signal D0 up", RooArgList(gaussian, crystal), RooArgList(frac_D0_up))
# Model Chebyshev
a0 = RooRealVar("a0", "a0", -0.4, -5, 5)
background_D0_up = RooChebychev("chebyshev", "chebyshev", D0_M, RooArgList(a0))

# Generate normalization variables
Nsig_D0_up = RooRealVar("Nsig_D0_up", "Nsig D0 up", 0.95*Binned_D0_up.numEntries(), 0, Binned_D0_up.numEntries())
Nbkg_D0_up = RooRealVar("Nbkg_D0_up", "Nbkg D0 up", 0.05*Binned_D0_up.numEntries(), 0, Binned_D0_up.numEntries())
# Generate model
model_D0_up = RooAddPdf("model_D0_up", "model D0 up", [signal_D0_up, background_D0_up], [Nsig_D0_up, Nbkg_D0_up])
simultaneous_pdf.addPdf(model_D0_up, "Binned_D0_up_sample")





binned_sample.defineType("Binned_D0bar_up_sample")
# Model Signal
frac_D0bar_up = RooRealVar("frac D0bar up", "frac D0bar up", 0.567, 0, 1)
signal_D0bar_up = RooAddPdf("signa_D0bar_up", "signal D0bar up", RooArgList(gaussian, crystal), RooArgList(frac_D0bar_up))
# Model Chebyshev
a0 = RooRealVar("a0", "a0", -0.4, -5, 5)
background_D0bar_up = RooChebychev("chebyshev", "chebyshev", D0_M, RooArgList(a0))

# Generate normalization variables
Nsig_D0bar_up = RooRealVar("Nsig_D0bar_up", "Nsig D0bar up", 0.95*Binned_D0bar_up.numEntries(), 0, Binned_D0bar_up.numEntries())
Nbkg_D0bar_up = RooRealVar("Nbkg_D0bar_up", "Nbkg D0bar up", 0.05*Binned_D0bar_up.numEntries(), 0, Binned_D0bar_up.numEntries())
# Generate model
model_D0bar_up = RooAddPdf("model_D0bar_up", "model D0bar up", [signal_D0bar_up, background_D0bar_up], [Nsig_D0bar_up, Nbkg_D0bar_up])
simultaneous_pdf.addPdf(model_D0bar_up, "Binned_D0bar_up_sample")



imports = [ROOT.RooFit.Import("Binned_D0_up_sample", Binned_D0_up), ROOT.RooFit.Import("Binned_D0bar_up_sample", Binned_D0bar_up)]
simultaneous_data = RooDataHist("simultaneous_data", "simultaneous data", RooArgList(D0_M), ROOT.RooFit.Index(binned_sample), *imports)


fitResult = simultaneous_pdf.fitTo(simultaneous_data, PrintLevel=-1, Save=True, Extended=True)
fitResult.Print()