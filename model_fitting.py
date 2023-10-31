"""
model_fitting.py

This code is used to fit the data in one of the bims. It then returns the relevant plots of the best fit to the data and a .txt file containing the values and errors on the normalization constant of both signal and background, the mean and standard deviation of the pull distribution and the reduced chi squared value. It can both fit the data using a binned approach or an unbinned one. The model used consists of a Gaussian function and a Crystal Ball function for the signal, and a Chevychev polynomial for the background. Some of the parameters are fixed to be the same as the best-fit values obtained during the global fit in order to obtain better convergence.
The year of interest, size of the data, meson of interest and polarity to be analysed must be specified using the required flags --year --size --meson --polarity. It is also required to specify the bin to be analyzed using the flag --bin, and if the fit should be done on the binned data or the unbinned data using the flag --binned_fit. There also are the flags --input --parameteers_path and --path, which are not required. These are used to specify the directory where the input data is located, where the global best-fit parameters can be found and where the output should be written, respectively. By default it is set to be the current working directory.

Author: Marc Oriol Pérez (marc.oriolperez@student.manchester.ac.uk)
Last edited: 16th September 2023
"""

# - - - - - - IMPORT STATEMENTS - - - - - - #

import ROOT
import argparse
import os
from utils import plot
import numpy as np
from ROOT import TChain, RooRealVar, RooDataSet, RooGaussian, RooCrystalBall, RooChebychev, RooAddPdf, RooArgList, RooFit, RooArgSet, RooDataHist

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
lower_boundary = 1800
upper_boundary = 1900

if options.binned_fit=="y" or options.binned_fit=="Y":
    binned = True
else:
    binned = False

# Reads in the fit parameters generated by fit_global.py, these will be either for a binned/unbinned fit depending on if fit_global.py was ran as a binned fit or not
parameters = np.loadtxt(f"{options.parameters_path}/fit_parameters.txt", delimiter=',')

# Read data
ttree = TChain("D02Kpi_Tuple/DecayTree")
ttree.Add(f"{options.input}/{options.meson}_{options.polarity}_data_{options.year}_{options.size}_clean.root")

ttree.SetBranchStatus("*", 0)
ttree.SetBranchStatus("D0_MM", 1)
D0_M = RooRealVar("D0_MM", "D0 mass / [MeV]", 1810, 1910) # D0_MM - invariant mass

if binned:
   mD0_bins = D0_M.setBins(numbins)

# Define variables for signal model
mu = RooRealVar("mu", "mu", 1865, 1862, 1868)
Gsig = RooRealVar("sigma", "sigma", 6.59, 0, 100)
Gauss = RooGaussian("Gauss", "Gaussian", D0_M, mu, Gsig)

Csig = RooRealVar("Csig", "Csig", 10.65, 0, 100)
aL = RooRealVar("aL", "aL", 1.77, -10, 10)
nL = RooRealVar("nL", "nL", 9.5, -10, 10)
aR = RooRealVar("aR", "aR", parameters[6])
nR = RooRealVar("nR", "nR", parameters[7])
Crystal = RooCrystalBall("Crystal", "Crystal Ball", D0_M, mu, Csig, aL, nL, aR, nR)

frac = RooRealVar("frac", "frac", 0.575, 0, 1)
# Define variables for background model
a = RooRealVar("a", "a", parameters[8])
chebychev = RooChebychev("Chebychev", "Chebychev", D0_M, RooArgList(a))
# Define Normalisation constants for signal and background
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
    # Creates the histogram for the meson by converting the TTree D0_MM data inside the TChain to a TH1(base class of ROOT histograms)
    # TTree.Draw plots a histogram with name D0_Hist and given bin parameters and saves it to memory using: >>
    ttree.Draw(f"D0_MM>>D0_Hist({numbins},{lower_boundary},{upper_boundary})")
    # D0_Hist recalled from memory and saved to the local variable
    D0_Hist = ROOT.gPad.GetPrimitive("D0_Hist")
    # Creating Binned container sets using RooDataHist
    Binned_data = RooDataHist("Binned_data", "Binned Data Set", RooArgList(D0_M), D0_Hist)


    result = model["total"].fitTo(Binned_data, RooFit.Save(True), RooFit.Extended(True))

    frame = D0_M.frame(RooFit.Name(""))
    legend_entries = dict()

    Binned_data.plotOn(frame, ROOT.RooFit.Name("remove_me_A"))
    model["total"].plotOn(
        frame,
        RooFit.Name(model["total"].GetName()),
        RooFit.LineWidth(5),
        RooFit.LineColor(ROOT.kAzure),
    )
    pull_hist = frame.pullHist()

    legend_entries[model["total"].GetName()] = {"title": model["total"].GetTitle(), "style": "l"}

    # plot signal components
    signal_colours = [ROOT.kRed, ROOT.kSpring, ROOT.kAzure + 7, ROOT.kOrange + 7]
    signal_line_styles = [2, 7, 9, 10]
    i = 0
    for name, title in model["signals"].items():
        legend_name = f"S{i}"
        model["total"].plotOn(
            frame,
            ROOT.RooFit.Components(name),
            ROOT.RooFit.Name(legend_name),
            ROOT.RooFit.LineWidth(4),
            ROOT.RooFit.LineColor(signal_colours[i % len(signal_colours)]),
            ROOT.RooFit.LineStyle(signal_line_styles[i % len(signal_line_styles)]),
        )
        legend_entries[legend_name] = {"title": title, "style": "l"}
        i += 1

    # plot background components
    background_colours = [ROOT.kMagenta + 2, ROOT.kPink + 7, ROOT.kMagenta + 4]
    background_line_styles = [5, 8, 6]
    i = 0
    for name, title in model["backgrounds"].items():
        legend_name = f"B{i}"
        model["total"].plotOn(
            frame,
            ROOT.RooFit.Components(name),
            ROOT.RooFit.Name(legend_name),
            ROOT.RooFit.LineWidth(4),
            ROOT.RooFit.LineColor(background_colours[i % len(background_colours)]),
            ROOT.RooFit.LineStyle(background_line_styles[i % len(background_line_styles)]),
        )
        legend_entries[legend_name] = {"title": title, "style": "l"}
        i += 1
    
    # plot data points on top again
    Binned_data.plotOn(frame, ROOT.RooFit.Name("remove_me_B"))
    frame.remove("remove_me_A")
    frame.remove("remove_me_B")
    frame.addTH1(D0_Hist, "PE")
    legend_entries[D0_Hist.GetName()] = {"title": D0_Hist.GetTitle(), "style": "PE"}


    frame.SetYTitle(f"Entries MeV/c^{{2}})")

    c = ROOT.TCanvas("fit", "fit", 900, 800)
    fit_pad = ROOT.TPad("fit_pad", "fit pad", 0, 0.2, 1.0, 1.0)
    fit_pad.Draw()
    fit_pad.cd()
    # R.gPad.SetLeftMargin(0.15)
    # Draw the plot on a canvas
    # frame.GetYaxis().SetTitleOffset(1.4)
    # frame.GetYaxis().SetMaxDigits(len(str(int(plotted_data.GetMaximum()/1000)))+1)  # TODO : a better fix than this?
    frame.Draw()
    
    frame.GetXaxis().SetLabelSize(0)
    frame.GetXaxis().SetTitleSize(0)
    frame.Draw()
    title_size = frame.GetYaxis().GetTitleSize() * 2.5
    label_size = frame.GetYaxis().GetLabelSize() * 2.5

    # plot_type + total + signals + backgrounds + data
    nlines = 1 + 1 + len(model["signals"]) + len(model["backgrounds"]) + 1
    xwidth = 0.4
    ywidth = 0.03 * nlines
    legend = ROOT.TLegend(
        0.6, 0.89 - ywidth, 0.6 + xwidth, 0.89, "#bf{#it{LHCb Unofficial}}"
    )
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    for key, val in legend_entries.items():
        legend.AddEntry(key, val["title"], val["style"])
    legend.Draw("same")


    pull_frame = D0_M.frame(ROOT.RooFit.Title(" "))
    pull_TH1 = ROOT.TH1D("pull_TH1", "pull_TH1", numbins)
    bad_pull_TH1 = ROOT.TH1D("bad_pull_TH1", "bad_pull_TH1", numbins)
    for i in range(pull_hist.GetN()):
        if pull_hist.GetPointY(i) > 5:
            pull_TH1.SetBinContent(i + 1, 5)
            bad_pull_TH1.SetBinContent(i + 1, 5)
        elif pull_hist.GetPointY(i) < -5:
            pull_TH1.SetBinContent(i + 1, -5)
            bad_pull_TH1.SetBinContent(i + 1, -5)
        elif pull_hist.GetPointY(i) == 0:
            pull_TH1.SetBinContent(i + 1, 0)
            bad_pull_TH1.SetBinContent(i + 1, 0)
        else:
            pull_TH1.SetBinContent(i + 1, pull_hist.GetPointY(i))
            if abs(pull_hist.GetPointY(i)) >= 3:
                bad_pull_TH1.SetBinContent(i + 1, pull_hist.GetPointY(i))

    bad_pull_TH1.SetFillColor(ROOT.kRed)
    pull_frame.addTH1(pull_TH1, "bar min0")
    pull_frame.addTH1(bad_pull_TH1, "bar min0")

    c.cd(0)
    pull_pad = ROOT.TPad("pull_pad", "pull pad", 0.0, 0.0, 1.0, 0.31)
    pull_pad.SetBottomMargin(0.4)
    pull_pad.Draw()
    pull_pad.cd()


    pull_frame.GetXaxis().SetLabelSize(label_size)
    pull_frame.GetXaxis().SetTitleSize(title_size)
    pull_frame.GetXaxis().SetTitleOffset(1)
    pull_frame.GetYaxis().SetRangeUser(-5, 5)
    pull_frame.GetYaxis().SetNdivisions(5)
    pull_frame.GetYaxis().SetTitle("Pull [#sigma]")
    pull_frame.GetYaxis().SetLabelSize(label_size)
    pull_frame.GetYaxis().SetTitleSize(title_size)
    pull_frame.GetYaxis().SetTitleOffset(0.39)

    line = ROOT.TLine(D0_M.getMin(), 0, D0_M.getMax(), 0)
    pull_frame.Draw()
    line.Draw("same")

    three = ROOT.TLine(D0_M.getMin(), 3, D0_M.getMax(), 3)
    nthree = ROOT.TLine(D0_M.getMin(), -3, D0_M.getMax(), -3)
    three.SetLineColor(ROOT.kRed)
    three.SetLineStyle(9)
    nthree.SetLineColor(ROOT.kRed)
    nthree.SetLineStyle(9)
    three.Draw("same")
    nthree.Draw("same")

    c.SaveAs(f"{str(output_directory)}/D0_fit_ANA.root")
    c.SaveAs(f"{str(output_directory)}/D0_fit_ANA.C")
    c.SaveAs(f"{str(output_directory)}/D0_fit_ANA.pdf")
    c.SaveAs(f"{str(output_directory)}/D0_fit_ANA.jpg")

else:
    unbinned_data = RooDataSet("data", "Data", ttree, RooArgSet(D0_M))
    model["total"].fitTo(unbinned_data, RooFit.Save(), RooFit.Extended(1), RooFit.Minos(0))
    # Generate plots
    chi2, pull_mean, pull_std = plot(D0_M, unbinned_data, model, nbins=numbins, setlogy=False, save_to=f'{options.path}/{options.meson}_{options.polarity}_{options.year}_{options.size}', plot_type=f"20{options.year} Mag{(options.polarity).title()}", meson=options.meson)

# Write out results
file = open(f"{options.path}/yields_{options.meson}_{options.polarity}_{options.year}_{options.size}.txt", "w")
text = str(Nsig.getValV()) + ', ' + str(Nsig.getError()) + ', ' + str(Nbkg.getValV()) + ', ' + str(Nbkg.getError()) + ', ' + str(chi2) + ', ' + str(pull_mean) + ', ' + str(pull_std)
file.write(text)
file.close

print(ttree.GetEntries())