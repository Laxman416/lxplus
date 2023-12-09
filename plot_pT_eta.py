import os
import argparse
import numpy as np
from lhcbstyle import LHCbStyle
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
import awkward as ak
import sys
from iminuit import Minuit
import matplotlib.pyplot as plt
import mplhep; mplhep.style.use("LHCb2")
from scipy.stats import chi2
from matplotlib.patches import Rectangle


# - - - - - - - FUNCTIONS - - - - - - - #

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
    --path      Used to specify the directory in which the output files should be written. It is not required,
                in the case it is not specified, the default path is the current working directory.
    --input     Used to specify the directory in which the input data should be found. It is not required,
                in the case it is not specified, the default path is the current working directory.
    --bin_path  Used to specify the directory in which the binning scheme should be found. It is not required,
                in the case it is not specified, the default path is the current working directory.
    
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
        "--bin_path",
        type=dir_path,
        required=False,
        default=os.getcwd(),
        help="flag to set the path where the binning scheme should be found"
    )
    parser.add_argument(
        "--asymm_path",
        type=dir_path,
        required=False,
        default=os.getcwd(),
        help="flag to set the path where the production asymmetry for each bin should be found"
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["pT","eta"],
        required=True,
        help="flag to set whether a binned or an unbinned should be performed (y/n)"
    )
    return parser.parse_args()

def dir_path(string):
    '''
    Checks if a given string is the path to a directory.
    If affirmative, returns the string. If negative, gives an error.
    '''
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def read_asymmetry_values():
    with open(f'{args.path}/final_asymmetries_{args.scheme}_{args.year}_{args.size}.txt') as f:
        lines = f.readlines()
        binned_asymm = float(lines[0])
        binned_asymm_error = float(lines[1])
        unbinned_asymm = float(lines[2])
        unbinned_asymm_error = float(lines[3])
        f.close()
    return binned_asymm, binned_asymm_error , unbinned_asymm, unbinned_asymm_error

# - - - - - - - MAIN BODY - - - - - - - #
args = parse_arguments()

binned_asymm, binned_asymm_error , unbinned_asymm, unbinned_asymm_error = read_asymmetry_values()

asymmetry = []
asymmetry_error = []
for j in range(0,10):
    bin_num = str(j)
    with open(f'{args.asymm_path}/asymmetries_{args.year}_{args.size}_bin{bin_num}.txt') as f:
        lines = f.readlines()
        A_prod = float(lines[0])
        A_prod_err = float(lines[1])
    f.close()
    asymmetry.append(A_prod)
    asymmetry_error.append(A_prod_err)



x_value = []
x_value_error = []



file_path = f"{args.bin_path}/{args.year}_{args.size}_{args.scheme}_bins.txt"  # Replace with the path to your file

# Open the file in read mode
with open(file_path, 'r') as file:
    # Read all lines from the file and store them in a list
    bin_lines = [float(line.strip()) for line in file.readlines()]

print(bin_lines)

for i in range(0,10):
    # Get center of bin and width of bin as error
    x_value_indivual = (bin_lines[i]+bin_lines[i+1])/2
    x_value_error_indivual = (bin_lines[i+1]) - ((bin_lines[i]+bin_lines[i+1])/2)
    x_value_error.append(x_value_error_indivual)
    x_value.append(x_value_indivual)


if args.scheme == 'pT':
    x_value = [x / 1000 for x in x_value] # in Gev
    x_value_error = [x / 1000 for x in x_value_error] # in Gev


##########
bins = len(x_value)

# initialise globally so can print after
store = {}
def get_chi2(c: float):
    chi2 = 0.
    ndof = -1 # 1 for c

    for i in range(bins):
        if asymmetry_error[i] == 0:
            continue

        pol0 = c
        residual = (asymmetry[i] - pol0) / asymmetry_error[i]

        chi2 += residual * residual
        ndof += 1
    
    store["chi2"] = chi2
    store["ndof"] = ndof

    return chi2



m = Minuit(get_chi2, c=0) # 0 is initial guess
result = m.migrad()
values = list(m.values)
errors = list(m.errors)
prob = 1 - chi2.cdf(store["chi2"], store["ndof"])

result_string = "\n".join((
    rf"$c = {values[0]:.3f} \pm {errors[0]:.3f}, m=0$",
    rf"$\chi^{{2}} / $nDOF$ = {store['chi2']:.2f} / {store['ndof']}$",
    rf"$p = {prob*100:.1f} \%$"
))

_, ax = plt.subplots()

ax.errorbar(x_value, asymmetry, yerr=asymmetry_error, fmt="k.", label="data")
ax.plot(x_value,[values[0] for _ in range(bins)],color="b", label="fit")
ax.text(0.55, 0.1, result_string, transform=ax.transAxes, fontsize=30)
ax.set_xlabel(r"$p_T$ [GeV$/c$]")
ax.set_ylabel("Production asymmetry")
ax.legend(loc="best")
plt.savefig("asym_flatness.pdf", bbox_inches="tight")


###################

# Plotting
fig, ax = plt.subplots()
if args.scheme == 'pT':
    ax.set_xlabel(r'$p_{T}$ [GeV$c^{-1}$]', fontsize = 50)
elif args.scheme == 'eta':
    ax.set_xlabel(r'$\eta$', fontsize = 50)

ax.set_ylabel(r'$A_{\mathrm{prod}}$ [%]', fontsize = 50)
ax.tick_params(axis='both', which='both', labelsize=30)

line1 = ax.axhline(binned_asymm, color='blue', linestyle='dashed', linewidth=5)
fill1 = plt.axhspan(binned_asymm-binned_asymm_error, binned_asymm+binned_asymm_error, color='blue', alpha=0.35, lw=0)

line2 = ax.axhline(unbinned_asymm, color='red', linestyle='solid', linewidth=5)
fill2 = plt.axhspan(unbinned_asymm-unbinned_asymm_error, unbinned_asymm+unbinned_asymm_error, color='red', alpha=0.35, lw=0)

Data = ax.errorbar(x_value, asymmetry, yerr=asymmetry_error,xerr=x_value_error, fmt='o', capsize=5, color = 'black', label = 'Data')
extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

Fit = ax.axhline(values[0], color='green', linestyle=':', linewidth=5)

if args.scheme == 'pT':
    ax.legend([(line1,fill1),(line2,fill2),Data,Fit,extra],[r'Average result over $p_{T}$ bins', r'Bin integrated result','Data','Fit',f'{result_string}'])#, loc='upper right')
elif args.scheme == 'eta':
    ax.legend([(line1,fill1),(line2,fill2),Data,Fit,extra],[r'Average result over $\eta$ bins', r'Bin integrated result','Data','Fit',f'{result_string}'])#,='upper right')

if args.scheme == 'pT':
    plt.savefig(f'{args.path}/pT_Asymm_{args.year}_{args.size}.pdf', bbox_inches = "tight")
elif args.scheme == 'eta':
    plt.savefig(f'{args.path}/eta_Asymm_{args.year}_{args.size}.pdf', bbox_inches = "tight")



plt.show()
