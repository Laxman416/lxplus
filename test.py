"""
plot_phase_space.py

This code plots relevant histograms to show the event distribution across the phase space and it also shows the edges of the bins.
The year of interest, size of the data, polarity and meson to be analysed must be specified using the required flags --year --size --polarity --meson. There also are the flags --input --path and --bin_path, which are not required. These are used to specify the directory where the input data is located, where the binning scheme can be found and where the output file should be written, respectively. By default it is set to be the current working directory.
It outputs several pdf files containing the relevant histograms.

Author: Marc Oriol Pérez (marc.oriolperez@student.manchester.ac.uk)
Last edited: 15th September 2023
"""
# - - - - - - IMPORT STATEMENTS - - - - - - #

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
        "--input",
        type=dir_path,
        required=False,
        default=os.getcwd(),
        help="flag to set the path where the input data should be found"
    )
    parser.add_argument(
        "--bin_path",
        type=dir_path,
        required=False,
        default=os.getcwd(),
        help="flag to set the path where the binning scheme should be found"
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

def chunk_list(input_list, chunk_size):
    """Split a list into chunks of a specified size."""
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


# - - - - - - - MAIN BODY - - - - - - - #
args = parse_arguments()

# Import data

bins = np.loadtxt(f"{args.bin_path}/{args.year}_{args.size}_bins.txt", delimiter=',')
bins[0] = bins[0]/1000 #GeV
viridis = colormaps['YlOrRd']
newcolors = viridis(np.linspace(0, 1, 25))
newcmp = ListedColormap(newcolors)



p = [] 
e = [] # Contain all bin lines on eta

for index in np.arange(0,10):
    if index!=0:
        # Co-ordinates of bin lines on pT
        p.append(bins[0,index])
    for j in np.arange(0,10):
        if j!=0:
            # Co-ordinates of bin lines on eta
            e.append(bins[index+1, j])

points = [] # Co-ordinates of points in each bin
for i in np.arange(0, 9): # for each pT line
    for j in np.arange(0,9): # for each eta line in pT 
        l = 9*i + j # to go through the eta list.
        value = 0.01  # to remove a certain value so its a above or below bin line
        value2 = 0.01  # to remove a certain value so its a above or below bin line
        #print(p[i]- value,e[l]- value2) #might need to decrease 0.001 if bin size becomes very small
        points.append([p[i]-value,e[l]-value2])
        if j != 0: # dont need to add value to first line. Only subtract
            if j % 8 == 0: # to get the coordinates of the top row of bins. need to add value to eta.
                #print(p[i]-value,e[l]+value2)
                points.append([p[i]-value,e[l]+value2]) # top row of bins
    if i == 8:
        for j in np.arange(0,9):
            l = 9*i + j
            value = 0.01
            value2 = 0.01
            #print(p[i]+value,e[l]-value2)
            points.append([p[i]+value,e[l]-value2]) # to get right hand side bins
            if j != 0: 
                if j % 8 == 0: # to get top right corner bin
                    #print(p[i]+value,e[l]+value2)
                    points.append([p[i]+value,e[l]+value2])
if (len(points)) == 100:
    print(len(points))
else:
    print("Error: Expected 100 datapoints only got ", (len(points)))
    sys.exit(1)


e_bins = [] # binning depending on x
e_bins = chunk_list(e, 9) #only has 9 values so far in each list. need to append bottom and top. Need to append eta values of 2 and 5
y_max = 5 #Range of eta is from 2 to 5. #also y_max
y_min = 2 #y_min
x_min = 2
x_max = 10
modified_e_bins = [[y_min] + sublist + [y_max] for sublist in e_bins]
print(len(modified_e_bins[0]))


x_values = []
y_values = []

for coord in points:
    x_values.append(coord[0])
    y_values.append(coord[1])
asymmetry = np.random.rand(len(x_values))


x = chunk_list(x_values, 10) #split into 10 lists of 10
y= chunk_list(y_values,10)   #split into 10 lists of 10
A = chunk_list(asymmetry,10) #split into 10 lists of 10


# # Loop: Histogram 2 to 9
fig, ax = plt.subplots()

h2d = ax.hist2d(
    np.true_divide(x[0], 1),
    np.true_divide(y[0], 1),
    weights=np.true_divide(A[0], 1),
    bins=[1, modified_e_bins[0]],
    cmap=newcmp,
    cmin=0,
    cmax=1,
    range=[[x_min, x[0][0]], [y_min, y_max]]
)
for i in range(len(x)):
    if i == 0:
        continue
    q = i-1
    # Plot the histogram for each iteration
    h2d = ax.hist2d(
        np.true_divide(x[i], 1),
        np.true_divide(y[i], 1),
        weights=np.true_divide(A[i], 1),
        bins=[1, modified_e_bins[i]],
        cmap=newcmp,
        cmin=0,
        cmax=1,
        range=[[x[q][0], x[i][0]], [y_min, y_max]]
    )


# Plot the 10th histogram - need to do seperately due to x_max in range
h2d = ax.hist2d(
    np.true_divide(x[9], 1),
    np.true_divide(y[9], 1),
    weights=np.true_divide(A[9], 1),
    bins=[1, modified_e_bins[9]],
    cmap=newcmp,
    cmin=0,
    cmax=1,
    range=[[x[9][0], x_max], [y_min, y_max]]
)




ax.set_xlabel(r'$p_{T}$ [GeV/c]')
ax.set_ylabel(r'$\eta$')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
cbar = fig.colorbar(h2d[3], ax=ax, label='Asymmetry')



for index in np.arange(0,10):
    if index!=0:
        ax.axvline(bins[0,index], ymin=0, ymax=1, color='blue', linestyle = 'dashdot')
    for j in np.arange(0,10):
        if j!=0:
            ax.axhline(bins[index+1, j], xmin=(bins[0,index]-2)/8, xmax=(bins[0,index+1]-2)/8, color='blue', linestyle = 'dashdot')
    
plt.savefig(f'{args.path}/test_{args.meson}_{args.polarity}_{args.year}_{args.size}.pdf')