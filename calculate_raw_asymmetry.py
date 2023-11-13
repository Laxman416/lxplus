"""
calculate_asymmetry.py

This code is used to process the signal normalization yields and obtain the raw asymmetries. It also uses the input from using another signal model to obtain the systematic uncertainty due to the fit model. It finally outputs the results obtained both to the secreen and to a .txt file.
The year of interest and size of the data to be analysed must be specified using the required flags --year --size.There also are the flags --input and --path which are not required. These are used to specify the directory where the input data is located and where the output file should be written, respectively. By default it is set to be the current working directory.
This code is  inspired on the work of Camille Jarvis-Stiggants and Michael England. The code has been completely rewritten and reorganised, and some features have been added to add flexibility to the code, but some of the original functions have been used here as well.

Author: Marc Oriol Pérez (marc.oriolperez@student.manchester.ac.uk)
Last edited: 16th September 2023
"""

# - - - - - - IMPORT STATEMENTS - - - - - - #

import random
import os
import argparse
import numpy as np
import re

# - - - - - - - FUNCTIONS - - - - - - - #


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
    "--model",
    type=int,
    choices=[15,17],
    required=True,
    help="flag to set the path where the input data should be found"
)

parser.add_argument(
    "--input",
    type=dir_path(),
    required=False,
    default=os.getcwd(),
    help="flag to set the path where the input data should be found"
)

options = parser.parse_args()

def dir_path(string):
    '''
    Checks if a given string is the path to a directory.
    If affirmative, returns the string. If negative, gives an error.
    '''
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
        
def read_from_file(meson, polarity):
    '''
    Opens a .txt files and reads the values of the signal normalization constant and its uncertainty.
    
    Returns these two values.
    '''
    with open(f'{options.input}/tightcuts_{options.model}_{meson}_{polarity}_{options.year}_{options.size}.txt') as f:
        file_content = f.read()
        list_file_content = re.findall(r'[\d.]+', file_content)
        Nsig = float(list_file_content[0])
        Nsig_err = float(list_file_content[1])
        f.close()
            
    return Nsig, Nsig_err

def get_yield():
    '''
    Gets all the normalization yields, and their uncertainties, necessary to calculate the raw asymmetries.
    This takes into account both D0 and D0bar and both magnet polarities.
    
    Returns all the signal normalization constants, together with their uncertainties.
    '''
    
    yield_D0_up = read_from_file("D0", "up")
    yield_D0bar_up = read_from_file("D0bar", "up")
    yield_D0_down = read_from_file("D0", "down")
    yield_D0bar_down = read_from_file("D0bar", "down")
    
    return yield_D0_up[0], yield_D0_up[1], yield_D0bar_up[0], yield_D0bar_up[1], yield_D0_down[0], yield_D0_down[1], yield_D0bar_down[0], yield_D0bar_down[1]

def calculate_raw_asymmetry(norm_D0, norm_D0bar, bin_width, N_D0_err, N_D0bar_err):
    '''
    It takes the normalization yields for D0 and D0bar as arguments and then calculates the raw
    asymmetries from these. It also propagates the uncertainties.
    
    Returns both the asymmetry and its uncertainty as a percentage.
    '''
    
    N_D0 = abs(norm_D0)/abs(bin_width)
    N_D0bar = abs(norm_D0bar)/abs(bin_width)
    
    A = (N_D0 - N_D0bar)/(N_D0 + N_D0bar)
    A_err = 2*(((N_D0bar**2)*(N_D0_err**2) + (N_D0**2)*(N_D0bar_err**2))**0.5)*((N_D0 + N_D0bar)**(-2))
          
    return 100*A, 100*A_err

def output_results(A_raw, A_raw_err, A_raw_up, A_raw_up_err, A_raw_down, A_raw_down_err):
    '''
    This function takes as arguments all the necessary values and outputs them to the screen in a nicely formatted way.
    It also outputs them to a .txt file, written in the directory established by the user.
    '''
    
    print('The MagUp raw asymmetry is: (', round(A_raw_up, 3), '+/-', round(A_raw_up_err, 3), ') %')
    print('The MagDown raw asymmetry is: (', round(A_raw_down, 3), '+/-', round(A_raw_down_err, 3), ') %')
    
    asymmetry = str(round(A_raw, 3)) + ' +/- ' + str(round(A_raw_err, 3)) + ' (stat) +/- '
    print(f'The 20{options.year} raw asymmetry of bin is:', asymmetry)
    
    array = np.array([A_raw, A_raw_err, A_raw_up, A_raw_up_err, A_raw_down, A_raw_down_err])
    np.savetxt(f"{options.path}/asymmetries_{options.year}_{options.size}.txt", array, delimiter=',')

    return
    
def calculate_prod_asymmetry(A_raw_up, A_raw_up_err, A_raw_down, A_raw_down_err, year = options.year):
    if year == "16":
        A_kspi_up = -0.87534228056
        A_kspi_err_up = 0.265077797764

        A_kpipi_up = -1.35398359189
        A_kpipi_err_up = 0.13115828851

        A_kspi_down = -0.355750007642
        A_kspi_err_down = 0.247579432594

        A_kpipi_down = -0.637694362926
        A_kpipi_err_down = 0.123796822258

    elif year == "17":
        A_kspi_up = -0.654235263918
        A_kspi_err_up = 0.273509295656

        A_kpipi_up = -1.38335612668
        A_kpipi_err_up = 0.121595927374

        A_kspi_down = -0.126746550532
        A_kspi_err_down = 0.269402552773

        A_kpipi_down = -1.02345488078
        A_kpipi_err_down = 0.127466759335

    elif year == "18":
        A_kspi_up = -0.942058542057
        A_kspi_err_up = 0.270644276803

        A_kpipi_up = -1.5471233568
        A_kpipi_err_up = 0.131391285254

        A_kspi_down = -0.277769785338
        A_kspi_err_down = 0.288008079473

        A_kpipi_down = -1.27619403857
        A_kpipi_err_down = 0.129960950228




    # production asymmetry for K0 found from a paper

    A_k0 = 0.054
    A_k0_err = 0.014
    
    # detector asymmetry is the same for both models

    A_det_up_local = A_kpipi_up - A_kspi_up - A_k0
    A_det_up_err_local = np.sqrt(((A_kpipi_err_up)**2+(A_kspi_err_up)**2+(A_k0_err)**2))



    A_det_down_local = A_kpipi_down - A_kspi_down - A_k0
    A_det_down_err_local =  np.sqrt(((A_kpipi_err_down)**2+(A_kspi_err_down)**2+(A_k0_err)**2))



    A_prod_up_local = A_raw_up - A_det_up
    A_prod_down_local = A_raw_down - A_det_down

    A_prod_local = (A_prod_up + A_prod_down) / 2

    
    A_prod_up_err_local = (A_raw_up_err**2 + A_det_up_err**2)**(0.5)
    A_prod_down_err_local = (A_raw_down_err**2 + A_det_down_err**2)**(0.5)
    A_prod_err_local = ((p_err_up**2+p_err_down**2)**(0.5))/2

    return A_det_up_local, A_det_up_err_local, A_det_down_local, A_det_down_err_local,A_prod_up_local,A_prod_down_local, A_prod_local, A_prod_up_err_local, A_prod_down_err_local, A_prod_err_local#

# - - - - - - - MAIN CODE - - - - - - - #

def output_results_prod(A_det_up, A_det_up_err, A_det_down, A_det_down_err,A_prod_up,A_prod_down, A_prod, A_prod_up_err, A_prod_down_err, A_prod_err):
        print('The MagUp detector asymmetry is: (', round(A_det_up, 2), '+/-', round(A_det_up_err, 2), ') %')
        print('The MagDown detector asymmetry is: (', round(A_det_down, 2), '+/-', round(A_det_down_err, 2), ') %')

        print('The MagUp prod asymmetry is: (', round(A_prod_up, 2), '+/-', round(A_prod_up_err, 2), ') %')
        print('The MagDown prod asymmetry is: (', round(A_prod_down, 2), '+/-', round(A_prod_down_err, 2), ') %')

        prod_asymmetry = str(round(A_prod, 3)) + ' +/- ' + str(round(A_prod_err, 3)) + ' (stat) +/- '
        print(f'The 20{options.year} raw asymmetry of bin is:', prod_asymmetry)

        array = np.array([A_prod, A_prod_err, A_prod_up, A_prod_up_err, A_prod_down, A_prod_down_err])
        np.savetxt(f"{options.path}/prod_asymmetries_{options.year}_{options.size}.txt", array, delimiter=',')
        
        return





# get normalization yield from desired model 
N_D0_up, N_D0_up_err, N_D0bar_up, N_D0bar_up_err, N_D0_down, N_D0_down_err, N_D0bar_down, N_D0bar_down_err = get_yield()

# get raw asymmetries for main model
A_raw_up, A_raw_up_err = calculate_raw_asymmetry(N_D0_up, N_D0bar_up, 1, N_D0_up_err, N_D0bar_up_err)
A_raw_down, A_raw_down_err = calculate_raw_asymmetry(N_D0_down, N_D0bar_down, 1, N_D0_down_err, N_D0bar_down_err)
A_raw = (A_raw_up + A_raw_down) / 2
A_raw_err = ((A_raw_up_err**2 + A_raw_down_err**2)**0.5) /2

# output results
output_results(A_raw, A_raw_err, A_raw_up, A_raw_up_err, A_raw_down, A_raw_down_err)
A_det_up, A_det_up_err, A_det_down, A_det_down_err,A_prod_up,A_prod_down, A_prod,A_prod_up_err, A_prod_down_err, A_prod_err = calculate_prod_asymmetry(A_raw_up, A_raw_up_err, A_raw_down, A_raw_down_err)
output_results_prod(A_det_up, A_det_up_err, A_det_down, A_det_down_err,A_prod_up,A_prod_down, A_prod, A_prod_up_err, A_prod_down_err, A_prod_err)