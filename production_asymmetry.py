
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
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# - - - - - - - FUNCTIONS - - - - - - - #

def parse_arguments():
    '''
    Parses the arguments needed along the code. Arguments:
    
    --year      Used to specify the year at which the data was taken the user is interested in.
                The argument must be one of: [16, 17, 18]. These referr to 2016, 2017 & 2018, respectively.
    --size      Used to specify the amount of events the user is interested in analysing.
                The argument must be one of: [large, small, medium, 1-8]. The integers specify the number of root
                files to be read in. Large is equivalent to 8. Medium is equivalent to 4. Small takes 200000 events.
    --path      Used to specify the directory in which the output files should be written. It is not required,
                in the case it is not specified, the default path is the current working directory.
    --input     Used to specify the directory in which the input data should be found. It is not required,
                in the case it is not specified, the default path is the current working directory. 
    --seedval   Used to set a set a seed for the blinding.
    
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
        "--input",
        type=dir_path,
        required=False,
        default=os.getcwd(),
        help="flag to set the path where the input data should be found"
    )
    parser.add_argument(
    "--blind",
    type=str,
    required=True,
    choices=["y", "Y", "n", "N"],
    help="flag to set whether the asymmetry should be blinded or not (y/n)."
)
    parser.add_argument(
    "--seedval",
    type=int,
    required=False,
    default=12,
    help="input any number or word to set a seed for the blinding."
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
        
def read_from_file(meson, polarity, bin_num):
    '''
    Opens a .txt files and reads the values of the signal normalization constant and its uncertainty.
    
    Returns these two values.
    '''
    with open(f'{options.input}/{bin_num}/yields_{meson}_{polarity}_{options.year}_{options.size}_bin{bin_num}.txt') as f:
        for line in f:
            currentline = line.split(",")
            Nsig = float(currentline[0])
            Nsig_err = float(currentline[1])
        f.close()

    return Nsig, Nsig_err

def get_yield(bin_num):
    '''
    Gets all the normalization yields, and their uncertainties, necessary to calculate the raw asymmetries.
    This takes into account both D0 and D0bar and both magnet polarities.
    
    Returns all the signal normalization constants, together with their uncertainties.
    '''
    
    yield_D0_up = read_from_file("D0", "up", bin_num)
    yield_D0bar_up = read_from_file("D0bar", "up", bin_num)
    yield_D0_down = read_from_file("D0", "down", bin_num)
    yield_D0bar_down = read_from_file("D0bar", "down", bin_num)
   
    return yield_D0_up[0], yield_D0_up[1], yield_D0bar_up[0], yield_D0bar_up[1], yield_D0_down[0], yield_D0_down[1], yield_D0bar_down[0], yield_D0bar_down[1]


def A_Det():
      # Detector asymmetry for D0(Kpi) = A_raw(D+->K-pi+pi+) - A_raw(D+->Ks0pi+) - A(Ks0)
        # Running the program AsymmetryTools from GitLab over PuTTY outputs the following:
        if options.year == 16:
            A_kspi_up = -0.87534228056
            A_kspi_err_up = 0.265077797764

            A_kpipi_up = -1.35398359189
            A_kpipi_err_up = 0.13115828851

            A_kspi_down = -0.355750007642
            A_kspi_err_down = 0.247579432594

            A_kpipi_down = -0.637694362926
            A_kpipi_err_down = 0.123796822258

        elif options.year == 17:
            A_kspi_up = -0.654235263918
            A_kspi_err_up = 0.273509295656

            A_kpipi_up = -1.38335612668
            A_kpipi_err_up = 0.121595927374

            A_kspi_down = -0.126746550532
            A_kspi_err_down = 0.269402552773

            A_kpipi_down = -1.02345488078
            A_kpipi_err_down = 0.127466759335

        elif options.year == 18:
            A_kspi_up = -0.942058542057
            A_kspi_err_up = 0.270644276803

            A_kpipi_up = -1.5471233568
            A_kpipi_err_up = 0.131391285254

            A_kspi_down = -0.277769785338
            A_kspi_err_down = 0.288008079473

            A_kpipi_down = -1.27619403857
            A_kpipi_err_down = 0.129960950228

        # Production asymmetry for K0 found from a paper
        A_k0 = 0.054
        A_k0_err = 0.014

        # Calculating the detector asymmetry and the associated uncertainty    
        A_det_up = A_kpipi_up - A_kspi_up - A_k0
        A_det_up_error = np.sqrt(((A_kpipi_err_up)**2+(A_kspi_err_up)**2+(A_k0_err)**2))
        # Detector asymmetry is the same for all models
        

        A_det_down = A_kpipi_down - A_kspi_down - A_k0
        A_det_down_error =  np.sqrt(((A_kpipi_err_down)**2+(A_kspi_err_down)**2+(A_k0_err)**2))
        

        return A_det_up, A_det_down, A_det_down_error, A_det_up_error



##################################################
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

def output_results(A_raw, A_raw_err, A_raw_up, A_raw_up_err, A_raw_down, A_raw_down_err, bin_num):
    '''
    This function takes all the necessary values and outputs them to the screen in a nicely formatted way.
    It also outputs them to a .txt file, written in the directory established by the user.
    '''
    
    asymmetry = str(round(A_raw, 3)) + ' +/- ' + str(round(A_raw_err, 3)) + ' (stat) +/- '
    print(f'The 20{options.year} raw asymmetry of bin {bin_num} is:', asymmetry)
    
    array = np.array([A_raw, A_raw_err, A_raw_up, A_raw_up_err, A_raw_down, A_raw_down_err])
    np.savetxt(f"{options.path}/asymmetries_{options.year}_{options.size}_bin{bin_num}.txt", array, delimiter=',')

def blind(A_raw, A_raw_error, string):
    ''' 
    This function blinds the values of the raw asymmetry in order to prevent bias experimenter's bias, the unintended biasing of a result in a particular direction.
    A raw is randomly multiplied by 1 or -1 then a pseudo-random constant is added. The pseudo-random constant is randomly chosen from within 3 standard deviations of a standard Gaussian distribution.
    The blinded result is reproudicble because the randomness is seeded.
    '''
    factorchoice = [-1, 1]
    factor = np.random.choice(factorchoice)
    np.random.seed(options.seedval)
    # Draw random samples from a normal (Gaussian) distribution, with a mean of 0, std of 1 and then the sample is within 3std of the mean.
    constant = np.random.normal(0, 3*A_raw_error, 1)[0]
    Ablind = factor * A_raw + constant

    if string == 'up':
        print('The blinded Magup raw asymmetry is ', round(Ablind*100, 2))
    if string == 'down':
        print('The blinded MagDown raw asymmetry is ', round(Ablind*100, 2))
    
    return 100*Ablind


def production_asymm(A_raw_up, A_raw_down, A_raw_up_err, A_raw_down_err, A_det_up, A_det_down, A_det_down_err, A_det_up_err):
    
    A_prod_up = A_raw_up - A_det_up
    A_prod_down = A_raw_down - A_det_down

    A_prod = (A_prod_up + A_prod_down) / 2

    
    A_prod_up_err = (A_raw_up_err**2 + A_det_up_err**2)**(0.5)
    A_prod_down_err = (A_raw_down_err**2 + A_det_down_err**2)**(0.5)
    A_prod_err = ((A_prod_up_err**2+A_prod_down_err**2)**(0.5))/2


    return A_prod, A_prod_up, A_prod_down, A_prod_up_err, A_prod_down_err, A_prod_err


def plot_histogram(val, result, uncertainty, text):
    
    initial = text[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(lw=0.5, ls='--', alpha=0.2)
    ax.hist(A_raw, 15, histtype='stepfilled', ec='k', color='wheat', alpha=0.6)
    ax.set_xlabel(f"{text}Asymmetry")
    ax.set_ylabel("Bins")
    ax.set_title(f"{text}Asymmetry Distribution")
    textstr = '\n \n'.join((
        fr'Weighted $ A_{initial} =$',
        r'${:.3f} \pm {:.3f}$'.format(result, uncertainty)))

    plt.text(-4.5, 30, textstr, horizontalalignment='center', verticalalignment='center', bbox=dict(boxstyle='square,pad=1', fc='w', ec='k', alpha=1))
    fig.savefig(f"{options.path}/1D_asymmetry_distribution_{options.year}_{options.size}.pdf", dpi=600)
    plt.close(fig)




def integrated_asym(val, err):
    
    weight = err**-2
    numerator = np.sum(val*weight)
    denominator = np.sum(weight)
    weighted_mean = numerator/denominator
    uncertainty = np.sum(weight)**-0.5

    return weighted_mean, uncertainty





def plot_2Dhistogram(val, text):
    val = np.asarray(val).ravel()
    length = int(np.sqrt(len(val)))
    data = np.asarray(val).reshape(length, length)
    mpl.rcParams['pdf.fonttype'] = 42  # Make sure text is saved as editable text in PDF
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams.update({'axes.unicode_minus' : False})
    sns.set()
    ax2 = sns.heatmap(data, vmax=0, vmin=-2.5, annot=True, annot_kws={'size': 8}, cmap ='YlOrBr_r')

    ax2.invert_yaxis()
    ax2.set_title(f"{text}Asymmetry distribution")
    ax2.set_xlabel("x index")
    ax2.set_ylabel("y index")
    fig2 = ax2.get_figure()
    fig2.savefig(f"{options.path}/2D_asymmetry_distribution_{options.year}_{options.size}.pdf", dpi=600)
    plt.clf()
    plt.close(fig2)

# - - - - - - - MAIN CODE - - - - - - - #

options = parse_arguments()

for j in range(0,10):
    for i in range (0,10):
        bin_num = str(j)+str(i)
        # get normalization yield from desired model 
        N_D0_up, N_D0_up_err, N_D0bar_up, N_D0bar_up_err, N_D0_down, N_D0_down_err, N_D0bar_down, N_D0bar_down_err = get_yield(bin_num)

        # get raw asymmetries for main model
        A_raw_up, A_raw_up_err = calculate_raw_asymmetry(N_D0_up, N_D0bar_up, 1, N_D0_up_err, N_D0bar_up_err)
        A_raw_down, A_raw_down_err = calculate_raw_asymmetry(N_D0_down, N_D0bar_down, 1, N_D0_down_err, N_D0bar_down_err)
        #A_raw = (A_raw_up + A_raw_down) / 2
        A_raw_err = ((A_raw_up_err**2 + A_raw_down_err**2)**0.5) /2

        A_det_up, A_det_down, A_det_down_error, A_det_up_error = A_Det()



        if options.blind == 'y' or options.blind == 'Y':
            A_unblind_up = A_raw_up
            # Asymmetry is blinded
            A_raw_up = blind(A_raw_up, A_raw_up_err, 'up')
            A_unblind_down = A_raw_down
            # Asymmetry is blinded
            A_raw_down = blind(A_raw_down, A_raw_down_err, 'down')

            A_raw = (A_raw_down + A_raw_up) /2
            # output results
            output_results(A_raw, A_raw_err, A_raw_up, A_raw_up_err, A_raw_down, A_raw_down_err, bin_num)

        else:
            # output results
            output_results(A_raw, A_raw_err, A_raw_up, A_raw_up_err, A_raw_down, A_raw_down_err, bin_num)
        
        # Calculates A_det

        
if options.blind == 'y' or options.blind == 'Y':
    # Calculates the integrated production asymmetry for both and unblinded asymmetries
    print("------------------------------")
    print("------------------------------")
    print("------------------------------")

    #A raw blinded and unblinded Up
    blind_integrated_raw_up = integrated_asym(A_raw_up , A_raw_up_err)
    unblind_integrated_raw_up = integrated_asym(A_unblind_up, A_raw_up_err)
    
    

    #A raw blinded and unblinded Down
    blind_integrated_raw_down = integrated_asym(A_raw_down , A_raw_down_err)
    unblind_integrated_raw_down = integrated_asym(A_unblind_down, A_raw_down_err)
    
    #A raw error
    A_raw_err_up = unblind_integrated_raw_up[1]
    A_raw_err_down = unblind_integrated_raw_down[1]

    #A raw blinded and unblinded Total
    A_raw_blinded = (blind_integrated_raw_up[0] + blind_integrated_raw_down[0]) / 2
    A_raw_unblinded = (unblind_integrated_raw_up[0] + unblind_integrated_raw_down[0])/2
    A_raw_err = ((A_raw_err_up**2 + A_raw_err_up**2)**0.5) /2

    


    # A_prod, A_prod_up, A_prod_down, A_prod_up_err, A_prod_down_err, A_prod_err: Blinded and Unblinded are given from production_asymm
    # production_asymm requires A_raw_up, A_raw_down, A_raw_up_err, A_raw_down_err, A_det_up, A_det_down, A_det_down_err, A_det_up_err
    blinded_prod = production_asymm(blind_integrated_raw_up[0], blind_integrated_raw_down[0], A_raw_err_up, A_raw_err_down, A_det_up, A_det_down, A_det_down_error, A_det_up_error)
    Unblinded_prod = production_asymm(unblind_integrated_raw_up[0], unblind_integrated_raw_down[0], A_raw_err_up, A_raw_err_down, A_det_up, A_det_down, A_det_down_error, A_det_up_error)

    print("------------------------------")
    print("------------------------------")
    print("------------------------------")

    print('The MagUp detector asymmetry is: (', round(A_det_up, 2), '+/-', round(A_det_up_error, 2), ') %')
    print('The MagDown detector asymmetry is: (', round(A_det_down, 2), '+/-', round(A_det_down_error, 2), ') %')

    print("------------------------------")
    print("------------------------------")
    print("------------------------------")


    #print all unblinded
    print(f"The 20{options.year} unblinded integrated raw MagUp asymmetry is: ", round(unblind_integrated_raw_up[0],3), " +/-", round(unblind_integrated_raw_up[1], 3), ') %')
    print(f"The 20{options.year} unblinded integrated raw MagDown asymmetry is: ", round(unblind_integrated_raw_down[0],3), " +/-", round(unblind_integrated_raw_down[1], 3), ') %')
    print(f"The 20{options.year} unblinded integrated total raw asymmetry is: ", round(A_raw_unblinded,3), " +/-", round(A_raw_err, 3), ') %')
    print(f"The 20{options.year} unblinded integrated prod MagUp asymmetry is: ", round(Unblinded_prod[1],3), " +/-", round(Unblinded_prod[3], 3), ') %')
    print(f"The 20{options.year} unblinded integrated prod MagDown asymmetry is: ", round(Unblinded_prod[2],3), " +/-", round(Unblinded_prod[4], 3), ') %')
    print(f"The 20{options.year} unblinded integrated total prod asymmetry is: ", round(Unblinded_prod[0],3), " +/-", round(Unblinded_prod[5], 3), ') %')

    print("------------------------------")
    print("------------------------------")
    print("------------------------------")


    #print all blinded
    print(f"The 20{options.year} blinded integrated raw MagUp asymmetry is: ", round(blind_integrated_raw_up[0],3), " +/-", round(blind_integrated_raw_up[1], 3), ') %')
    print(f"The 20{options.year} blinded integrated raw MagDown asymmetry is: ", round(blind_integrated_raw_down[0],3), " +/-", round(blind_integrated_raw_down[1], 3), ') %')
    print(f"The 20{options.year} blinded integrated total raw asymmetry is: ", round(A_raw_blinded,3), " +/-", round(A_raw_err, 3), ') %')
    print(f"The 20{options.year} blinded integrated prod MagUp asymmetry is: ", round(blinded_prod[1],3), " +/-", round(blinded_prod[3], 3), ') %')
    print(f"The 20{options.year} blinded integrated prod MagDown asymmetry is: ", round(blinded_prod[2],3), " +/-", round(blinded_prod[4], 3), ') %')
    print(f"The 20{options.year} blinded integrated total prod asymmetry is: ", round(blinded_prod[0],3), " +/-", round(blinded_prod[5], 3), ') %')


    print("------------------------------")
    print("------------------------------")
    print("------------------------------")



    plot_histogram(Unblinded_prod[0], Unblinded_prod[0], Unblinded_prod[5], "Unblinded Production")
    plot_2Dhistogram(Unblinded_prod[0], "Unblinded Production")

else:
    integrated_asymm = integrated_asym(Unblinded_prod[0] , A_p_error)
    print('The MagUp detector asymmetry is: (', round(A_det_up, 2), '+/-', round(A_det_up_error, 2), ') %')
    print('The MagDown detector asymmetry is: (', round(A_det_down, 2), '+/-', round(A_det_down_error, 2), ') %')
    print(f"The 20{options.year} integrated production asymmetry is: ", round(integrated_asymm[0],3), " +/-", round(Unblinded_prod[5], 3), ') %')

    plot_histogram(Unblinded_prod[0], integrated_asymm[0], Unblinded_prod[5], "Production")
    plot_2Dhistogram(Unblinded_prod[0], "Production")