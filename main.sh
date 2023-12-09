# # Runs the complete analysis on a set of raw data of D0 meson decays to obtain the asymmetry in local regions of the phase space. The output is stored in the specified directory, and is organized in several directories generated by this same script. Take into account that if a directory with the same name already exists this code might not work as intended. Note that making changes to any of the individual scripts while this code is running can lead to a malfunction.
# # When running the code the output directory, the year the data to be analysed was taken, the size of the data to be analysed and whether or not the data should be binned when fitting must be given as arguments, in that order. The year must be one of: 16, 17 or 18. The size must be one of: small, medium, large, 1, 2, 3, 4, 5, 6, 7 or 8. The binned fitting argument must either be y/Y or n/N.
# # Author: Marc Oriol Pérez (marc.oriolperez@student.manchester.ac.uk)
# # Last modified: 16th September 2023

directory=$1
year=$2
size=$3
binned=$4


# if [[ "$binned" != "y" ]]; then
#     if [[ "$binned" != "Y" ]]; then
#         if [[ "$binned" != "n" ]]; then
#             if [[ "$binned" != "N" ]]; then
#                 echo "WARNING: You did not select a valid option for the binned fit"
#                 echo
#                 echo "An binned fit will be performed"
#                 binned="y"
#             fi   
#         fi
#     fi
# fi

# Create necessary directories to store output



# mkdir $directory
# mkdir $directory"/selected_data"
# mkdir $directory"/binned_data"
# mkdir $directory"/binned_data/binning_scheme"
# mkdir $directory"/model_fitting"
# mkdir $directory"/model_fitting/global"
# mkdir $directory"/model_fitting/local"
# mkdir $directory"/model_fitting/pT"
# mkdir $directory"/model_fitting/eta"
# for ind in {0..99}
# do
#     index=$( printf '%02d' $ind)
#     mkdir $directory"/model_fitting/local/"$index
# done
# for ind in {0..9}
# do
#     index=$( printf '%01d' $ind)
#     mkdir $directory"/model_fitting/pT/"$index
#     mkdir $directory"/model_fitting/eta/"$index
# done
# mkdir $directory"/raw_asymmetry_outcome"
# mkdir $directory"/raw_asymmetry_outcome/chi_squared"
# mkdir $directory"/raw_asymmetry_outcome/raw_asymmetry"
# mkdir $directory"/raw_asymmetry_outcome/raw_asymmetry/pT"
# mkdir $directory"/raw_asymmetry_outcome/raw_asymmetry/eta"
# mkdir $directory"/raw_asymmetry_outcome/raw_asymmetry/local"
# mkdir $directory"/results"
# mkdir $directory"/binned_data/eta"
# mkdir $directory"/binned_data/pT"
# mkdir $directory"/binned_data/local"

# echo "The necessary directories have been created"
# echo


# Run the code

# python selection_of_events.py --year $year --size $size --path $directory"/selected_data"

# echo
# for polar in up down
# do

#     python multiple_candidates.py --year $year --size $size --polarity $polar --path $directory"/selected_data"
# done
# echo "Multiple candidates have been removed"



# python fit_global.py --year $year --size $size --path $directory"/model_fitting/global" --binned_fit $binned --input $directory"/selected_data" --scheme "total"
# for meson in D0 D0bar 
# do 
#     for polar in up down
#     do    
#         python model_fitting.py --year $year --size $size --meson $meson --polarity $polar  --path $directory"/model_fitting/global" --input $directory"/selected_data" --parameters_path $directory"/model_fitting/global" --scheme 'total' --binned_fit $binned
#     done
# done

# echo "The global fit has been completed"
# echo


# python create_binning_scheme.py --year $year --size $size --path $directory"/binned_data/binning_scheme" --input $directory"/selected_data"
# for meson in D0 D0bar
# do 
#     for polar in up down 
#     do    
#         python apply_binning_scheme.py --year $year --size $size --meson $meson --polarity $polar --path $directory"/binned_data" --input $directory"/selected_data" --bin_path $directory"/binned_data/binning_scheme"
#         python plot_phase_space.py --year $year --size $size --meson $meson --polarity $polar --path $directory"/binned_data/binning_scheme" --input $directory"/selected_data" --bin_path $directory"/binned_data/binning_scheme"
#         echo "Ploted 2D graph"
#     done
# done

# echo "The data has been binned"
# echo

# for ind in {0..99}
#         do
#             index=$( printf '%02d' $ind)
#             python fit_global.py --year $year --size $size --path $directory"/model_fitting/local/"$index --binned_fit $binned --input $directory"/binned_data/local" --bin $index --scheme 'pT_eta'
#             echo "Fitted Bin "$index
#         done

for meson in D0 D0bar 
do
   for polar in down up
   do 
        for ind in {75..76}
        do
            index=$( printf '%02d' $ind)
            python model_fitting.py --year $year --size $size --meson $meson --polarity $polar  --path $directory"/model_fitting/local/"$index --input $directory"/binned_data/local" --parameters_path $directory"/model_fitting/local/"$index --bin $index --binned_fit $binned --scheme 'pT_eta'
        done
    done
done


# echo "Local fitting completed"
# echo

# # python analyse_chisquared.py --year $year --size $size --path $directory"/raw_asymmetry_outcome/chi_squared" --input $directory"/model_fitting/local"

# python production_asymmetry.py --year $year --size $size --path $directory"/raw_asymmetry_outcome/raw_asymmetry/local" --input $directory"/model_fitting/" --blind 'Y' --results_path $directory"/results" --scheme 'pT_eta'

# python plot_asymm.py --year $year --size $size --bin_path $directory"/binned_data/binning_scheme" --asymm_path $directory"/raw_asymmetry_outcome/raw_asymmetry/local" --path $directory"/results"

# for ind in {0..9}
#         do
#             index=$( printf '%01d' $ind)
#             python fit_global.py --year $year --size $size --path $directory"/model_fitting/pT/"$index --binned_fit $binned --input $directory"/binned_data/pT" --bin $index --scheme 'pT'
#             python fit_global.py --year $year --size $size --path $directory"/model_fitting/eta/"$index --binned_fit $binned --input $directory"/binned_data/eta" --bin $index --scheme 'eta'
#             echo "Fitted Bin "$index
#         done

# for meson in D0bar D0
# do
#    for polar in down up
#    do 
#         for ind in {0..9}
#         do
#             index=$( printf '%01d' $ind)
            
#             python model_fitting.py --year $year --size $size --meson $meson --polarity $polar  --path $directory"/model_fitting/pT/"$index --input $directory"/binned_data/pT" --parameters_path $directory"/model_fitting/pT/"$index --bin $index --binned_fit $binned --scheme 'pT'
#             python model_fitting.py --year $year --size $size --meson $meson --polarity $polar  --path $directory"/model_fitting/eta/"$index --input $directory"/binned_data/eta" --parameters_path $directory"/model_fitting/eta/"$index --bin $index --binned_fit $binned --scheme 'eta'
#         done
#     done
# done

# echo "pT and eta fitting completed"
# echo

# python production_asymmetry.py --year $year --size $size --path $directory"/raw_asymmetry_outcome/raw_asymmetry/pT" --input $directory"/model_fitting/" --blind 'Y' --results_path $directory"/results" --scheme 'pT'
# python production_asymmetry.py --year $year --size $size --path $directory"/raw_asymmetry_outcome/raw_asymmetry/eta" --input $directory"/model_fitting/" --blind 'Y' --results_path $directory"/results" --scheme 'eta'

# python plot_pT_eta.py --year $year --size $size --bin_path $directory"/binned_data/binning_scheme" --asymm_path $directory"/raw_asymmetry_outcome/raw_asymmetry/pT" --path $directory"/results" --scheme 'pT'
# python plot_pT_eta.py --year $year --size $size --bin_path $directory"/binned_data/binning_scheme" --asymm_path $directory"/raw_asymmetry_outcome/raw_asymmetry/eta" --path $directory"/results" --scheme 'eta'

exit