# # Runs the complete analysis on a set of raw data of D0 meson decays to obtain the asymmetry in local regions of the phase space. The output is stored in the specified directory, and is organized in several directories generated by this same script. Take into account that if a directory with the same name already exists this code might not work as intended. Note that making changes to any of the individual scripts while this code is running can lead to a malfunction.
# # When running the code the output directory, the year the data to be analysed was taken, the size of the data to be analysed and whether or not the data should be binned when fitting must be given as arguments, in that order. The year must be one of: 16, 17 or 18. The size must be one of: small, medium, large, 1, 2, 3, 4, 5, 6, 7 or 8. The binned fitting argument must either be y/Y or n/N.
# # Author: Marc Oriol Pérez (marc.oriolperez@student.manchester.ac.uk)
# # Last modified: 16th September 2023

directory=$1
year=$2
size=$3
binned=$4
selected=$5

# if [[ "$binned" != "y" ]]; then
#     if [[ "$binned" != "Y" ]]; then
#         if [[ "$binned" != "n" ]]; then
#             if [[ "$binned" != "N" ]]; then
#                 echo "WARNING: You did not select a valid option for the binned fit"
#                 echo
#                 echo "An unbinned fit will be performed"
#                 binned="n"
#             fi   
#         fi
#     fi
# fi

# # Create necessary directories to store output


# if [[ "$selected" = "n" ]]; then
#     mkdir $directory
#     mkdir $directory"/selected_data"
# fi
# mkdir $directory"/binned_data"
# mkdir $directory"/binned_data/binning_scheme"
# mkdir $directory"/model_fitting"
# mkdir $directory"/model_fitting/global"
# mkdir $directory"/model_fitting/local"
# for ind in {0..99}
# do
#     index=$( printf '%02d' $ind)
#     mkdir $directory"/model_fitting/local/"$index
# done
# mkdir $directory"/raw_asymmetry_outcome"
# mkdir $directory"/raw_asymmetry_outcome/chi_squared"
# mkdir $directory"/raw_asymmetry_outcome/raw_asymmetry"
# mkdir $directory"/results"

# echo "The necessary directories have been created"
# echo


# # Run the code
# if [[ "$selected" = "n" ]]; then
#     python selection_of_events.py --year $year --size $size --path $directory"/selected_data"

#     echo
#     for polar in up down
#     do

#         python multiple_candidates.py --year $year --size $size --polarity $polar --path $directory"/selected_data"
#     done
#     echo "Multiple candidates have been removed"
# fi

# if [[ "$selected" = "y" ]]; then
#     echo "Skipping selection of events and multiple candidates"
# fi
# python fit_global.py --year $year --size $size --path $directory"/model_fitting/global" --binned_fit $binned --input $directory"/selected_data"
# for meson in D0 D0bar
# do 
#     for polar in up down 
#     do    
#         python model_fitting.py --year $year --size $size --meson $meson --polarity $polar  --path $directory"/model_fitting/global" --input $directory"/selected_data" --parameters_path $directory"/model_fitting/global" --global_local 'n' --binned_fit $binned
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

# for meson in D0 D0bar
# do
#    for polar in up down
#    do 
#         for ind in {0..99}
#         do
#             index=$( printf '%02d' $ind)
#             python model_fitting.py --year $year --size $size --meson $meson --polarity $polar  --path $directory"/model_fitting/local/"$index --input $directory"/binned_data" --parameters_path $directory"/model_fitting/global" --bin $index --binned_fit $binned --global_local 'y'
#         done
#     done
# done

# echo "Local fitting completed"
# echo

# #python analyse_chisquared.py --year $year --size $size --path $directory"/raw_asymmetry_outcome/chi_squared" --input $directory"/model_fitting/local"

#python production_asymmetry.py --year $year --size $size --path $directory"/raw_asymmetry_outcome/raw_asymmetry" --input $directory"/model_fitting/local" --blind 'Y'
for meson in D0 
do 
    for polar in up 
    do    
        python test.py --year $year --size $size --meson $meson --polarity $polar --input $directory"/selected_data" --bin_path $directory"/binned_data/binning_scheme"
    done
done
exit