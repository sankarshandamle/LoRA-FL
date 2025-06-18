#!/bin/bash
# params=("Tm_Adult_Baseline" "Tm_Adult_10_EO" "Tm_Adult_20_EO" "Tm_Adult_30_EO" "Tm_Adult_40_EO")
params=("Tm_Bank_Baseline" "Tm_Bank_10_EO" "Tm_Bank_20_EO" "Tm_Bank_30_EO" "Tm_Bank_40_EO")

# source activate /mnt/lia/scratch/rokvic/miniconda3/envs/server/

# Loop through each parameter and run the command
for param in "${params[@]}"
do
  echo "Running with parameter: $param"
  python3 main.py "$param" "4" 
done

echo "All runs completed." 
