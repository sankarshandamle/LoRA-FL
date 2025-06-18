#!/bin/bash
params=("FedAvg_Adult_Baseline" "FedAvg_Adult_10_EO" "FedAvg_Adult_20_EO" "FedAvg_Adult_30_EO" "FedAvg_Adult_40_EO" "Krum_Adult_Baseline" "Krum_Adult_10_EO" "Krum_Adult_20_EO" "Krum_Adult_30_EO" "Krum_Adult_40_EO" "Tm_Adult_Baseline" "Tm_Adult_10_EO" "Tm_Adult_20_EO" "Tm_Adult_30_EO" "Tm_Adult_40_EO")

# source activate /mnt/lia/scratch/rokvic/miniconda3/envs/server/

# Loop through each parameter and run the command
for param in "${params[@]}"
do
  echo "Running with parameter: $param"
  python3 main.py "$param" "4" 
done

echo "All runs completed." 
