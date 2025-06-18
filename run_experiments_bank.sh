#!/bin/bash
params=("FedAvg_Bank_Baseline" "FedAvg_Bank_10_EO" "FedAvg_Bank_20_EO" "FedAvg_Bank_30_EO" "FedAvg_Bank_40_EO" "Krum_Bank_Baseline" "Krum_Bank_10_EO" "Krum_Bank_20_EO" "Krum_Bank_30_EO" "Krum_Bank_40_EO" "Tm_Bank_Baseline" "Tm_Bank_10_EO" "Tm_Bank_20_EO" "Tm_Bank_30_EO" "Tm_Bank_40_EO")

# source activate /mnt/lia/scratch/rokvic/miniconda3/envs/server/

# Loop through each parameter and run the command
for param in "${params[@]}"
do
  echo "Running with parameter: $param"
  python3 main.py "$param" "4" 
done

echo "All runs completed." 
