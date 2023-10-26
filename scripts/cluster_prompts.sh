#!/bin/bash

# Script to cluster prompts using PromptClusterer

# Check for the right number of arguments
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_csv> <output_csv> <nid_mod> <num_clusters>"
    exit 1
fi

input_csv="$1"
output_csv="$2"
nid_mod="$3"
num_clusters="$4"

export PYTHONPATH=$PYTHONPATH:../

# Call the Python script with the provided arguments
python3 ../src/input/prompt_clusterer.py "$input_csv" "$output_csv" --nid_mod "$nid_mod" --num_clusters "$num_clusters"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Prompts successfully clustered. Output saved to $output_csv."
else
    echo "An error occurred while clustering prompts."
    exit 1
fi
