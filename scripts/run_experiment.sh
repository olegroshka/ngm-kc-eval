#!/bin/bash

# Ensure the script stops on first error
set -e

# Function to display usage
usage() {
    echo "Usage: $0 -i INPUT_FILE -m MODEL_NAME -o OUTPUT_FILE"
    echo "Arguments:"
    echo "  -i  Path to the CSV file containing prompts."
    echo "  -m  Name of the pre-trained model to use."
    echo "  -o  Path to the output file where results will be saved."
    exit 1
}

# Check if no arguments were provided
if [ "$#" -eq 0 ]; then
    usage
fi

# Parse arguments
while getopts ":i:m:o:" opt; do
  case $opt in
    i) INPUT_FILE="$OPTARG"
    ;;
    m) MODEL_NAME="$OPTARG"
    ;;
    o) OUTPUT_FILE="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        usage
    ;;
  esac
done

# Ensure all arguments are provided
if [ -z "$INPUT_FILE" ] || [ -z "$MODEL_NAME" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Error: Missing arguments."
    usage
fi

# Capture start time
START_TIME=$(date +%s)

# Call the Python script
python ../src/experiment/ngm_experiment.py --input_file "$INPUT_FILE" --model_name "$MODEL_NAME" --output_file "$OUTPUT_FILE"

# Compute elapsed time
END_TIME=$(date +%s)ls
ELAPSED_TIME=$((END_TIME - START_TIME))

echo "Experiment completed successfully!"
echo "Time elapsed: $ELAPSED_TIME seconds"
