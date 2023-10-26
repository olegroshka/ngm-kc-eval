#!/bin/bash

# Ensure the script stops on first error
set -e

# Function to display usage
usage() {
    echo "Usage: $0 -i INPUT_FILE -t HEURISTIC_TYPE -n TEMPLATE_NAME -m MAX_PROMPTS -o OUTPUT_FILE"
    echo "Arguments:"
    echo "  -i  Path to the HANS jsonl file. Default is 'heuristics_evaluation_set.jsonl'."
    echo "  -t  Type of heuristic to filter by."
    echo "  -n  Template name to filter by."
    echo "  -m  Maximum number of prompts to extract."
    echo "  -o  Path to the output CSV file. Default is 'extracted_prompts.csv'."
    exit 1
}

# Check if no arguments were provided
if [ "$#" -eq 0 ]; then
    usage
fi

# Parse arguments
while getopts ":i:t:n:m:o:" opt; do
  case $opt in
    i) INPUT_FILE="$OPTARG"
    ;;
    t) HEURISTIC_TYPE="$OPTARG"
    ;;
    n) TEMPLATE_NAME="$OPTARG"
    ;;
    m) MAX_PROMPTS="$OPTARG"
    ;;
    o) OUTPUT_FILE="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
        usage
    ;;
  esac
done

# Ensure all required arguments are provided
if [ -z "$HEURISTIC_TYPE" ] || [ -z "$TEMPLATE_NAME" ] || [ -z "$MAX_PROMPTS" ]; then
    echo "Error: Missing arguments."
    usage
fi

# Default values for optional arguments
INPUT_FILE=${INPUT_FILE:-"heuristics_evaluation_set.jsonl"}
OUTPUT_FILE=${OUTPUT_FILE:-"extracted_prompts.csv"}

echo "input file: $INPUT_FILE"
echo "output file: $OUTPUT_FILE"

# Call the Python script
python ../src/input/hans_prompt_extractor.py --input_file "$INPUT_FILE" --heuristic_type "$HEURISTIC_TYPE" --template_name "$TEMPLATE_NAME" --max_prompts "$MAX_PROMPTS" --output_file "$OUTPUT_FILE"

echo "Extraction completed successfully!"
