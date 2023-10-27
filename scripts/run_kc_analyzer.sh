#!/bin/bash

# Function to display usage
function usage() {
    echo "Usage: $0 -i INPUT_FILE -k KC_MODE -o OUTPUT_FILE [OPTIONS]"
    echo
    echo "Arguments:"
    echo "  -i    Path to the input HDF5 file."
    echo "  -k    KC mode (ncp or gcp)."
    echo "  -o    Path to the output HDF5 file."
    echo
    echo "Optional arguments:"
    echo "  -p    Preprocessing strategy (default: Normalization)."
    echo "  -h    Display this help message."
    exit 1
}

# Parse arguments
while getopts ":i:k:o:p:h" opt; do
    case $opt in
        i) INPUT_FILE="$OPTARG";;
        k) KC_MODE="$OPTARG";;
        o) OUTPUT_FILE="$OPTARG";;
        p) PREPROCESS_STRATEGY="$OPTARG";;
        h) usage;;
        \?) echo "Invalid option -$OPTARG" >&2; usage;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage;;
    esac
done

export PYTHONPATH=$PYTHONPATH:../

# Validate mandatory arguments
if [ -z "$INPUT_FILE" ] || [ -z "$KC_MODE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Error: Missing mandatory arguments."
    usage
fi

# Optionally, activate a virtual environment if you're using one
# source /path/to/your/virtualenv/bin/activate

# Call the KCAttentionAnalyzer with parsed arguments
python ../src/experiment/kc_attention_analyzer.py --input_file "$INPUT_FILE" --kc_mode "$KC_MODE" --output_file "$OUTPUT_FILE"

echo "KCAttentionAnalyzer completed successfully!"