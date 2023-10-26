import json
import hashlib
import argparse
import csv


class HANSPromptExtractor:
    def __init__(self, input_file, sentence='sentence1'):
        self.input_file = input_file
        self.prompts = self._load_data()
        self.sentence = sentence # Extracting sentence1; adjust if you need sentence2

    def _load_data(self):
        with open(self.input_file, 'r') as f:
            return [json.loads(line.strip()) for line in f]

    def extract_prompts(self, heuristic_type, template_name, max_prompts):
        extracted_prompts = []

        for data in self.prompts:
            if data['heuristic'] == heuristic_type and data['template'] == template_name:
                sentence = data[self.sentence]
                sentence_id = hashlib.md5(sentence.encode()).hexdigest()  # Creating a unique ID using MD5 checksum
                extracted_prompts.append((sentence_id, sentence))
                if len(extracted_prompts) >= max_prompts:
                    break

        return extracted_prompts

    def save_to_csv(self, extracted_prompts, output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prompt"])  # CSV header
            writer.writerows(extracted_prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract prompts from HANS based on heuristic and template.")
    parser.add_argument("--input_file", type=str, default="heuristics_evaluation_set.jsonl",
                        help="Path to the HANS jsonl file.")
    parser.add_argument("--heuristic_type", type=str, required=True, help="Type of heuristic to filter by.")
    parser.add_argument("--template_name", type=str, required=True, help="Template name to filter by.")
    parser.add_argument("--max_prompts", type=int, required=True, help="Maximum number of prompts to extract.")
    parser.add_argument("--output_file", type=str, default="extracted_prompts.csv", help="Path to the output CSV file.")

    args = parser.parse_args()

    extractor = HANSPromptExtractor(args.input_file)
    extracted_prompts = extractor.extract_prompts(args.heuristic_type, args.template_name, args.max_prompts)
    extractor.save_to_csv(extracted_prompts, args.output_file)
