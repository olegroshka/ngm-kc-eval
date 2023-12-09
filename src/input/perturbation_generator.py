import csv
import logging
import warnings

from src.input.perturbation import TextPerturbation

logging.basicConfig(filename='../logs/perturbation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class PerturbationGenerator:
    """
    A class to generate perturbations for text data from a CSV file and write the results to another CSV file.

    Attributes:
        input_csv (str): Input CSV file name.
        num_perturbations (int): Number of perturbations per input.
        output_csv (str): Output CSV file name.
    """

    def __init__(self, input_csv, num_perturbations, output_csv, perturbation: TextPerturbation):
        self.input_csv = input_csv
        self.num_perturbations = num_perturbations
        self.output_csv = output_csv
        self.perturbation = perturbation

    def generate_perturbations(self):
        with open(self.input_csv, mode='r', encoding='utf-8') as infile, \
                open(self.output_csv, mode='w', newline='', encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=['promptId', 'prompt', 'perturbation', 'refText'])
            writer.writeheader()

            for row in reader:
                promptId = row['promptId']
                prompt = row['prompt']
                refText = row['refText']

                # Write original data
                writer.writerow({'promptId': promptId, 'prompt': prompt, 'perturbation': False, 'refText': refText})

                # Generate and write perturbed data
                perturbations = self.perturbation.generate_perturbations(prompt, self.num_perturbations)
                logging.info("Created perturbations for promptId: " + promptId)
                for i in range(self.num_perturbations):
                    perturbed_text = perturbations[i]
                    writer.writerow({'promptId': f"{promptId}_{i + 1}", 'prompt': perturbed_text,  'perturbation': True, 'refText': refText})


# Example usage
generator = PerturbationGenerator(
    '../../data/XSum/xsum_data_100.csv',
    1, '../../data/XSum/xsum_data_33x1_prt.csv',
    TextPerturbation())

generator.generate_perturbations()
