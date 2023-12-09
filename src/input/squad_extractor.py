import csv
import random
from datasets import load_dataset
from src.kckit.nid.gcdd_calculator import GCDDCalculator

class SQUADExtractor:
    """
    A class to extract a subset of data from the SQuAD dataset and write it to a CSV file.

    This class is designed to perform stratified sampling of the SQuAD dataset based on
    certain criteria like context length and question type, in order to ensure a diverse
    selection of examples for analysis or model evaluation.

    Attributes:
        max_size (int): The maximum number of examples to extract from the dataset.
        output_file (str): The name of the output CSV file where the data will be saved.

    Methods:
        categorize_example(example): Categorizes an example from SQuAD based on predefined criteria.
        extract_data(): Extracts the data based on the sampling and writes it to a CSV file.
    """

    def __init__(self, max_size, output_file):
        """
        Initializes the SQUADExtractor with the given maximum dataset size and output file name.

        Args:
            max_size (int): The maximum number of examples to extract.
            output_file (str): The file name for the output CSV file.
        """
        self.max_size = max_size
        self.output_file = output_file

    def categorize_example(self, example):
        """
        Categorizes a given SQuAD example based on the length of the context and the question type.

        Args:
            example (dict): A dictionary representing a SQuAD example.

        Returns:
            dict: A dictionary containing the categorization of the example.
        """
        context_length = len(example['context'].split())  # Number of words in the context
        question_type = example['question'].split()[0]   # Type of question (what, who, where, etc.)

        # Categorize based on length and question type
        if context_length < 150:
            context_category = 'short'
        elif context_length < 300:
            context_category = 'medium'
        else:
            context_category = 'long'

        return {'context_category': context_category, 'question_type': question_type}

    def extract_data(self):
        """
        Extracts a subset of examples from the SQuAD dataset and writes them to a CSV file.

        The method applies stratified sampling based on the categorization criteria to 
        select a diverse set of examples. The extracted data includes the `promptId`, 
        `prompt`, and `refText` for each example.
        """        # Load the SQuAD dataset
        squad_dataset = load_dataset('squad', split='train')

        # Categorize all examples
        categorized_examples = [self.categorize_example(example) for example in squad_dataset]

        # Criteria for selection
        criteria = {
            'context_category': ['short', 'medium', 'long'],
            'question_type': ['What', 'Who', 'Where', 'When', 'Why', 'How']
        }

        # Sample examples meeting the criteria
        selected_examples = []
        for category_type, category_values in criteria.items():
            for value in category_values:
                matching_examples = [ex for ex, cat in zip(squad_dataset, categorized_examples) if cat[category_type] == value]
                num_samples_per_category = self.max_size // (len(criteria) * len(category_values))
                selected_examples.extend(random.sample(matching_examples, min(num_samples_per_category, len(matching_examples))))

        # Write to CSV
        with open(self.output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['promptId', 'prompt', 'refText'])

            for item in selected_examples[:self.max_size]:
                promptId = item['id']
                prompt = item['context']
                refText = item['answers']['text'][0]  # Taking the first answer as reference

                writer.writerow([promptId, prompt, refText])

# Example usage
extractor = SQUADExtractor(max_size=20, output_file='../../data/SQuAD/squad_data_10.csv')
extractor.extract_data()
