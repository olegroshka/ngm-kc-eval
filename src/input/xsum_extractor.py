import csv
import random
from datasets import load_dataset


class XSumExtractor:
    """
       A class to extract a subset of data from the XSum dataset and write it to a CSV file.

       This class facilitates the sampling of diverse examples from the XSum dataset, focusing on
       the variability in article and summary lengths. It's useful for summarization tasks,
       data analysis, or model training.

       Attributes:
           max_size (int): The maximum number of examples to extract from the dataset.
           output_file (str): The name of the output CSV file where the data will be saved.

       Methods:
           categorize_example(example): Categorizes an example from XSum based on predefined criteria.
           extract_data(): Extracts the data based on the sampling and writes it to a CSV file.
       """

    def __init__(self, max_size, output_file):
        """
        Initializes the XSumExtractor with the given maximum dataset size and output file name.
        Args:
            max_size (int): The maximum number of examples to extract.
            output_file (str): The file name for the output CSV file.
        """
        self.max_size = max_size
        self.output_file = output_file

    def categorize_example(self, example):
        """
        Categorizes a given XSum example based on the length of the article and the summary.
        Args:
            example (dict): A dictionary representing an XSum example.
            Returns:
                dict: A dictionary containing the categorization of the example.
        """
        article_length = len(example['document'].split())  # Number of words in the article
        summary_length = len(example['summary'].split())  # Number of words in the summary

        # Categorize based on length
        if article_length < 300:
            article_category = 'short'
        elif article_length < 600:
            article_category = 'medium'
        else:
            article_category = 'long'

        if summary_length < 25:
            summary_category = 'short'
        elif summary_length < 50:
            summary_category = 'medium'
        else:
            summary_category = 'long'

        return {'article_category': article_category, 'summary_category': summary_category}

    def extract_data(self):
        """
        Extracts a subset of examples from the XSum dataset and writes them to a CSV file.

        The method applies stratified sampling based on the categorization criteria to
        select a diverse set of examples. The extracted data includes the `promptId`,
        `prompt`, and `refText` for each example.
        """
        # Load the XSum dataset
        xsum_dataset = load_dataset('xsum', split='train')

        # Categorize all examples
        categorized_examples = [self.categorize_example(example) for example in xsum_dataset]

        # Criteria for selection
        criteria = {
            'article_category': ['short', 'medium', 'long'],
            'summary_category': ['short', 'medium', 'long']
        }

        # Sample examples meeting the criteria
        selected_examples = []
        for category_type, category_values in criteria.items():
            for value in category_values:
                matching_examples = [ex for ex, cat in zip(xsum_dataset, categorized_examples) if
                                     cat[category_type] == value]
                num_samples_per_category = self.max_size // (len(criteria) * len(category_values))
                selected_examples.extend(
                    random.sample(matching_examples, min(num_samples_per_category, len(matching_examples))))

        # Write to CSV
        with open(self.output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['promptId', 'prompt', 'refText'])

            for item in selected_examples[:self.max_size]:
                promptId = item['id']
                prompt = item['document']
                refText = item['summary']

                writer.writerow([promptId, prompt, refText])


# Example usage
extractor = XSumExtractor(max_size=10, output_file='../../data/XSum/xsum_data_10.csv')
extractor.extract_data()
