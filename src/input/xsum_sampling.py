import random
from datasets import load_dataset

# Load the XSum dataset
xsum_dataset = load_dataset('xsum', split='train')

# Function to categorize examples based on length
def categorize_example(example):
    article_length = len(example['document'].split())  # Number of words in the article
    summary_length = len(example['summary'].split())   # Number of words in the summary

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

# Categorize all examples
categorized_examples = [categorize_example(example) for example in xsum_dataset]

# Criteria for selection
criteria = {
    'article_category': ['short', 'medium', 'long'],
    'summary_category': ['short', 'medium', 'long']
}

# Sample 100 examples meeting the criteria
selected_examples = []
for category_type, category_values in criteria.items():
    for value in category_values:
        matching_examples = [ex for ex, cat in zip(xsum_dataset, categorized_examples) if cat[category_type] == value]
        selected_examples.extend(random.sample(matching_examples, 100 // (len(criteria) * len(category_values))))

# Now you have a diverse sample of 100 examples
print(f"Selected {len(selected_examples)} examples.")
