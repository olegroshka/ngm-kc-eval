import random
from datasets import load_dataset

# Load the SQuAD dataset
squad_dataset = load_dataset('squad', split='train')

# Function to categorize examples based on length and question type
def categorize_example(example):
    context_length = len(example['context'].split())  # Number of words in the context
    question_type = example['question'].split()[0]   # Type of question (what, who, where, etc.)

    # Categorize based on length
    if context_length < 150:
        context_category = 'short'
    elif context_length < 300:
        context_category = 'medium'
    else:
        context_category = 'long'

    return {'context_category': context_category, 'question_type': question_type}

# Categorize all examples
categorized_examples = [categorize_example(example) for example in squad_dataset]

# Criteria for selection
criteria = {
    'context_category': ['short', 'medium', 'long'],
    'question_type': ['What', 'Who', 'Where', 'When', 'Why', 'How']
}

# Sample 100 examples meeting the criteria
selected_examples = []
for category_type, category_values in criteria.items():
    for value in category_values:
        matching_examples = [ex for ex, cat in zip(squad_dataset, categorized_examples) if cat[category_type] == value]
        num_samples_per_category = 100 // (len(criteria) * len(category_values))
        selected_examples.extend(random.sample(matching_examples, min(num_samples_per_category, len(matching_examples))))

# Now you have a diverse sample of 100 examples
print(f"Selected {len(selected_examples)} examples.")
#print in a separate row each example
for example in selected_examples:
    print(example)
