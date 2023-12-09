from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from bert_score import score

# Load the model and tokenizer
model_name = 't5-small'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load 10 examples from XSum dataset
dataset = load_dataset('xsum', split='train[:10]')

# Function to generate summaries
def generate_summary(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate summaries and calculate BERTScore
for i in range(10):
    input_text = dataset[i]['document']
    reference_text = dataset[i]['summary']
    generated_text = generate_summary(input_text)

    # Calculate BERTScore
    P, R, F1 = score([generated_text], [reference_text], lang='en', rescale_with_baseline=True)
    print(f"    Input {i+1}: {input_text}")
    print(f"    Generated: {generated_text}")
    print(f"    Reference: {reference_text}")
    print(f"    BERTScore: Precision - {P[0]}, Recall - {R[0]}, F1 - {F1[0]}\n")
