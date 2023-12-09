from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from bert_score import score

# Load the model and tokenizer
model_name = 't5-small'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load examples from the SQuAD dataset
dataset = load_dataset('squad', split='validation[:10]')

# Function to generate answer
def generate_answer(context, question):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs['input_ids'], max_length=50, min_length=5, length_penalty=1.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate answers and calculate BERTScore
for i in range(10):
    context = dataset[i]['context']
    question = dataset[i]['question']
    reference_answer = dataset[i]['answers']['text'][0]  # Taking the first answer as reference
    generated_answer = generate_answer(context, question)

    # Calculate BERTScore
    P, R, F1 = score([generated_answer], [reference_answer], lang='en', rescale_with_baseline=True)
    print(f"Question {i+1}: {question}")
    print(f"Context: {context}")
    print(f"Generated Answer: {generated_answer}")
    print(f"Reference Answer: {reference_answer}")
    print(f"BERTScore: Precision - {P[0]}, Recall - {R[0]}, F1 - {F1[0]}\n")
