from bert_score import score
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load T5 model and tokenizer
#model_name = "t5-base"#"t5-large"#"t5-small"#"t5-large"#"t5-3b"
model_name = "t5-small"#"t5-large"#"t5-small"#"t5-large"#"t5-3b"
#model_name = "t5-large"#"t5-small"#"t5-large"#"t5-3b"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate model predictions (simplified for demonstration)
def generate_text(input_text, max_length=256):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=max_length)[0]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

# Example input and reference texts
input_texts = [
    "The government has announced a new healthcare plan, aiming to provide affordable medical services to all citizens. This plan includes increased funding for hospitals and reduced prescription costs.",
    "This latest smartphone model features a high-resolution camera, a long-lasting battery, and a user-friendly interface. It’s perfect for photography enthusiasts and casual users alike.",
    "The film presents a gripping story with stunning visuals and exceptional performances by the cast. However, its pacing feels slow at times.",
    "Paris is a city of romance, known for its iconic Eiffel Tower, exquisite cuisine, and rich history. Visitors should not miss the Louvre Museum and the charming Montmartre neighborhood.",
    "Thank you for your inquiry. We apologize for the inconvenience caused. Your request has been forwarded to our support team, and they will contact you within 24 hours."
]

reference_texts = [
    "A new healthcare policy has been introduced by the government to ensure accessible and affordable healthcare for everyone, with more funds for hospitals and lower prices for medications.",
    "The new smartphone offers an advanced camera, extended battery life, and an intuitive interface, making it ideal for both photography lovers and everyday users.",
    "The movie offers an engaging narrative and impressive visual effects, along with outstanding acting, though it occasionally suffers from slow pacing.",
    "Famous for its romantic ambiance, Paris boasts landmarks like the Eiffel Tower and historical sites. The Louvre and the picturesque district of Montmartre are must-visit spots.",
    "We appreciate your reaching out and regret any inconvenience. Your issue has been escalated to our support team, and you will receive a response within the next day."
]

generated_texts = [generate_text(text) for text in input_texts]

# Calculate BERTScore
P, R, F1 = score(generated_texts, reference_texts, lang='en', rescale_with_baseline=True)

# Output the results
for i in range(len(input_texts)):
    print(f"Input: {input_texts[i]}")
    print(f"Generated: {generated_texts[i]}")
    print(f"Reference: {reference_texts[i]}")
    print(f"BERTScore: Precision - {P[i]}, Recall - {R[i]}, F1 - {F1[i]}\n")
