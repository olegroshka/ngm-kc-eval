import pandas as pd
from bert_score import score
from transformers import T5Tokenizer, T5ForConditionalGeneration

class SemanticScoreEvaluator:
    """
    A class to evaluate the semantic similarity between generated texts and reference texts
    using BERTScore for calculating precision, recall, and F1 scores.
    """

    def evaluate(self, generated_texts, reference_texts):
        """
        Computes BERTScores (precision, recall, F1) for pairs of generated and reference texts.

        Args:
            generated_texts (list of str): Generated text strings.
            reference_texts (list of str): Reference text strings.

        Returns:
            tuple: Three lists containing precision, recall, and F1 scores for each text pair.
        """
        P, R, F1 = score(generated_texts, reference_texts, lang='en', rescale_with_baseline=True)
        return P.tolist(), R.tolist(), F1.tolist()

if __name__ == "__main__":
    # Example usage of SemanticScoreEvaluator

    # Load T5 model and tokenizer
    # model_name = "t5-base"#"t5-large"#"t5-small"#"t5-large"#"t5-3b"
    model_name = "t5-small"  # "t5-large"#"t5-small"#"t5-large"#"t5-3b"
    # model_name = "t5-large"#"t5-small"#"t5-large"#"t5-3b"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)


    # Function to generate model predictions (simplified for demonstration)
    def generate_text(input_text, max_length=64):
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
        outputs = model.generate(inputs.input_ids, max_new_tokens=max_length, attention_mask=inputs.attention_mask, pad_token_id=tokenizer.eos_token_id)
        predicted_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return predicted_output
        # input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        # output_ids = model.generate(input_ids, max_length=max_length)[0]
        # return tokenizer.decode(output_ids, skip_special_tokens=True)


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

    df = pd.read_csv("../../tmp/t5-small_squad_data_10x1_prt_out.csv")#'../../data/SQuAD/squad_data_10x1_prt.csv')

    # Extract prompts and process them
    input_texts = df['prompt_text'].tolist()
    reference_texts = df['ref_text'].tolist()

    #generated_texts = [generate_text(text) for text in input_texts]
    generated_texts = df['output'].tolist()
    p = df['precision'].tolist()

    evaluator = SemanticScoreEvaluator()
    precision, recall, f1 = evaluator.evaluate(generated_texts, reference_texts)

    # Print the results
    for i in range(len(generated_texts)):
        print(f"Input: {generated_texts[i]}")
        print(f"Reference: {reference_texts[i]}")
        print(f"BERTScore: Precision - {precision[i]}, Recall - {recall[i]}, F1 - {f1[i]}\n")

