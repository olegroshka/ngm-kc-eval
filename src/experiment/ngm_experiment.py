import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import time
import argparse
import logging

class NGMExperiment:
    """
    Class to conduct experiments on Neural Generative Models (NGM) using input prompts.

    Attributes:
    - input_file (str): Path to the CSV file containing prompts.
    - model_name (str): Name of the pre-trained model to use.
    - output_file (str): Path to the output file where results will be saved.
    - prompts_df (DataFrame): DataFrame containing sorted prompts from the input file.
    - tokenizer (AutoTokenizer): Tokenizer corresponding to the given model.
    - model (AutoModelForSeq2SeqLM): Pre-trained model for the experiment.
    - device (torch.device): Device (CUDA or CPU) where the model will be loaded.
    """

    def __init__(self, input_file, model_name, output_file):
        """
        Initialize the NGMExperiment with given input file, model name, and output file.
        Load and sort the prompts and initialize the tokenizer and model.
        """
        self.input_file = input_file
        self.model_name = model_name
        self.output_file = output_file
        self.prompts_df = self._load_and_sort_prompts()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _load_and_sort_prompts(self):
        """
        Load prompts from the input CSV file and sort them based on their NCD values in ascending order.

        Returns:
        - DataFrame: Sorted DataFrame with prompts.
        """
        df = pd.read_csv(self.input_file)
        return df.sort_values(by="NID", ascending=True)

    def run_experiment(self):
        """
        Conduct the experiment by feeding each prompt to the model, collecting attention scores,
        model outputs, and elapsed time for each prompt. Save the results to the specified output file.
        """
        results = []
        #input  columns: id,prompt,cluster,NID
        for idx, row in self.prompts_df.iterrows():
            prompt_id = row["id"]
            prompt_text = row["prompt"]
            cluster = row["cluster"]
            nid = row["NID"]

            # Log start of processing for this prompt
            logging.info(f"Starting processing for prompt_id: {prompt_id}, NID rank: {nid}")
            try:
                result = self.infer(prompt_id, prompt_text)
                results.append(result)
                # Log end of processing for this prompt
                logging.info(f"Finished processing for prompt_id: {prompt_id}")
            except Exception as e:
                logging.error(f"Error processing prompt_id: {prompt_id}. Error: {str(e)}")

        # Save results to HDF5 format
        self.save_results(results)

    def infer(self, prompt_id, prompt_text):
        # Tokenize input and collect attention scores
        start_time = time.time()
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_attention_mask=True).to(self.device)
        outputs = self.model(**inputs, output_attentions=True)
        elapsed_time = time.time() - start_time

        # Extract attention scores and model output
        attention = outputs.attentions[-1][0].cpu().detach().numpy()

        if hasattr(outputs, "logits"):
            # This is for models that have logits, like T5, GPT-2, etc.
            predicted_token_ids = outputs.logits.argmax(dim=-1).cpu().detach().numpy()
            predicted_output = self.tokenizer.decode(predicted_token_ids[0])
        elif hasattr(outputs, "last_hidden_state"):
            # This is for models like BERT which give embeddings instead of logits
            # Here, you can simply take the embeddings or, if you want a simplified version, take the mean
            embeddings = outputs.last_hidden_state.cpu().detach().numpy()
            predicted_output = embeddings
            #predicted_token_ids = embeddings.mean(axis=1)  # This is just one way to represent the embeddings

        else:
            raise ValueError("The model output type is not recognized.")

        result = {
            "prompt_id": prompt_id,
            "attention": attention,
            "output": predicted_output,
            "elapsed_time": elapsed_time
        }
        return result

    def save_results(self, results):
        with pd.HDFStore(self.output_file, 'w') as store:
            for idx, result in enumerate(results):
                df = pd.DataFrame({
                    "prompt_id": [result["prompt_id"]],
                    "output": [result["output"]],
                    "elapsed_time": [result["elapsed_time"]]
                })
                # Use a prefix for naming
                store.put(f"/results/result_{idx}", df)

                # Reshape the attention tensor to be 2D
                reshaped_attention = result["attention"].reshape(-1, result["attention"].shape[-1])
                attention_df = pd.DataFrame(reshaped_attention)

                store.put(f"/attention/attention_{idx}", attention_df)

    def get_attention_scores(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
        output = self.model(**inputs)
        return output.attentions

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run experiments on Neural Generative Models using input prompts.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the CSV file containing prompts.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model to use.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file where results will be saved.")

    args = parser.parse_args()

    # Initialize and run the experiment
    experiment = NGMExperiment(args.input_file, args.model_name, args.output_file)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
