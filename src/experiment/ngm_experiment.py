import pandas as pd
import torch
import time
import argparse
import logging
import warnings
from pandas.errors import PerformanceWarning

from src.experiment.auto_model import AutoNGMModel
from src.experiment.bert_model import BERTNGMModel
from src.experiment.gpt2_model import GPT2NGMModel
from src.experiment.mistral_model import MistralNGMModel
from src.experiment.ngm_model import NGMModel

logging.basicConfig(filename='../logs/experiment.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class NGMExperiment:

    MODEL_CLASSES = {
        "gpt2": GPT2NGMModel,
        "gpt2-medium": GPT2NGMModel,
        "gpt2-large": GPT2NGMModel,
        "gpt2-xl": GPT2NGMModel,
        "openai-gpt": GPT2NGMModel,
        "microsoft/phi-1_5": GPT2NGMModel,
        "filipealmeida/Mistral-7B-v0.1-sharded": MistralNGMModel,
        "bert": BERTNGMModel,
        "bert-base-uncased": BERTNGMModel,
        "bert-large-uncased": BERTNGMModel,
        "bert-base-cased": BERTNGMModel,
        "bert-large-cased": BERTNGMModel,
        "bert-base-multilingual-uncased": BERTNGMModel,
        "bert-base-multilingual-cased": BERTNGMModel,

        # ... (add other specific models here as needed)
    }

    """
    Class to conduct experiments on Neural Generative Models (NGM) using input prompts.

    Attributes:
    - input_file (str): Path to the CSV file containing prompts.
    - model_name (str): Name of the pre-trained model to use.
    - output_file (str): Path to the output file where results will be saved.
    - device (torch.device): Device (CUDA or CPU) where the model will be loaded.
    """

    def __init__(self, input_file, model_name, output_file):
        self.input_file = input_file
        self.model_name = model_name
        self.output_file = output_file
        self.prompts_df = self._load_and_sort_prompts()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ngm_model: NGMModel = self._initialize_model()

    def _initialize_model(self):
        return NGMExperiment.MODEL_CLASSES.get(self.model_name, AutoNGMModel)(self.model_name)


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
            logging.info(f"Starting processing for prompt_id: {prompt_id}, NID rank: {nid}, cluster: {cluster}")
            try:
                result = self.infer(prompt_id, prompt_text, cluster, nid)
                print("Processed prompt: ", prompt_text)
                results.append(result)
                # Log end of processing for this prompt
                logging.info(f"Finished processing for prompt_id: {prompt_id}")
            except Exception as e:
                logging.error(f"Error processing prompt_id: {prompt_id}. Error: {str(e)}")

        # Save results to HDF5 format
        self.save_results(results)

    def infer(self, prompt_id, prompt_text, prompt_cluster, prompt_nid):
        # Tokenize input and collect attention scores
        start_time = time.time()
        model_result = self.ngm_model.infer(prompt_text)
        elapsed_time = time.time() - start_time

        result = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "prompt_cluster": prompt_cluster,
            "prompt_nid": prompt_nid,
            "attention": model_result.attentions,
            "output": model_result.prediction,
            "elapsed_time": elapsed_time
        }

        return result

    def save_results(self, results):
        with pd.HDFStore(self.output_file, 'w') as store:
            for idx, result in enumerate(results):
                df = pd.DataFrame({
                    "prompt_id": [result["prompt_id"]],
                    "prompt_text": [result["prompt_text"]],
                    "prompt_cluster": [result["prompt_cluster"]],
                    "prompt_nid": [result["prompt_nid"]],
                    "output": [result["output"]],
                    "elapsed_time": [result["elapsed_time"]]
                })

                # Suppress the warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=PerformanceWarning)
                    store.put(f"/results/result_{idx}", df)

                # Check if attention scores are in tuple format
                if isinstance(result["attention"], tuple):
                    attention_tensor = result["attention"][-1].cpu().detach().numpy()
                else:
                    attention_tensor = result["attention"].cpu().detach().numpy()

                # Convert the 4D tensor attention scores to MultiIndex DataFrame
                layers, heads, tokens_from, tokens_to = attention_tensor.shape
                multi_index = pd.MultiIndex.from_product(
                    [range(layers), range(heads), range(tokens_from), range(tokens_to)],
                    names=['layer', 'head', 'token_from', 'token_to']
                )
                attention_df = pd.DataFrame(attention_tensor.reshape(-1), index=multi_index, columns=['attention_score'])

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
