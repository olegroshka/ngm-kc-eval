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
from src.experiment.t5_model import T5NGMModel
from src.experiment.ngm_model import NGMModel

MAX_NEW_TOKENS = 64

logging.basicConfig(filename='../../logs/experiment.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NGMExperiment:

    MODEL_CLASSES = {
        "t5-base": T5NGMModel,
        "t5-small": T5NGMModel,
        "t5-large": T5NGMModel,
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
        return NGMExperiment.MODEL_CLASSES.get(self.model_name, AutoNGMModel)(self.model_name, max_new_tokens=MAX_NEW_TOKENS)


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
        result_df = pd.DataFrame()
        #input  columns: id,prompt,cluster,NID
        for idx, row in self.prompts_df.iterrows():
            prompt_id = row["promptId"]
            prompt_text = row["prompt"]
            perturbation = row["perturbation"]
            cluster = row["cluster"]
            nid = row["NID"]
            refText = row["refText"]
            refCluster = row["refCluster"]
            refNid = row["refNID"]

            # Log start of processing for this prompt
            logging.info(f"Starting processing for prompt_id: {prompt_id}, NID rank: {nid}, cluster: {cluster}")
            try:
                result = self.infer(prompt_id, prompt_text, cluster, nid)

                #append the reference text and other info
                result["ref_text"] = refText
                result["ref_cluster"] = refCluster
                result["ref_nid"] = refNid
                result["perturbation"] = perturbation

                #add columns from result to result_df
                new_row = pd.Series(result)
                result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
                #result_df = result_df.append(new_row, ignore_index=True)

                # Log end of processing for this prompt
                logging.info(f"Finished processing for prompt_id: {prompt_id}")
            except Exception as e:
                logging.error(f"Error processing prompt_id: {prompt_id}. Error: {str(e)}")

        # Save results to HDF5 format
        self.save_results_csv(result_df)

    def infer(self, prompt_id, prompt_text, prompt_cluster, prompt_nid, capture_attention=False):
        # Tokenize input and collect attention scores
        start_time = time.time()
        model_result = self.ngm_model.infer(prompt_text)
        elapsed_time = time.time() - start_time

        if capture_attention:
            attentions = model_result.attentions
        else:
            attentions = None

        result = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text,
            "prompt_cluster": prompt_cluster,
            "prompt_nid": prompt_nid,
            "attention": attentions,
            "output": model_result.prediction,
            "elapsed_time": elapsed_time
        }

        return result

    def save_results_csv(self, result_df):
        """
        Save the results DataFrame to a CSV file.

        Args:
            result_df (pd.DataFrame): The DataFrame containing the experiment results.
        """
        try:
            # Save the DataFrame to CSV
            result_df.to_csv(self.output_file, index=False)

            # Logging the success message
            logging.info(f"Results successfully saved to {self.output_file}")

        except Exception as e:
            # Logging any error that might occur
            logging.error(f"Error in saving results to CSV: {str(e)}")

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
    experiment = NGMExperiment(
        "../../tmp/squad_data_100x1_prt_clustered.csv",
        "gpt2",
        "../../tmp/max_len_64/gpt2_squad_data_100x1_prt_results_128.csv")

    experiment.run_experiment()

    #main()
