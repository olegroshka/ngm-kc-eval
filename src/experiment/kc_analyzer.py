import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import mstats
from sklearn.preprocessing import MinMaxScaler

from src.kckit.nid.gcdd_calculator import GCDDCalculator
from src.kckit.nid.ncd_calculator import NCDCalculator
from src.kckit.nid.nid_clustering import NIDHierarchicalClusterer
from src.semantic_eval.semantic_score_evaluator import SemanticScoreEvaluator


class AttentionsPreprocessor:
    def preprocess(self, attentions):
        raise NotImplementedError

class NormalizationPreprocessor(AttentionsPreprocessor):
    def preprocess(self, attentions):
        # Normalize between 0 and 1
        min_val = np.min(attentions)
        max_val = np.max(attentions)
        return (attentions - min_val) / (max_val - min_val)


class ScalingPreprocessor(AttentionsPreprocessor):
    """
    Scale the tensor values to a given range and converts them to integers.
    By default, scales between 0 and 10,000 to represent fixed-point numbers with 4 decimal places.
    """

    def __init__(self, feature_range=(0, 100000)):
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def preprocess(self, attentions: np.ndarray) -> np.ndarray:
        shape = attentions.shape
        scaled_data = self.scaler.fit_transform(attentions.reshape(-1, 1))
        return np.round(scaled_data).astype(int).reshape(shape)

class ByteRepresentationPreprocessor(AttentionsPreprocessor):
    """
    Convert tensors into their byte representation.
    """
    def preprocess(self, attentions: np.ndarray) -> np.ndarray:
        byte_list = [tensor.tobytes() for tensor in attentions]
        return np.array(byte_list)
class ScalePreprocessor1(AttentionsPreprocessor):
    def __init__(self, scale_factor=1024):
        self.scale_factor = scale_factor

    def preprocess(self, attentions):
        return (attentions * self.scale_factor).long()

class WinzorizationPreprocessor(AttentionsPreprocessor):
    def __init__(self, lower_percentile=5, upper_percentile=95, normaliser=ScalingPreprocessor()):
        """
        Initialize the Winzorization processor with the given percentiles.

        :param lower_percentile: Lower percentile for Winzorizing.
        :param upper_percentile: Upper percentile for Winzorizing.
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.normaliser = normaliser


    def preprocess(self, attentions):
        # Apply Winzorization
        winsorized = mstats.winsorize(attentions, limits=[self.lower_percentile / 100, self.upper_percentile / 100])
        scaled = self.normaliser.preprocess(winsorized)

        return scaled


class StandardizationPreprocessor(AttentionsPreprocessor):
    def preprocess(self, attentions):
        # Standardize to zero mean and unit variance
        mean = np.mean(attentions)
        std = np.std(attentions)
        return (attentions - mean) / std

class Rounding(AttentionsPreprocessor):
    def preprocess(self, attentions):
        # Round values
        return np.round(attentions, 2)

class LogTransform(AttentionsPreprocessor):
    def preprocess(self, attentions):
        # Apply a log transformation
        return np.log1p(attentions)

class KCResultsAnalyzer:
    def __init__(self, input_file, kc_mode, output_file, nid_mod='ncd', num_clusters=11, preprocessor=ScalingPreprocessor()):
        self.input_file = input_file
        self.kc_mode = kc_mode
        self.output_file = output_file
        self.nid_mod = nid_mod
        if self.nid_mod == 'ncd':
            # Use NCD calculator here
            self.nid_calculator = NCDCalculator()
        elif self.nid_mod == 'gcdd':
            # Use GCDD calculator here
            self.nid_calculator = GCDDCalculator()
        self.num_clusters = num_clusters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = preprocessor
        self.byte_preprocessor = ByteRepresentationPreprocessor()

    def process_results(self):
        df = pd.read_csv(self.input_file)

        # Calculate NID and cluster values for all outputs
        prompts = df['prompt_text'].tolist()
        ref_texts = df['ref_text'].tolist()
        outputs = df['output'].tolist()

        output_clusterer = NIDHierarchicalClusterer(outputs, nid_calculator=self.nid_calculator)
        output_clusters = output_clusterer.cluster(outputs, num_clusters=self.num_clusters)
        df["output_cluster"] = output_clusters[:, 0]
        df["output_nid"] = output_clusters[:, 1]

        #calculate the NID between the prompt and the reference text
        df['prompt_ref_nid'] = df.apply(
            lambda row: self.nid_calculator.compute_distance(row['prompt_text'], row['ref_text']), axis=1)

        #calculate the NID between the prompt and the output text
        df['prompt_output_nid'] = df.apply(
            lambda row: self.nid_calculator.compute_distance(row['prompt_text'], row['output']), axis=1)

        #calculate the NID between the reference text and the output text
        df['ref_output_nid'] = df.apply(
            lambda row: self.nid_calculator.compute_distance(row['ref_text'], row['output']), axis=1)

        # Initialize the evaluator
        evaluator = SemanticScoreEvaluator()

        # Evaluate BERTScores
        precision, recall, f1 = evaluator.evaluate(outputs, ref_texts)

        # Append the results to the DataFrame
        df['precision'] = precision
        df['recall'] = recall
        df['f1'] = f1

        self.store_results(df)

    def store_results(self, df):
        """
               Save the results DataFrame to a CSV file.

               Args:
                   result_df (pd.DataFrame): The DataFrame containing the experiment results.
               """
        try:
            # Save the DataFrame to CSV
            df.to_csv(self.output_file, index=False)

            # Logging the success message
            print(f"Results successfully saved to {self.output_file}")

        except Exception as e:
            # Logging any error that might occur
            print(f"Error in saving results to CSV: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze attention scores using KC.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--kc_mode", type=str, choices=['ncd', 'gcdd'], required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    analyzer = KCResultsAnalyzer(args.input_file, args.kc_mode, args.output_file)
    analyzer.process_results()


if __name__ == "__main__":
    #analyzer = KCResultsAnalyzer("../../tmp/gpt2_squad_data_100x1_prt_results.csv", "ncd", "../../tmp/gpt2_squad_data_100x1_prt_out.csv")
    #analyzer = KCResultsAnalyzer("../../tmp/t5-base_squad_data_100x1_prt_results.csv", "ncd", "../../tmp/t5-base_squad_data_100x1_prt_out.csv")
    analyzer = KCResultsAnalyzer("../../tmp/t5-small_squad_data_100x1_prt_results.csv", "ncd", "../../tmp/t5-small_squad_data_100x1_prt_out.csv")
    analyzer.process_results()
    #main()
