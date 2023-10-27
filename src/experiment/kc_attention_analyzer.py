import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import mstats
from sklearn.preprocessing import MinMaxScaler

from src.kckit.nid.gcdd_calculator import GCDDCalculator
from src.kckit.nid.ncd_calculator import NCDCalculator
from src.kckit.nid.nid_clustering import NIDHierarchicalClusterer


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

class KCAttentionAnalyzer:
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
        # Step 1: Load all outputs from the HDF5 file
        with pd.HDFStore(self.input_file, 'r') as store:
            all_outputs = [store[f"/results/result_{idx}"]["output"].iloc[0] for idx in range(len(store.keys()) // 2)]



        # Step 2: Calculate NID and cluster values for all outputs
        output_clusterer = NIDHierarchicalClusterer(all_outputs, nid_calculator=self.nid_calculator)
        output_clusters = output_clusterer.cluster(all_outputs, num_clusters=self.num_clusters)

        results = []

        layer_idx = 0
        # Step 3: For each attention layer
        while True:  # Loop until there's no more attention data
            with pd.HDFStore(self.input_file, 'r') as store:
                try:
                    # Load all attention scores related to that layer
                    all_attentions = [store[f"/attention/attention_{idx}"].loc[layer_idx].values for idx in range(len(store.keys()) // 2)]
                except KeyError:
                    break  # Exit the loop if there's no data for this layer

            # Calculate NID and cluster values for those attention scores
            preprocessed_attention = self.preprocessor.preprocess(np.array(all_attentions))
            preprocessed_attention_bytes = self.byte_preprocessor.preprocess(preprocessed_attention)
            attention_clusterer = NIDHierarchicalClusterer(preprocessed_attention_bytes, nid_calculator=self.nid_calculator)
            attention_clusters = attention_clusterer.cluster(preprocessed_attention_bytes, num_clusters=self.num_clusters)

            results.append(attention_clusters)

            layer_idx += 1

        # Step 4: Merge the calculated NID and cluster values with the original data
        self.store_results(layer_idx, output_clusters, results)

    def store_results(self, layer_count, output_clusters, results):
        with pd.HDFStore(self.input_file, 'r') as in_store:  # Load data from the input file
            with pd.HDFStore(self.output_file, 'w') as out_store:  # 'w' mode to create/overwrite the file
                for idx in range(len(output_clusters)):
                    df = in_store[f"/results/result_{idx}"].copy()  # Load the original data

                    # Append the output cluster and NID values
                    df["output_cluster"] = output_clusters[idx, 0]
                    df["output_nid"] = output_clusters[idx, 1]

                    # Append the attention clusters and NID values for each layer
                    for i in range(layer_count):  # Loop through all layers
                        df[f"attention_cluster_{i}"] = results[i][idx, 0]
                        df[f"attention_nid_{i}"] = results[i][idx, 1]

                    # Save merged data to the new HDF5 file
                    out_store.put(f"/results/result_{idx}", df)



def main():
    parser = argparse.ArgumentParser(description="Analyze attention scores using KC.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--kc_mode", type=str, choices=['ncd', 'gcdd'], required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    analyzer = KCAttentionAnalyzer(args.input_file, args.kc_mode, args.output_file)
    analyzer.process_results()


if __name__ == "__main__":
    ##analyzer = KCAttentionAnalyzer("../../tmp/gpt2_ncd_11_lex_over_tmpl1_snt1_100.hdf", "ncd", "../../tmp/gpt2_ncd_11_lex_over_tmpl1_snt1_100_results.hdf")
    #analyzer.process_results()
    main()
