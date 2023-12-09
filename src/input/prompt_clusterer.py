import argparse

import pandas as pd

from src.kckit.nid.ncd_calculator import NCDCalculator
from src.kckit.nid.gcdd_calculator import GCDDCalculator
from src.kckit.nid.nid_clustering import NIDHierarchicalClusterer
from src.kckit.nid.nid_interface import InformationDistanceCalculator
class PromptClusterer:
    def __init__(self, nid_mod='ncd'):
        """
        Initialize the PromptClusterer with the desired NID mode.

        :param nid_mod: The mode of Normalized Information Distance (NID) calculator. Options are 'ncd' and 'gcdd'.
        """
        if nid_mod == 'ncd':
            # Use your NCD calculator here
            self.nid_calculator = NCDCalculator()
        elif nid_mod == 'gcdd':
            # Use your GCDD calculator here
            self.nid_calculator = GCDDCalculator()
        else:
            raise ValueError(f"Unsupported NID mode: {nid_mod}")


    def process_prompts(self, input_csv: str, output_csv: str, num_clusters: int):
        """
        Process the prompts: cluster them and calculate their NID values.

        :param input_csv: Path to the input CSV file.
        :param output_csv: Path to the output CSV file.
        :param num_clusters: Desired number of clusters.
        """
        # Load the CSV
        df = pd.read_csv(input_csv)

        # Ensure the CSV has the expected columns
        assert 'promptId' in df.columns and 'prompt' in df.columns, "Input CSV should have 'id' and 'prompt' columns."

        # Extract prompts and process them
        prompts = df['prompt'].tolist()
        prompt_nid_clusterer = NIDHierarchicalClusterer(prompts, self.nid_calculator)
        prompt_nid_results = prompt_nid_clusterer.cluster(prompts, num_clusters)

        # Add prompt cluster and NID columns to the DataFrame
        df['cluster'] = prompt_nid_results[:, 0].astype(int)
        df['NID'] = prompt_nid_results[:, 1]

        # Extract ref text and process them
        refs = df['refText'].tolist()
        ref_nid_clusterer = NIDHierarchicalClusterer(refs, self.nid_calculator)
        ref_nid_results = ref_nid_clusterer.cluster(refs, num_clusters)

        # Add prompt cluster and NID columns to the DataFrame
        df['refCluster'] = ref_nid_results[:, 0].astype(int)
        df['refNID'] = ref_nid_results[:, 1]

        # Save the processed DataFrame to the output CSV
        df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Cluster prompts based on NID and save results to a CSV.")
    #parser.add_argument("input_csv", type=str, help="Path to the input CSV file containing prompts.")
    #parser.add_argument("output_csv", type=str, help="Path to save the output CSV with cluster and NID values.")
    #parser.add_argument("--nid_mod", type=str, default="ncd", choices=["ncd", "gcdd"],
    #                    help="The mode of Normalized Information Distance (NID) calculator. Options are 'ncd' and 'gcdd'. Default is 'ncd'.")
    #parser.add_argument("--num_clusters", type=int, required=True, help="Desired number of clusters.")

    #args = parser.parse_args()

    #clusterer = PromptClusterer(nid_mod=args.nid_mod)
    #clusterer.process_prompts(args.input_csv, args.output_csv, args.num_clusters)

    # Demo
    clusterer = PromptClusterer(nid_mod='ncd')
    clusterer.process_prompts('../../data/SQuAD/squad_data_100x1_prt.csv', "../../tmp/squad_data_100x1_prt_clustered.csv", num_clusters=11)
    #clusterer.process_prompts('../../data/SQuAD/squad_data_10x1_prt.csv', "../../tmp/squad_data_10x1_prt_clustered.csv", num_clusters=5)
