from typing import List

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import scipy


from src.kckit.nid.ncd_calculator import NCDCalculator
from src.kckit.nid.gcdd_calculator import GCDDCalculator
from src.kckit.nid.nid_interface import InformationDistanceCalculator
from src.kckit.nid.centroid_sentence_builder import ReferenceSentenceBuilder

class NIDHierarchicalClusterer:
    """
        Hierarchical clustering of prompts based on normalized information distance (NID).

        This class accepts a list of prompts and an instance of InformationDistanceCalculator (which could be
        an NCD or GCDD calculator, for example). It then computes the pairwise distances between the prompts
        and performs hierarchical clustering based on these distances. The results can be visualized using a dendrogram.

        Attributes:
            prompts (list of str): The list of prompts to be clustered.
            nid_calculator (InformationDistanceCalculator): An instance of a class implementing the
                InformationDistanceCalculator interface to compute the distances between prompts.
            nid_matrix (numpy.ndarray): A matrix storing the pairwise distances between prompts.
            reference_builder_mode: The mode for ReferenceSentenceBuilder ('centroid', 'tfidf', 'hybrid').

        Methods:
            cluster_prompts(): Perform hierarchical clustering on the prompts.
            plot_dendrogram(): Visualize the clustering results using a dendrogram.

        """
    def __init__(self,
                 prompts,
                 nid_calculator: InformationDistanceCalculator = NCDCalculator(compression_method='bz2'),
                 reference_builder_mode=None):
        """
        Initialize the clusterer with a list of prompts and a distance calculator.

        Args:
            prompts (list of str): The list of prompts to be clustered.
            distance_calculator (InformationDistanceCalculator): An instance of a class implementing the
            InformationDistanceCalculator interface.
            reference_builder_mode: Mode to use with ReferenceSentenceBuilder for creating a reference sentence.
        """
        self.prompts = prompts
        self.nid_calculator = nid_calculator
        self.reference_builder_mode = reference_builder_mode
        if reference_builder_mode:
            self.reference_builder = ReferenceSentenceBuilder(method=reference_builder_mode)
        else:
            self.reference_builder = None
        self.nid_calculator.train_dictionary(prompts)
        self.nid_matrix = self._compute_pairwise_nid_matrix(self.prompts)

    def _compute_pairwise_nid_matrix(self, prompts: List[str]) -> np.ndarray:
        """
        Compute a pairwise NID matrix (or vector if reference is used) for a list of prompts.

        :param prompts: List of prompts.
        :return: 2D NID matrix or 1D NID vector.
        """
        if self.reference_builder:
            reference_prompt = self.reference_builder.build_reference(prompts)
            return np.array([self.nid_calculator.compute_distance(prompt, reference_prompt) for prompt in prompts])

        n = len(prompts)
        nid_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):  # Only compute for unique pairs
                nid_value = self.nid_calculator.compute_distance(prompts[i], prompts[j])
                nid_matrix[i][j] = nid_value
                nid_matrix[j][i] = nid_value  # Symmetric matrix

        return nid_matrix

    def cluster_prompts(self, max_clusters):
        """
        Perform hierarchical clustering on the prompts.

        Returns:
            list of tuple: Linkage matrix suitable for use by scipy's dendrogram function.
        """
        linked = linkage(self.nid_matrix, 'single')

        # Cut the dendrogram to obtain clusters
        cluster_labels = fcluster(linked, max_clusters, criterion='maxclust')

        return cluster_labels

    def cluster(self, prompts: List[str], num_clusters: int) -> np.ndarray:
        """
        Cluster the prompts based on their NID values.

        :param prompts: List of prompts.
        :param num_clusters: Desired number of clusters.
        :return: An array of shape (n, 2) where n is the number of prompts. The first column contains cluster labels
                 and the second contains NID values.
        """
        if self.reference_builder:
            sorted_indices = np.argsort(self.nid_matrix)
            cluster_labels = np.arange(len(prompts))  # Each prompt is its own "cluster" when sorted by NID to reference
            return np.column_stack((cluster_labels, self.nid_matrix[sorted_indices]))

        # Use linkage to get hierarchical clustering
        linked = linkage(self.nid_matrix, method="single")
        cluster_labels = fcluster(linked, t=num_clusters, criterion="maxclust")

        # Get NID complexity values
        complexity_values = np.array([np.mean(self.nid_matrix[i]) for i in range(len(prompts))])

        return np.column_stack((cluster_labels, complexity_values))

    def visualize_dendrogram(self):
        linked = linkage(self.nid_matrix, 'single')
        dendrogram(linked)
        #plt.savefig("dendrogram.png")
        plt.show()

    def plot_dendrogram(self):
        """
        Plot a dendrogram to visualize the hierarchical clustering of prompts.

        """
        linked = linkage(self.nid_matrix, 'single')

        plt.figure(figsize=(10, 5))
        dendrogram(linked, orientation='top', labels=self.prompts, distance_sort='descending')
        plt.xticks(rotation=25)
        plt.show()


if __name__ == "__main__":
    # Demo prompts for illustration
    demo_prompts = [
        "The cat sat on the mat.",
        "A dog barked loudly.",
        "Birds fly in the sky.",
        "Fish swim in the water.",
        "The sun is bright and shiny.",
        "Stars twinkle at night.",
        "Trees have green leaves.",
        "Flowers bloom in spring.",
        "Snow falls in winter.",
        "Rain brings a rainbow.",
        "Winds blow during a storm."
    ]

    # Initialize the clusterer
    nid_calculator = GCDDCalculator()
    #nid_calculator.train_dictionary(demo_prompts)
    #nid_calculator = NCDCalculator(compression_method='bz2')
    clusterer = NIDHierarchicalClusterer(demo_prompts, nid_calculator=nid_calculator)

    # Cluster the prompts into a desired number of clusters
    max_clusters = 11
    clusters = clusterer.cluster_prompts(max_clusters)

    print(f"Cluster assignments for {max_clusters} clusters, method {nid_calculator.__class__.__name__}:")
    for prompt, cluster in zip(demo_prompts, clusters):
        print(f"'{prompt}' -> Cluster {cluster}")

    # Visualize the dendrogram
    clusterer.visualize_dendrogram()

    # Displaying the results
    print("Complexity Rankings (Cluster, Rank):")
    results = clusterer.cluster(demo_prompts, num_clusters=11)
    for sentence, (cluster, nid_value) in zip(demo_prompts, results):
        print(f"'{sentence}' -> Cluster {cluster}, NID Value: {nid_value:.2f}")


