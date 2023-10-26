import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list, dendrogram
import matplotlib.pyplot as plt
from typing import Union, List, Optional


class NIDHierarchicalClusterer:
    def __init__(self, nid_calculator, reference_sentence_builder=None, reference_mod=None):
        """
        Initialize the clusterer with the desired Normalized Information Distance (NID) calculator.

        :param nid_calculator: An instance of a class that calculates NID between two pieces of data.
        :param reference_sentence_builder: An instance of ReferenceSentenceBuilder if using a reference.
        :param reference_mod: The mode for ReferenceSentenceBuilder ('centroid', 'tfidf', 'hybrid').
        """
        self.nid_calculator = nid_calculator
        self.reference_sentence_builder = reference_sentence_builder
        self.reference_mod = reference_mod
        self.reference_sentence = None

    def _compute_pairwise_nid_matrix(self, prompts: List[str]) -> np.ndarray:
        """
        Compute a pairwise NID matrix (or vector if reference is used) for a list of prompts.

        :param prompts: List of prompts.
        :return: 2D NID matrix or 1D NID vector.
        """
        if self.reference_sentence:
            return np.array([self.nid_calculator.compute_nid(prompt, self.reference_sentence) for prompt in prompts])

        n = len(prompts)
        nid_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):  # Only compute for unique pairs
                nid_value = self.nid_calculator.compute_nid(prompts[i], prompts[j])
                nid_matrix[i][j] = nid_value
                nid_matrix[j][i] = nid_value  # Symmetric matrix

        return nid_matrix

    def cluster(self, prompts: List[str], num_clusters: int) -> np.ndarray:
        """
        Cluster the prompts based on their NID values.

        :param prompts: List of prompts.
        :param num_clusters: Desired number of clusters.
        :return: An array of shape (n, 2) where n is the number of prompts. The first column contains cluster labels and the second contains NID values.
        """
        if self.reference_sentence_builder and not self.reference_sentence:
            self.reference_sentence = self.reference_sentence_builder.build_reference_sentence(prompts,
                                                                                               method=self.reference_mod)

        nid_matrix = self._compute_pairwise_nid_matrix(prompts)

        if self.reference_sentence:
            sorted_indices = np.argsort(nid_matrix)
            cluster_labels = np.arange(len(prompts))  # Each prompt is its own "cluster" when sorted by NID to reference
            return np.column_stack((cluster_labels, nid_matrix[sorted_indices]))

        # Use linkage to get hierarchical clustering
        linked = linkage(nid_matrix, method="single")
        cluster_labels = fcluster(linked, t=num_clusters, criterion="maxclust")

        # Get NID complexity values
        complexity_values = np.array([np.mean(nid_matrix[i]) for i in range(len(prompts))])

        return np.column_stack((cluster_labels, complexity_values))

    def plot_dendrogram(self, prompts: List[str]):
        """
        Plot a dendrogram for the given prompts.

        :param prompts: List of prompts.
        """
        if self.reference_sentence:
            print("Cannot plot dendrogram when using a reference sentence.")
            return

        nid_matrix = self._compute_pairwise_nid_matrix(prompts)
        linked = linkage(nid_matrix, method="single")

        plt.figure(figsize=(10, 5))
        dendrogram(linked, orientation='top', labels=prompts, distance_sort='descending')
        plt.xticks(rotation=45)
        plt.show()


# Demo
if __name__ == "__main__":
    class DummyNIDCalculator:
        def compute_nid(self, data1, data2):
            return abs(len(data1) - len(data2))


    sentences = [
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

    clusterer = NIDHierarchicalClusterer(DummyNIDCalculator())
    results = clusterer.cluster(sentences, num_clusters=11)
    for sentence, (cluster, nid_value) in zip(sentences, results):
        print(f"'{sentence}' -> Cluster {cluster}, NID Value: {nid_value:.2f}")

    clusterer.plot_dendrogram(sentences)
