import zstandard as zstd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

from src.kckit.nid.centroid_sentence_builder import ReferenceSentenceBuilder
from src.kckit.nid.nid_interface import InformationDistanceCalculator

class GCDDCalculator(InformationDistanceCalculator):
    def __init__(self, compressor='zstd', dict_size=131072): #110 << 10
        self.compressor = compressor
        self.dict_size = dict_size
        self.dict_data = None

    def train_dictionary(self, samples):
        # Train a dictionary using zstandard's train_dictionary method
        self.dict_data = zstd.train_dictionary(self.dict_size, [sample.encode() for sample in samples])

    def _compute_entropy(self, data):
        value, counts = np.unique(list(data), return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def compress_size(self, data):
        if not self.dict_data:
            raise ValueError("Dictionary not trained. Please train the dictionary first.")

        cctx = zstd.ZstdCompressor(dict_data=self.dict_data)
        return len(cctx.compress(data.encode()))

    def compute_gcdd(self, data1, data2):
        compressed_size_data1 = self.compress_size(data1)
        compressed_size_data2 = self.compress_size(data2)
        compressed_size_data1_data2 = self.compress_size(data1 + data2)

        gcdd = (compressed_size_data1_data2 - min(compressed_size_data1, compressed_size_data2)) / \
               max(compressed_size_data1, compressed_size_data2)

        return gcdd

    def compute_distance(self, data1, data2):
        return self.compute_gcdd(data1, data2)

    def rank_prompts(self, prompts, reference):
        gcdd_values = [self.compute_gcdd(reference, prompt) for prompt in prompts]
        ranked_prompts = [x for _, x in sorted(zip(gcdd_values, prompts), key=lambda pair: pair[0])]
        return ranked_prompts


def plot_dendrogram(sentences, reference):
    """
    Plot a dendrogram based on GCDD values of sentences relative to a reference.
    """
    gcdd_calculator = GCDDCalculator()
    gcdd_calculator.train_dictionary(sentences)
    distances = [gcdd_calculator.compute_gcdd(sentence, reference) for sentence in sentences]
    linked = linkage(np.array(distances).reshape(-1, 1), 'single')

    plt.figure(figsize=(10, 5))
    dendrogram(linked, orientation='top', labels=sentences, distance_sort='descending')
    plt.xticks(rotation=15)  # Rotate x labels by 15 degrees
    #plt.savefig("gcdd_dendrogram.png")
    plt.show()


if __name__ == '__main__':
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

    rsb = ReferenceSentenceBuilder()
    reference_sentence = rsb.build_reference(sentences)

    print(f"\nUsing compression method: zstd")
    gcdd_calculator = GCDDCalculator()
    gcdd_calculator.train_dictionary(sentences)
    sorted_sentences = gcdd_calculator.rank_prompts(sentences, reference_sentence)

    for sentence in sorted_sentences:
        print(sentence)

    plot_dendrogram(sentences, reference_sentence)
