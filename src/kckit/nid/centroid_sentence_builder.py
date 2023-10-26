from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ReferenceSentenceBuilder:
    def __init__(self, method='hybrid'):
        """
        Initialize the ReferenceSentenceBuilder with a specified method.

        :param method: Method to use for building the reference sentence.
                       Supported methods: 'centroid', 'tfidf', 'hybrid'.
        """
        self.method = method

    def build_reference(self, prompts):
        """
        Build a reference sentence from a list of prompts using the specified method.

        :param prompts: List of string prompts.
        :return: Reference sentence as a string.
        """
        if self.method == 'centroid':
            return self._centroid_of_clusters(prompts)
        elif self.method == 'tfidf':
            return self._tfidf_weighted_sentence(prompts)
        elif self.method == 'hybrid':
            return self._hybrid_reference(prompts)
        else:
            raise ValueError("Unsupported method")

    def _centroid_of_clusters(self, prompts):
        """
        Find the reference sentence closest to the centroid of a cluster of prompts.

        :param prompts: List of string prompts.
        :return: Reference sentence as a string.
        """
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(prompts)
        kmeans = KMeans(n_clusters=min(5, len(prompts)))
        kmeans.fit(X)
        centroid = kmeans.cluster_centers_[0]
        closest_index = np.argmax(cosine_similarity(X, centroid.reshape(1, -1)))
        return prompts[closest_index]

    def _tfidf_weighted_sentence(self, prompts):
        """
        Construct a reference sentence using words with the highest TF-IDF scores.

        :param prompts: List of string prompts.
        :return: Reference sentence as a string.
        """
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(prompts)
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(X.toarray()).flatten()[::-1]
        top_n_words = feature_array[tfidf_sorting][:5]  # Taking top 5 words
        return ' '.join(top_n_words)

    def _hybrid_reference(self, prompts):
        """
        A hybrid method that combines centroid-based and TF-IDF methods.

        :param prompts: List of string prompts.
        :return: Reference sentence as a string.
        """
        cluster_reference = self._centroid_of_clusters(prompts)
        tfidf_reference = self._tfidf_weighted_sentence(prompts)
        return cluster_reference  # Using the centroid-based reference in this setup


if __name__ == '__main__':
    prompts = [
        "The cat sat on the mat.",
        "Birds fly in the sky.",
        "Stars twinkle at night.",
        "Fish swim in the water.",
        "The sun is bright and shiny."
    ]
    builder = ReferenceSentenceBuilder(method='hybrid')
    reference_sentence = builder.build_reference(prompts)
    print("Reference Sentence:", reference_sentence)
