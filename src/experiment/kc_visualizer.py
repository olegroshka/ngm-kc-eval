import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram

class NIDVisualizer:

    def __init__(self, hdf_path):
        self.hdf_path = hdf_path


    def _load_data(self):
        with pd.HDFStore(self.hdf_path, 'r') as store:
            data_frames = [store[key] for key in store.keys()]
        return pd.concat(data_frames, ignore_index=True)

    def plot_dendrogram(self, data, title, column_name):
        linked = linkage(data, 'single')
        plt.figure(figsize=(10, 5))
        dendrogram(linked)
        plt.title(f"{title} Dendrogram for {column_name}")
        plt.show()

    def scatter_plot(self, x_column, y_column):
        data = self._load_data()
        plt.scatter(data[x_column], data[y_column])
        plt.title(f"Scatter plot of {x_column} vs {y_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def heatmap(self, layer_idx):
        data = self._load_data()
        attention_data = data.filter(like=f'attention_{layer_idx}')
        sns.heatmap(attention_data, cmap="YlGnBu")
        plt.title(f"Heatmap for Attention Layer {layer_idx}")
        plt.show()

    def line_plot(self, x_column, y_column):
        data = self._load_data()
        plt.plot(data[x_column], data[y_column])
        plt.title(f"Line plot of {x_column} vs {y_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def histogram(self, column_name):
        data = self._load_data()
        plt.hist(data[column_name], bins=50)
        plt.title(f"Histogram for {column_name}")
        plt.show()

    def box_plot(self, column_name):
        data = self._load_data()
        plt.boxplot(data[column_name])
        plt.title(f"Box plot for {column_name}")
        plt.show()


class NIDClusterVisualizer:

    def __init__(self, hdf_file_path):
        self.file_path = hdf_file_path
        self.data = self._load_data()
        self.layer_count = self._detect_layer_count()

    def _detect_layer_count(self):
        # Use the naming convention to determine how many layers exist in the data
        return sum([1 for col in self.data.columns if "attention_nid_" in col])

    def _load_data(self):
        with pd.HDFStore(self.file_path, 'r') as store:
            df_list = [store[key] for key in store.keys() if "/results" in key]
            return pd.concat(df_list, ignore_index=True)

    def clustered_scatter_plot(self, x_column, y_column, cluster_column):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=self.data, x=x_column, y=y_column, hue=cluster_column, palette="tab10", s=60)
        plt.title(f"Clustered Scatter plot of {x_column} vs {y_column}")
        plt.show()

    def cluster_distribution(self, cluster_column):
        sns.countplot(data=self.data, x=cluster_column, palette="viridis")
        plt.title(f"Cluster Distribution for {cluster_column}")
        plt.show()

    def box_plot_by_cluster(self, value_column, cluster_column):
        plt.figure(figsize=(10, 8))
        sns.boxplot(data=self.data, x=cluster_column, y=value_column, palette="viridis")
        plt.title(f"Box plot of {value_column} by {cluster_column}")
        plt.show()

    def clustered_pair_plot(self, columns, cluster_column):
        sns.pairplot(data=self.data[columns + [cluster_column]], hue=cluster_column, palette="tab10")
        #plt.savefig(f"pairplot_{cluster_column}_gpt2_large.png")
        plt.show()

    def plot_attention_layers(self):
        for i in range(self.layer_count):
            self.clustered_scatter_plot(f"attention_nid_{i}", "output_nid", f"attention_cluster_{i}")
            self.cluster_distribution(f"attention_cluster_{i}")
            self.box_plot_by_cluster(f"attention_nid_{i}", f"attention_cluster_{i}")
    def plot_dendrograms(self, column_name1, column_name2):
        """
        Plot dendrograms for input and output NID data side by side.
        """

        # Extract input and output NID data
        input_data = self.data[column_name1].values
        output_data = self.data[column_name2].values

        # Compute linkages for hierarchical clustering
        input_linkage = linkage(input_data.reshape(-1, 1), method='ward')
        output_linkage = linkage(output_data.reshape(-1, 1), method='ward')

        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # Input dendrogram
        dendrogram(input_linkage, ax=axs[0], orientation='left')
        axs[0].set_title(f"Dendrogram for {column_name1}")

        # Output dendrogram
        dendrogram(output_linkage, ax=axs[1], orientation='left')
        axs[1].set_title(f"Dendrogram for {column_name2}")

        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    # Usage
    hdf_path = "../../tmp/gpt2_ncd_11_lex_over_tmpl1_snt1_100_results.hdf"
    #hdf_path = "../../tmp/gpt2_large_ncd_11_lex_over_tmpl1_snt1_100_results.hdf"
    visualizer = NIDClusterVisualizer(hdf_path)

    # Display visualizations for demonstration purposes
    visualizer.plot_dendrograms("prompt_nid", "output_nid")
    visualizer.clustered_scatter_plot("prompt_nid", "output_nid", "prompt_cluster")
    visualizer.clustered_scatter_plot("prompt_nid", "output_nid", "output_cluster")
    #visualizer.cluster_distribution("prompt_nid")
    visualizer.box_plot_by_cluster("prompt_nid", "prompt_cluster")
    visualizer.box_plot_by_cluster("output_nid", "output_cluster")
    visualizer.clustered_pair_plot(["prompt_nid", "output_nid"], "prompt_cluster")
    visualizer.clustered_pair_plot(["prompt_nid", "output_nid"], "output_cluster")

    visualizer.plot_attention_layers()

