import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram

class NIDVisualizer:

    def __init__(self, file_path):
        self.file_path = file_path


    def _load_data(self):
        return pd.read_csv(self.file_path)
        # with pd.HDFStore(self.hdf_path, 'r') as store:
        #     data_frames = [store[key] for key in store.keys()]
        # return pd.concat(data_frames, ignore_index=True)

    def plot_dendrogram(self, data, title, column_name):
        linked = linkage(data, 'single')
        plt.figure(figsize=(10, 5))
        dendrogram(linked)
        plt.title(f"{title} Dendrogram for {column_name}")
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

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._load_data()
        self.layer_count = self._detect_layer_count()

    def _detect_layer_count(self):
        # Use the naming convention to determine how many layers exist in the data
        return sum([1 for col in self.data.columns if "attention_nid_" in col])

    def _load_data(self):
        return pd.read_csv(self.file_path)

    def plot_f1_vs_nid(self):
        data = self._load_data()
        plt.scatter(data["output_nid"], data["f1"])
        plt.title("F1 vs Output NID")
        plt.xlabel("NID")
        plt.ylabel("F1")
        plt.savefig("f1_vs_nid.png")
        plt.show()

    def plot_f1_vs_nid_by_cluster(self, cluster_column):
        data = self._load_data()
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=data, x="output_nid", y="f1", hue=cluster_column, palette="tab10", s=60)
        plt.title("F1 vs NID by Cluster")
        plt.xlabel("NID")
        plt.ylabel("F1")
        plt.savefig("f1_vs_nid_by_cluster.png")
        plt.show()

    def plot_f1_plain_vs_perturbed(self):
        data = self._load_data()
        # Assuming 'perturbation' is a boolean column indicating whether the row is perturbed
        sns.barplot(x='perturbation', y='f1', data=data)
        plt.title('BERTScore F1: Plain vs Perturbed')
        plt.savefig('bertscore_f1_plain_vs_perturbed.png')
        plt.show()

    def plot_hierarchical_clusters(self, col1, col2, title, model_name):
        data = self._load_data()
        linked = linkage(data[[col1, col2]], 'single')
        dendrogram(linked, orientation='top', labels=data, distance_sort='descending')
        plt.figure(figsize=(10, 5))
        plt.xticks(rotation=25)
        plt.title(f"{title} Hierarchical Clustering for {col1} and {col2} ({model_name})")
        plt.savefig(f"hierarchical_clustering-{col1}-{col2}-{model_name}.png")
        plt.show()

    def correlation_heatmap_between_metrics(self, model_name):
        data = self._load_data()
        metric_columns = ['precision', 'recall', 'f1', 'prompt_nid', 'output_nid', 'prompt_ref_nid', 'prompt_output_nid', 'ref_output_nid'] #, 'ref_nid', 'prompt_cluster', 'output_cluster']
        ax = sns.heatmap(data[metric_columns].corr(), annot=True, cmap='coolwarm')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
        plt.title(f'Correlation Between Metrics ({model_name}))')
        plt.savefig(f'correlation_between_metrics-{model_name}.png')
        plt.show()

    def scatter_plot(self, x_column, y_column, model_name, hue_column=None, title=None):
        data = self._load_data()
        sns.scatterplot(x=x_column, y=y_column, hue='perturbation', data=data)
        if title:
            plt.title(title)
        else:
            plt.title(f"Scatter plot of {x_column} vs {y_column}")
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.savefig(f"scatter_plot-{model_name}-{x_column}-{y_column}.png")
        plt.show()

    def plot_nid_correlations(self):
        df = self._load_data()
        nid_columns = ['prompt_nid', 'ref_nid', 'output_nid', 'prompt_ref_nid', 'prompt_output_nid', 'ref_output_nid']
        sns.heatmap(df[nid_columns].corr(), annot=True, cmap='coolwarm')
        plt.title('NID Distance Correlations')
        plt.savefig('nid_distance_correlations.png')
        plt.show()

    def clustered_scatter_plot(self, x_column, y_column, cluster_column):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=self.data, x=x_column, y=y_column, hue=cluster_column, palette="tab10", s=60)
        plt.title(f"Clustered Scatter plot of {x_column} vs {y_column}")
        plt.savefig(f"clustered_scatter_plot-{x_column}-{y_column}.png")
        plt.show()

    def cluster_distribution(self, cluster_column):
        sns.countplot(data=self.data, x=cluster_column, palette="viridis")
        plt.title(f"Cluster Distribution for {cluster_column}")
        plt.savefig(f"cluster_distribution-{cluster_column}.png")
        plt.show()

    def plot_box_plots(self, model_name):
        df = self.data
        # Split the DataFrame into plain and perturbed data
        plain_df = df[df['perturbation'] == False]
        perturbed_df = df[df['perturbation'] == True]

        # Add a column to indicate the type (Plain or Perturbed)
        plain_df['Type'] = 'Plain'
        perturbed_df['Type'] = 'Perturbed'

        # Combine the data for easier plotting
        combined_df = pd.concat([plain_df, perturbed_df])

        # Define the metrics for which you want to create box plots
        metrics = ['precision', 'recall', 'f1']

        # Create a box plot for each metric
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Type', y=metric, data=combined_df)
            plt.title(f'Box Plot of {metric} for Plain and Perturbed Data (model: {model_name})')
            plt.ylabel(metric)
            plt.xlabel('Data Type')
            plt.savefig(f'box_plot-{model_name}-{metric}.png')
            plt.show()

    def box_plot_by_cluster(self, value_column, cluster_column):
        plt.figure(figsize=(10, 8))
        sns.boxplot(data=self.data, x=cluster_column, y=value_column, palette="viridis")
        plt.title(f"Box plot of {value_column} by {cluster_column}")
        plt.savefig(f"box_plot-{value_column}-{cluster_column}.png")
        plt.show()

    def clustered_pair_plot(self, columns, cluster_column):
        sns.pairplot(data=self.data[columns + [cluster_column]], hue=cluster_column, palette="tab10")
        plt.savefig(f"pairplot_{cluster_column}_gpt2_large.png")
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
        plt.savefig(f"dendrogram-{column_name1}-{column_name2}.png")
        plt.show()

    def plot_model_trends(self, col1, col2, col3, model_name):
         # Load your data into DataFrames
         df_base = pd.read_csv("../../tmp/t5-base_squad_data_100x1_prt_out.csv")
         df_small = pd.read_csv("../../tmp/t5-small_squad_data_100x1_prt_out.csv")
         df_gpt2 = pd.read_csv("../../tmp/gpt2_squad_data_100x1_prt_out.csv")

         df_base_plain = df_base[df_base['perturbation'] == False]
         df_small_plain = df_small[df_small['perturbation'] == False]
         df_gpt2_plain = df_gpt2[df_gpt2['perturbation'] == False]

         df_base_prt = df_base[df_base['perturbation'] == True]
         df_small_prt = df_small[df_small['perturbation'] == True]
         df_gpt2_prt = df_gpt2[df_gpt2['perturbation'] == True]

         # Calculate mean metrics for each model
         mean_col1_base = df_base_plain[col1].mean()
         mean_col2_base = df_base_plain[col2].mean()
         mean_col3_base = df_base_plain[col3].mean()

         mean_col1_small = df_small_plain[col1].mean()
         mean_col2_small = df_small_plain[col2].mean()
         mean_col3_small = df_small_plain[col3].mean()

         mean_col1_gpt2 = df_gpt2_plain[col1].mean()
         mean_col2_gpt2 = df_gpt2_plain[col2].mean()
         mean_col3_gpt2 = df_gpt2_plain[col3].mean()

         #perturbed
         mean_col1_base_prt = df_base_prt[col1].mean()
         mean_col2_base_prt = df_base_prt[col2].mean()
         mean_col3_base_prt = df_base_prt[col3].mean()

         mean_col1_small_prt = df_small_prt[col1].mean()
         mean_col2_small_prt = df_small_prt[col2].mean()
         mean_col3_small_prt = df_small_prt[col3].mean()

         mean_col1_gpt2_prt = df_gpt2_prt[col1].mean()
         mean_col2_gpt2_prt = df_gpt2_prt[col2].mean()
         mean_col3_gpt2_prt = df_gpt2_prt[col3].mean()

         # Create a DataFrame for plotting
         df_metrics = pd.DataFrame({
             'Model': ['T5-small', 'T5-base', 'GPT-2'],
             f'{col1} (plain)': [mean_col1_small, mean_col1_base, mean_col1_gpt2],
             f'{col2} (plain)': [mean_col2_small, mean_col2_base, mean_col2_gpt2],
             f'{col3} (plain)': [mean_col3_small, mean_col3_base, mean_col3_gpt2],
             f'{col1} (perturbed)': [mean_col1_small_prt, mean_col1_base_prt, mean_col1_gpt2_prt],
             f'{col2} (perturbed)': [mean_col2_small_prt, mean_col2_base_prt, mean_col2_gpt2_prt],
             f'{col3} (perturbed)': [mean_col3_small_prt, mean_col3_base_prt, mean_col3_gpt2_prt]
         })

         # Plot line graphs
         plt.figure(figsize=(12, 6))

         for metric in [f'{col1} (plain)', f'{col2} (plain)', f'{col3} (plain)', f'{col3} (perturbed)', f'{col2} (perturbed)', f'{col3} (perturbed)']:
             plt.plot(df_metrics['Model'], df_metrics[metric], marker='o', label=metric)

         plt.title(f'Trend Analysis Over Different T5 Models {model_name}')
         plt.xlabel('Model')
         plt.ylabel('Metric Value')
         plt.legend()
         plt.grid(True)
         plt.savefig(f'trend_analysis-{col1}-{col2}-{col3}-{model_name}.png')
         plt.show()

if __name__ == "__main__":
    # Usage
    results_path = "../../tmp/t5-base_squad_data_100x1_prt_out.csv"
    model_name = "t5-base"

    #hdf_path = "../../tmp/gpt2_large_ncd_11_lex_over_tmpl1_snt1_100_results.hdf"
    visualizer = NIDClusterVisualizer(results_path)

    # Display visualizations for demonstration purposes
    #visualizer.plot_f1_vs_nid()
    #visualizer.scatter_plot("output_nid", "f1", model_name=model_name, hue_column="perturbation", title="F1 vs Output NID (t5-base)")
    #visualizer.scatter_plot("prompt_nid", "f1", model_name=model_name, hue_column="perturbation", title="F1 vs Input NID (t5-base)")
    #visualizer.plot_f1_plain_vs_perturbed()
    #visualizer.plot_box_plots("t5-base")
    #visualizer.scatter_plot("prompt_output_nid", "ref_output_nid", "t5-base", hue_column="perturbation", title="Input to Output NID vs Reference to Output NID (t5-base)")
    #visualizer.scatter_plot("prompt_ref_nid", "ref_output_nid", "t5-base", hue_column="perturbation",
    #                        title="Input to Reference NID vs Reference to Output NID (t5-base)")

    #visualizer.plot_model_trends("precision","recall", "f1", model_name=model_name)
    #visualizer.plot_model_trends("output_nid", "prompt_output_nid", "ref_output_nid", model_name=model_name)

    visualizer.correlation_heatmap_between_metrics(model_name=model_name)

    #visualizer.plot_dendrograms("prompt_nid", "output_nid")
    #visualizer.clustered_scatter_plot("prompt_nid", "output_nid", "prompt_cluster")
    #visualizer.clustered_scatter_plot("prompt_nid", "output_nid", "output_cluster")

    #visualizer.cluster_distribution("prompt_nid")
    #visualizer.box_plot_by_cluster("prompt_nid", "prompt_cluster")
    #visualizer.box_plot_by_cluster("output_nid", "output_cluster")
    #visualizer.clustered_pair_plot(["prompt_nid", "output_nid"], "prompt_cluster")
    #visualizer.clustered_pair_plot(["prompt_nid", "output_nid"], "output_cluster")

    #visualizer.plot_attention_layers()

