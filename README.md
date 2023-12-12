# ngm-kc-eval

# Evaluating Robustness of Neural Generative Models using Kolmogorov Complexity

## Motivation
Neural generative models, such as T5 and GPT-2, have shown impressive performance in NLP tasks. However, their robustness against varied and perturbed inputs is critical, especially as they are increasingly integrated into essential applications. This project explores the resilience of these models using Kolmogorov Complexity, providing insights into their behavior under different conditions and enhancing their training for better data variability handling.

## Method
We propose to employ Kolmogorov Conditional Complexity (KCC) in two main dimensions: adversarial input perturbations and varying input complexities. Key components of our methodology include:
- Designing a set of inputs, ranked by their KCC, that range from simplistic to complex, both by perturbing existing data and by creating inputs of varying complexity.
- Analysing model responses to these inputs, emphasising attention patterns and output deviations.
- Using BERTScore for semantic evaluation.
- Employing metrics such as the Normalised Compression Distance (NCD) for informational coherence and isights into model's complexity dynamics.
- Introducing a robustness metric that integrates the NCD of input, attention, and output, coupled with the NCD of the inputs themselves, as a measure of model resilience.

We plan to employ the Kolmogorov Complexity Chain Rule as a fundamental framework. This rule will enable us to examine the changes in KCC across all interconnected components—inputs, model, and output. By doing so, we aim to establish a better understanding of how adversarial input perturbations and varying input complexities affect not just isolated parts, but the entire system as a cohesive unit. Our hypothesis is that this approach could unveil insights into the model's resilience and behaviour under varying conditions.

## Experiments
Our experiments will:
- **Input Exploration**: Subject the model to both adversarially perturbed and inherently varied inputs, classified by KCC.
- **Model Analysis**: Assessing T5 and GPT-2's response to perturbed and varied inputs using BERTScore and NCD.
- **Attention Dynamics**: Decode attention maps to understand model focus across different input complexities and perturbations.
- **Robustness Assessment**: Evaluating the impact of Kolmogorov Complexity in enhancing model resilience and performance under perturbed conditions.
- **Robustness Metric Computation**: We aim to construct a conceptual robustness metric that inherently situates the ML model within the realm of complexity, rather than merely applying complexity measures to isolated components. One such attempt is represented by:
- **Training Enhancement**: Exploring the incorporation of information complexity into model training stages like pre-training and fine-tuning.

Robustness Metric = f(NCD(input), NCD(attention), NCD(output))

where \( f \) is a customizable function that can be adapted based on experimental observations and specific nuances of data and model behaviour.

## Concerns
- **Model Interpretability**: While attention maps provide insight into model behaviour, their interpretation is not always straightforward.
- **Complexity of Inputs**: Ranking inputs based solely on their Kolmogorov complexity might not always correlate with the perceived complexity from the model's perspective.
- **Metric Sensitivity**: The robustness metric might be sensitive to the specific choice of the function \( f \).
- **Computational Constraints**: Computing Kolmogorov complexity might impose significant computational overhead.
- **Generalizability**: Insights drawn might not directly generalise to other generative models or architectures.

## Optional Additions
- **Relevant Dataset**: Standard NLP datasets will serve as our foundation for crafting perturbed and varied complexity input sets.
- **Prior Research**: Foundational insights come from "Algorithmic Complexity for Short Binary Strings Applied to Psychology: A Primer" by José Hernández-Orallo and "Kolmogorov Complexity and its Applications" by Li and Vitanyi.

## References
1. C.H. Bennett et al., "Information Distance," IEEE Trans. Information Theory, 44:4(1998), 1407–1423.
2. P. M. B. Vitányi, "Similarity and denoising." [Online]. Available: [https://www.jstor.org/stable/41739976](https://www.jstor.org/stable/41739976)
3. M. Li et al., "The Similarity Metric." [Online]. Available: [https://browse.arxiv.org/pdf/cs/0111054v2.pdf](https://browse.arxiv.org/pdf/cs/0111054v2.pdf)
4. "Examples of the computational analysis based on Kolmogorov complexity." [Online]. Available: [https://link.springer.com/article/10.1007/s11071-020-05771-8](https://link.springer.com/article/10.1007/s11071-020-05771-8)
5. "Normalised compression distance." [Online]. Available: [https://www.biorxiv.org/content/10.1101/2020.07.22.216242v5.full](https://www.biorxiv.org/content/10.1101/2020.07.22.216242v5.full)
6. A. Radford et al., "Language Models are Unsupervised Multitask Learners." OpenAI Blog, 2019.
7. I. Goodfellow et al., "Explaining and Harnessing Adversarial Examples." arXiv:1412.6572, 2014.
8. R. Cilibrasi and P. Vitányi, "Clustering by Compression." [Online]. Available: [https://arxiv.org/abs/cs/0312044](https://arxiv.org/abs/cs/0312044)
9. M. Li, P.M.B. Vitányi. "An Introduction to Kolmogorov Complexity and Its Applications," 3rd Ed., Springer-Verlag, New York, 2008.
10. A. Bogomolov et al., "Generalised compression dictionary distance as universal similarity measure." arXiv preprint arXiv:1410.5792.
11. G. Salton, M. J. McGill, "Introduction to Modern Information Retrieval." McGraw-Hill, Inc., 1986.
12. S. C. Johnson, "Hierarchical Clustering Schemes." Psychometrika, 32(3), 1967, 241–254.
13. A. K. Jain, R. C. Dubes, "Algorithms for Clustering Data." Prentice-Hall, Inc., 1988.
14. T. Zhang et al., "BERTScore: Evaluating Text Generation with BERT." arXiv:1904.09675v5. [Online]. Available: [https://arxiv.org/abs/1904.09675](https://arxiv.org/abs/1904.09675)

For a detailed overview of the study, refer to the final report: [Final Report](https://docs.google.com/document/d/1HE77P98Vj0eh4cUSzXHUqg2sO0EhIVlxVpEvhH2tfbk/edit?usp=sharing)


## How to run 

Ensure you are in the `scripts` directory before running these commands.

### 1. Extract Data from HANS
Command:
```bash
./extract_from_hans.sh -i ../data/hans/heuristics_evaluation_set.jsonl -t lexical_overlap -n temp1 -m 100 -o ../data/hans/lex_over_tmpl1_snt1_100.csv
```
- **Purpose**: Extracts specific templates from the HANS dataset based on the heuristic type (`lexical_overlap` in this case).
- **Output**: A CSV file with extracted data.

### 2. Cluster Prompts
Command:
```bash
./cluster_prompts.sh ../data/hans/lex_over_tmpl1_snt1_100.csv ../data/hans/nid_lex_over_tmpl1_snt1_100.csv ncd 11
```
- **Purpose**: Clusters the prompts using the specified NID (Normalized Information Distance) method (e.g., `ncd`).
- **Output**: A CSV file with clustered prompts.

### 3. Run the Experiment
Command:
```bash
./run_experiment.sh -i ../data/hans/ncd_11_lex_over_tmpl1_snt1_100.csv -m bert-base-uncased -o bert_base_ncd_11_lex_over_tmpl1_snt1_100.hdf
```
- **Purpose**: Feeds the clustered prompts into a specified model (e.g., `bert-base-uncased`).
- **Output**: An HDF5 file storing the model outputs and attention scores.

### 4. Run KC Analyzer
Command:
```bash
./run_kc_analyzer.sh -i ../tmp/gpt2_large_ncd_11_lex_over_tmpl1_snt1_100.hdf -k ncd -o ../tmp/gpt2_large_ncd_11_lex_over_tmpl1_snt1_100_results.hdf
```
- **Purpose**: Processes the model outputs and attention scores using the KC (Kolmogorov Complexity) Analyzer.
- **Output**: An HDF5 file with NID values and clustering information.

### 5. Visualization
- **Purpose**: Generate plots and visualizations based on the results.
- **Method**: Utilize the `NIDClusterVisualizer` class from the codebase.

