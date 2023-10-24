# ngm-kc-eval

# CS229 Project: Evaluating Robustness of Neural Generative Models using Kolmogorov Complexity

## Category
Theory & Reinforcement Learning 

## Team Members
- Zongyuan C Li 
- Annika Sofia Mauro 
- Oleg Roshka

## Motivation
Neural generative models have displayed commendable performance in NLP tasks. Yet, their resilience against varying and adversarially perturbed inputs remains a pertinent question. As the integration of such models in critical applications escalates, understanding their behaviour and vulnerabilities is of utmost importance. Our project is poised to explore the robustness of the pretrained models, like Mistral-7B, Llama-2-7b, Microsoft/Phi-1_5, BERT, utilising Kolmogorov Complexity.

## Method
We propose to employ Kolmogorov Conditional Complexity (KCC) in two main dimensions: adversarial input perturbations and varying input complexities. Key components of our methodology include:
- Designing a set of inputs, ranked by their KCC, that range from simplistic to complex, both by perturbing existing data and by creating inputs of varying complexity.
- Analysing model responses to these inputs, emphasising attention patterns and output deviations.
- Employing metrics such as the Normalised Compression Distance (NCD).
- Introducing a robustness metric that integrates the NCD of input, attention, and output, coupled with the NCD of the inputs themselves, as a measure of model resilience.

We plan to employ the Kolmogorov Complexity Chain Rule as a fundamental framework. This rule will enable us to examine the changes in KCC across all interconnected components—inputs, model, and output. By doing so, we aim to establish a better understanding of how adversarial input perturbations and varying input complexities affect not just isolated parts, but the entire system as a cohesive unit. Our hypothesis is that this approach could unveil insights into the model's resilience and behaviour under varying conditions.

## Intended Experiments
Our experiments will:
- **Input Exploration**: Subject the model to both adversarially perturbed and inherently varied inputs, classified by KCC.
- **Attention Dynamics**: Decode attention maps to understand model focus across different input complexities and perturbations.
- **Robustness Metric Computation**: We aim to construct a conceptual robustness metric that inherently situates the ML model within the realm of complexity, rather than merely applying complexity measures to isolated components. One such attempt is represented by:

\[ \text{Robustness Metric} = f(\text{NCD\_input}, \text{NCD\_attention}, \text{NCD\_output}) \]

where \( f \) is a customizable function that can be adapted based on experimental observations and specific nuances of data and model behaviour.

## Concerns
- **Model Interpretability**: While attention maps provide insight into model behaviour, their interpretation is not always straightforward.
- **Complexity of Inputs**: Ranking inputs based solely on their Kolmogorov complexity might not always correlate with the perceived complexity from the model's perspective.
- **Metric Sensitivity**: The robustness metric might be sensitive to the specific choice of the function \( f \).
- **Calibration**: The function \( f \) needs careful design.
- **Computational Constraints**: Computing Kolmogorov complexity might impose significant computational overhead.
- **Generalizability**: Insights drawn might not directly generalise to other generative models or architectures.

## Optional Additions
- **Relevant Dataset**: Standard NLP datasets will serve as our foundation for crafting perturbed and varied complexity input sets.
- **Prior Research**: Foundational insights come from "Algorithmic Complexity for Short Binary Strings Applied to Psychology: A Primer" by José Hernández-Orallo and "Kolmogorov Complexity and its Applications" by Li and Vitanyi.

## References
1. [Similarity and denoising by P. M. B. Vitányi](https://www.jstor.org/stable/41739976)
2. [The Similarity Metric by M. Li, X. Chen, X. Li, B. Ma, and P. M. B. Vitányi](https://browse.arxiv.org/pdf/cs/0111054v2.pdf)
3. [Examples of the computational analysis based on Kolmogorov complexity](https://link.springer.com/article/10.1007/s11071-020-05771-8)
4. [Normalised compression distance](https://www.biorxiv.org/content/10.1101/2020.07.22.216242v5.full)
5. [Language Models are Unsupervised Multitask Learners by A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever](https://openai.com/blog/better-language-models/)
6. [Explaining and Harnessing Adversarial Examples by I. Goodfellow, J. Shlens, and C. Szegedy](https://arxiv.org/abs/1412.6572)

