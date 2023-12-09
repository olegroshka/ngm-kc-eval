from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.transformations import CompositeTransformation, WordSwapEmbedding, WordSwapRandomCharacterDeletion, \
    WordSwapWordNet, WordSwapMaskedLM, WordSwapNeighboringCharacterSwap, WordSwapRandomCharacterSubstitution
from textattack.augmentation import Augmenter

class TextPerturbation:
    def __init__(self):
        # Combine different transformation methods
        transformations = CompositeTransformation([
            WordSwapWordNet(),
            #WordSwapEmbedding(),
            #WordSwapMaskedLM(masked_language_model="bert-base-uncased"),
            #WordSwapNeighboringCharacterSwap(),
            WordSwapRandomCharacterSubstitution(),
            WordSwapRandomCharacterDeletion()
        ])
        # Define constraints (optional, to prevent over-perturbing)
        constraints = [RepeatModification(), StopwordModification()]
        # Create an augmenter with the defined transformation and constraints
        self.augmenter = Augmenter(transformation=transformations, constraints=constraints)

    def generate_perturbations(self, input_text, num_perturbations=1):
        # Generate a specified number of perturbations
        perturbations = []
        for _ in range(num_perturbations):
            augmented_texts = self.augmenter.augment(input_text)
            # In case the augmenter returns multiple perturbations, take the first one
            perturbation = augmented_texts[0] if augmented_texts else input_text
            perturbations.append(perturbation)
        return perturbations


# Create a TextPerturbation instance
#perturbation = TextPerturbation()

# Input text
#input_text = "The quick brown fox jumps over the lazy dog."
#
# # Generate perturbations
# perturbations = perturbation.generate_perturbations(input_text, num_perturbations=5)
# print("input_text:")
# print(input_text)
# print("perturbations:")
# for p in perturbations:
#     print(p)
