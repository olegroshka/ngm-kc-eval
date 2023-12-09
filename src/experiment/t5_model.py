import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from src.experiment.ngm_model import NGMModel, ModelResult

class T5NGMModel(NGMModel):
    def __init__(self, model_name, max_new_tokens=64):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, output_attentions=True)
        self.max_new_tokens = max_new_tokens

    def infer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model.generate(inputs.input_ids, max_new_tokens=self.max_new_tokens, attention_mask=inputs.attention_mask, pad_token_id=self.tokenizer.eos_token_id)
        predicted_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # with torch.no_grad():
        #     attention_outputs = self.model(**inputs, output_attentions=True)
        #     attentions = attention_outputs.attentions

        return ModelResult(predicted_output, None)
