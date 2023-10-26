import  torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

MICROSOFT_PHI_1_5 = "microsoft/phi-1_5"

LLAMA_B_CHAT_HF = "meta-llama/Llama-2-7b-chat-hf"
FACEBOOK_BART_BASE = 'facebook/bart-base'
T5_SMALL = 't5-small'
BERT_BASE_UNCASED = 'bert-base-uncased'
MISTRAL_B_V = 'mistralai/Mistral-7B-v0.1'


class AttentionScoresDemo:
    def __init__(self, model_name=MICROSOFT_PHI_1_5):
        self.model_name = model_name
        #torch.set_default_device("cuda")
        #model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        #tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_attention_scores(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
        output = self.model(**inputs)
        #print("output: ", output)
        return output.attentions

if __name__ == "__main__":
    demo = AttentionScoresDemo()
    input_text = "Your text here"
    attention_scores = demo.get_attention_scores(input_text)
    print("attention_scores:", attention_scores)
