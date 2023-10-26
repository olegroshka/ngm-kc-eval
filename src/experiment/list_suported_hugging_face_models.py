import requests
from transformers import AutoTokenizer, AutoModel, CONFIG_MAPPING

class HFModeInfo:
    @staticmethod
    def list_available_models():
        return list(CONFIG_MAPPING.keys())

    @staticmethod
    def get_model_size_mb(model_name):
        model_url = f"https://huggingface.co/{model_name}/resolve/main/pytorch_model.bin"
        response = requests.head(model_url)
        size_in_bytes = int(response.headers['Content-Length'])
        size_in_megabytes = int(size_in_bytes / (1024 * 1024))
        return size_in_megabytes

if __name__ == "__main__":

    print("\nAvailable models in AutoModel:")
    for model_name in HFModeInfo.list_available_models():
        print(model_name)

    model_name = 'mistralai/Mistral-7B-v0.1'
    size = HFModeInfo.get_model_size_mb(model_name)
    print(f"\nModels size for {model_name}: ", size, "Mb")
