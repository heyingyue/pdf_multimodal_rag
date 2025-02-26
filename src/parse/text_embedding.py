from transformers import AutoTokenizer, AutoModel
import torch

# nomic-ai/nomic-embed-text-v1.5
class TextEmbedding:
    def __init__(self, text_embedding_model_path):
        self.init_text_model(text_embedding_model_path)

    def init_text_model(self, text_embedding_model_path):
        print('init text embedding model...')
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_embedding_model_path, from_tf=False, local_files_only=True, trust_remote_code=True)
        self.text_model = AutoModel.from_pretrained(text_embedding_model_path, from_tf=False, local_files_only=True, trust_remote_code=True)
        self.text_model = self.text_model.eval()

    def get_text_embedding(self, text):
        text_inputs = self.text_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            text_outputs = self.text_model(**text_inputs)
        text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
        return text_embeddings[0].detach().numpy()
