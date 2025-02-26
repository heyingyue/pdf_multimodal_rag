from transformers import AutoModel, AutoProcessor
import torch

# nomic-ai/nomic-embed-vision-v1.5
class ImageEmbedding:
    def __init__(self, image_embedding_model_path):
        self.init_image_model(image_embedding_model_path)

    def init_image_model(self, image_embedding_model_path):
        print('init image embedding model...')
        self.image_model = AutoModel.from_pretrained(image_embedding_model_path, from_tf=False, local_files_only=True, trust_remote_code=True)
        self.image_processor = AutoProcessor.from_pretrained(image_embedding_model_path, use_fast=True, from_tf=False, local_files_only=True, trust_remote_code=True)
        self.image_model.eval()

    def get_image_embedding(self, image):
        # image = Image.open('image_path')
        image_inputs = self.image_processor(images=image, return_tensors='pt')
        with torch.no_grad():
            image_outputs = self.image_model(**image_inputs)
        image_embeddings = image_outputs.last_hidden_state
        return image_embeddings.mean(dim=1).squeeze().cpu().numpy()