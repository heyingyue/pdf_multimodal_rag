import os
from PIL import Image

class EmbeddingParse:
    def __init__(self, pdf_extract, text_split, text_embedding, image_embedding):
        self.pdf_extract = pdf_extract
        self.text_split = text_split
        self.text_embedding = text_embedding
        self.image_embedding = image_embedding

    def parse(self, pdf_path, output_dir):
        print('start parse pdf...')
        texts, consp_time = self.pdf_extract.extract_pdf(pdf_path, output_dir)
        texts = self.text_split.split_texts(texts)
        texts = self.text_split.add_meta_data(texts, pdf_path)

        if len(texts) !=0:
            texts_embedding = [self.text_embedding.get_text_embedding(text.page_content) for text in texts]
            # texts_embedding_size = len(texts_embedding[0])

        images_embedding = []
        image_files = os.listdir(output_dir)
        for img in image_files:
            try:
                image = Image.open(os.path.join(output_dir, img))
                image_embedding = self.image_embedding.get_image_embedding(image)
                images_embedding.append(image_embedding)
            except Exception as e:
                print(f'Error processing: {img}: {e}')
        # image_embedding_size = len(images_embedding[0])

        return texts, texts_embedding, images_embedding

