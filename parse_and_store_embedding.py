import os

from init_load_model import embedding_parse, store_vec

if __name__ == '__main__':

    pdf_path = r'data/中国经济金融展望报告.pdf'
    output_dir = r'data/results'

    texts, texts_embedding, images_embedding = embedding_parse.parse(pdf_path, output_dir)
    texts_embedding_size = len(texts_embedding[0])
    image_embedding_size = len(images_embedding[0])
    image_files = os.listdir(output_dir)

    # 6.store embedding to db
    store_vec.create_client_collection(texts_embedding_size, image_embedding_size)
    store_vec.store_image_text_embedding(texts, texts_embedding, output_dir, image_files, images_embedding)

