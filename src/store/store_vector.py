import os

from qdrant_client import QdrantClient, models
import numpy as np
import uuid

class StoreVec:
    def __init__(self, embedding_vec_path):
        print('init qdrant client...')
        os.makedirs(embedding_vec_path, exist_ok=True)
        # self.q_client = QdrantClient(':memory:') # 本地内存，无持久化到本地
        self.q_client = QdrantClient(path=embedding_vec_path) # embedding持久化到本地

    def create_client_collection(self, text_embedding_size, image_embedding_size):
        print('create qdrant client...')
        if not self.q_client.collection_exists(('texts')):
            self.q_client.create_collection(
                collection_name='texts',
                vectors_config=models.VectorParams(
                    size=text_embedding_size,
                    distance=models.Distance.COSINE
                )
            )

        if not self.q_client.collection_exists('images'):
            self.q_client.create_collection(
                collection_name='images',
                vectors_config=models.VectorParams(
                    size=image_embedding_size,
                    distance=models.Distance.COSINE
                )
            )

    def store_image_text_embedding(self, texts, texts_embedding, output_dir, image_files, images_embedding):
        print('store embedding to db...')
        self.q_client.upload_points(
            collection_name='texts',
            points=[
                models.PointStruct(
                    id=text.metadata['uuid'],
                    vector=np.array(texts_embedding[idx]),
                    payload={
                        'metadata': text.metadata,
                        'content': text.page_content
                    }
                )
                for idx, text in enumerate(texts)
            ]
        )

        if len(images_embedding) > 0:
            self.q_client.upload_points(
                collection_name='images',
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=np.array(images_embedding[idx]),
                        payload={
                            'image_path': output_dir + '/' + str(image_files[idx])
                        }
                    )
                    for idx in range(len(image_files))
                ]
            )


