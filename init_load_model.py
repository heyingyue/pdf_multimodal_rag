from src.parse.image_embedding import ImageEmbedding
from src.parse.main_parse_embedding import EmbeddingParse
from src.parse.pdf_extraction import PDFExtract
from src.parse.split_texts import TextSplit
from src.parse.text_embedding import TextEmbedding
from src.store.store_vector import StoreVec
from src.table_det import TableDetector

# 1.pdf extract
table_det_model_path = 'weights/table_det.onnx'
table_det_model = TableDetector(model_path=table_det_model_path)
pdf_extract = PDFExtract(table_det_model)

# 2.text split
text_split = TextSplit()

# 3.text embedding
text_embedding_model_path = 'weights/nomic-ai/nomic-embed-text-v1.5'
text_embedding = TextEmbedding(text_embedding_model_path)

# 4.image embedding
image_embedding_model_path = 'weights/nomic-ai/nomic-embed-vision-v1.5'
image_embedding = ImageEmbedding(image_embedding_model_path)

# 5.parse and store
embedding_parse = EmbeddingParse(pdf_extract, text_split, text_embedding, image_embedding)

# 6.store embedding to db
store_vec = StoreVec('data/save_embedding/')
