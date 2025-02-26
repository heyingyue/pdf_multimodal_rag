from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

class TextSplit:
    def split_texts(self, texts, chunk_size=800, chunk_overlap=150):
        """
        使用递归字符分割器处理文本
        参数说明：
        - chunk_size：每个文本块的最大字符数，推荐 500-1000
        - chunk_overlap：相邻块之间的重叠字符数（保持上下文连贯），推荐 100-200
        """
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "。", "!", "?", "？", "！", "；", ";"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,  # 保留原始文档中的位置信息
        )

        texts_split = text_splitter.create_documents(texts)

        return texts_split

    def add_meta_data(self, texts_split, pdf_path):
        for i in range(len(texts_split)):
            uu_id = str(uuid.uuid4())
            texts_split[i].metadata['doc_info'] = pdf_path
            texts_split[i].metadata['uuid'] = uu_id
        return texts_split






