import os

import streamlit as st
import base64
from PIL import Image

from init_load_model import embedding_parse, store_vec
from rag_chat_from_llm import rag

save_pdf_dir = 'data/pdf'
st.set_page_config(page_title="PDF问答", page_icon="💖", layout="wide")

sysmenu = '''
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
'''
st.markdown(sysmenu,unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("上传pdf", type=["pdf"])
    if uploaded_file is not None:
        pdf_save_path = os.path.join(save_pdf_dir, uploaded_file.name)
        with open(pdf_save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if st.button('解析PDF'):
            if os.path.exists(pdf_save_path):
                # parse pdf
                img_output_dir = 'data/results'

                texts, texts_embedding, images_embedding = embedding_parse.parse(pdf_save_path, img_output_dir)
                texts_embedding_size = len(texts_embedding[0])
                image_embedding_size = len(images_embedding[0])
                image_files = os.listdir(img_output_dir)

                # store embedding to db
                store_vec.create_client_collection(texts_embedding_size, image_embedding_size)
                store_vec.store_image_text_embedding(texts, texts_embedding, img_output_dir, image_files,
                                                     images_embedding)
                st.write('解析PDF成功！')
            else:
                st.write('PDF文件不能为空！')

        # base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
        # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" type="application/pdf">'
        # st.markdown(pdf_display, unsafe_allow_html=True)

with col2:
    query = st.text_area('输入问句：', height=100)
    if st.button('问答'):
        if (query is not None) and (len(query) != 0):
            answer, text_hits, image_hits = rag(query)
            st.markdown('<table><tr><td bgcolor=DarkSeaGreen>最终答案</td></tr></table>', unsafe_allow_html=True)
            st.markdown(answer, unsafe_allow_html=True)
            st.markdown('<table><tr><td bgcolor=DarkSeaGreen>答案来自RAG最相关的文本和图片部分</td></tr></table>', unsafe_allow_html=True)

            st.markdown('<table><tr><td bgcolor=DarkSeaGreen>相关文本片段</td></tr></table>', unsafe_allow_html=True)
            for hit in text_hits:
                content = hit.payload['content']
                st.markdown(content, unsafe_allow_html=True)
                st.markdown('---------', unsafe_allow_html=True)

            st.markdown('<table><tr><td bgcolor=DarkSeaGreen>相关图片</td></tr></table>', unsafe_allow_html=True)
            for hit in image_hits:
                image_path = hit.payload['image_path']
                img = Image.open(image_path)
                st.image(img)
                st.markdown('---------', unsafe_allow_html=True)
        else:
            st.write("问句不能为空！")

