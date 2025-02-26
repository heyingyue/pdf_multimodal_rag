#coding: utf-8

import fitz
import os
import time
import cv2
import magic
from pdf2image import convert_from_path
import numpy as np
from PIL import Image

from src.table_det import TableDetector
from src.table_det.utils.extract_img import extract_table_img

class PDFExtract:
    def __init__(self, table_det_model):
        self.table_det_model = table_det_model

    def extract_images_from_pdf(self, pdf_path, output_dir):
        pdf_doc = fitz.open(pdf_path)
        # os.makedirs(output_dir, exist_ok=True)
        for p_num in range(len(pdf_doc)):
            page = pdf_doc[p_num]
            imgs = page.get_images(full=True)
            for img_idx, img in enumerate(imgs):
                xref = img[0]
                base_img = pdf_doc.extract_image(xref)
                img_bytes = base_img['image']
                img_ext = base_img['ext']

                img_filename = f"page_{p_num+1}_image_{img_idx+1}.{img_ext}"
                img_path = os.path.join(output_dir, img_filename)

                with open(img_path, 'wb') as img_file:
                    img_file.write(img_bytes)
                print('Saved: {}'.format(img_path))
        pdf_doc.close()

    def extract_text_from_pdf(self, pdf_path):
        pdf_doc = fitz.open(pdf_path)
        texts = []
        for page in pdf_doc:
            text = page.get_text()
            texts.append(text)
        return texts

    def extract_table_from_pdf(self, pdf_path, output_dir):

        pdf_doc = fitz.open(pdf_path)
        # os.makedirs(output_dir, exist_ok=True)
        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            pix = page.get_pixmap(dpi=120, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # 创建PIL图像对象
            img = np.array(img)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            result = self.table_det_model(img_bgr)
            for table_num, res in enumerate(result):
                file_name = f"page_{page_num + 1}_table_{table_num + 1}.png"
                img_path = os.path.join(output_dir, file_name)
                lt, rt, rb, lb = res["lt"], res["rt"], res["rb"], res["lb"]
                wrapped_img = extract_table_img(img_bgr, lt, rt, rb, lb)
                cv2.imwrite(img_path, wrapped_img)
                print('Saved: {}'.format(img_path))
        pdf_doc.close()

        # pdf_images = convert_from_path(pdf_path)
        # os.makedirs(output_dir, exist_ok=True)
        # for page_num, image in enumerate(pdf_images):
        #     img = np.array(image)
        #     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     result = self.table_det_model(img_bgr)
        #     for table_num, res in enumerate(result):
        #         file_name = f"page_{page_num + 1}_table_{table_num + 1}.png"
        #         img_path = os.path.join(output_dir, file_name)
        #         lt, rt, rb, lb = res["lt"], res["rt"], res["rb"], res["lb"]
        #         wrapped_img = extract_table_img(img_bgr, lt, rt, rb, lb)
        #         cv2.imwrite(img_path, wrapped_img)
        #         print('Saved: {}'.format(img_path))

    def extract_pdf(self, pdf_path, output_dir):
        if not os.path.exists(pdf_path) and self.is_pdf(pdf_path):
            raise ValueError("Not find file path or not pdf file : {}".format(
                pdf_path))
        else:
            st_time = time.time()
            # image
            self.extract_images_from_pdf(pdf_path, output_dir)
            # text
            texts = self.extract_text_from_pdf(pdf_path)
            # table
            self.extract_table_from_pdf(pdf_path, output_dir)
            consp_time = time.time() - st_time
            print("Time taken to extract text and images: {} seconds".format(consp_time))
            return texts, consp_time

    def is_pdf(self, file_path):
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        return file_type == 'application/pdf'

