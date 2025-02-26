from openai import OpenAI
import base64

from init_load_model import store_vec
from retriever_embedding_from_local import text_image_retriever

# llm部分需要调用第三方的或者自己部署的多模态模型

key = '第三方key'
llm_client = OpenAI(api_key=key, base_url="http://maas-api.cn-huabei-1.xf-yun.com/v1")
print(llm_client.models.list())

def text_image_rag(context: list, images: list, query: str):

    generation_prompt = f"""
    根据给定的上下文，必须回答用户的提问，上下文可以是表格、文本或图像。根据上下文给出答案。用户提问是: {query}
    上下文是: {context}\n
    输出:
    """

    # Helper function to encode an image as a base64 string
    def encode_image(image_path):
        if image_path:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        return None


    image_paths = images
    messages = [
        {
            "role": "system",
            "content": "你是一个PDF内容理解助手。"
        },
        {
            "role": "user",
            "content": generation_prompt,
        }
    ]

    # Encode images and add them to the messages if present
    for image_path in image_paths:
        img_base64 = encode_image(image_path)
        if img_base64:
            # Adding the image base64 string as part of the text content
            messages.append({
                "role": "user",
                "content": [
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{img_base64}"  # Send the base64-encoded image
                      },
                  },
              ],
            })

    chat_response = llm_client.chat.completions.create(
        model="xdeepseekr1",
        messages=messages,
        temperature=0.5,
        top_p=0.99
    )

    return chat_response.choices[0].message.content

def rag(query):
  text_hits, image_hits = text_image_retriever(query, store_vec.q_client)
  retrieved_images = [i.payload['image_path'] for i in image_hits]
  answer = text_image_rag(text_hits, retrieved_images, query)

  for hit in text_hits:
      content = hit.payload['content']
      print(f"内容: {content} | 分数: {hit.score}")

  for hit in image_hits:
      image_path = hit.payload['image_path']
      print(f"图片: {image_path} | 分数: {hit.score}")

  return answer, text_hits, image_hits

if __name__ == '__main__':
    query = '2024年第三季度GDP增长了多少？'
    answer,_,_ = rag(query)
    print('llm 的回答是以下内容：')
    print(answer)