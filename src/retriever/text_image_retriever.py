import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class TextImageRetriever:
    def __init__(self, q_client, text_embedding):
        self.q_client = q_client
        self.text_embedding = text_embedding

    def retriever(self, query):
        query = self.text_embedding.get_text_embedding(query)
        text_hits = self.q_client.query_points(
            collection_name='texts',
             query=query,
            limit=5
        ).points

        image_hits = self.q_client.query_points(
            collection_name='images',
            query=query,
            limit=5
        ).points

        return text_hits, image_hits

    def display(self, text_hits, image_hits, text_trunc_length=150):

        """
        display text and image results from a retriever.

        Parameters:
        ----------
        text_hits : list
            List of text results with `id`, `payload`, and `score`.
        image_hits : list
            List of image results with `id`, `payload`, and `score`.
        text_trunc_length : int, optional
            Maximum length for displayed text content (default is 150).

        display:
        --------
        - Text results in bold with IDs, truncated content, and scores.
        - Image results in a matplotlib plot with scores in titles.
        """

        print("\033[1mText Results:\033[0m")
        for i, hit in enumerate(text_hits, 1):
            print("NODEID:", hit.id)
            content = hit.payload['content']
            truncated_content = content[:text_trunc_length] + "..." if len(content) > text_trunc_length else content

            bold_truncated_content = f"\033[1m{truncated_content}"
            print(f"{i}. {bold_truncated_content} | Score: {hit.score}")

        print("\nImage Results:")

        fig, axes = plt.subplots(1, len(image_hits), figsize=(15, 5))  # Adjust figsize as needed
        for ax, hit in zip(axes, image_hits):
            image_path = hit.payload['image_path']
            print(f"Displaying image: {image_path} | Score: {hit.score}")

            img = mpimg.imread(image_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Score: {hit.score}", fontsize=10)

        plt.suptitle("Image Results", fontsize=16)
        plt.tight_layout()
        plt.show()



