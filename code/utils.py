import base64
from base64 import b64decode
from PIL import Image
import io

def display_base64_image(base64_code):
    """
    Function to display the image from base64 version.
    """
    image_data = base64.b64decode(base64_code)
    return Image.open(io.BytesIO(image_data))

def parse_docs_for_images_and_texts(docs):
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc.page_content)
            b64.append(doc.page_content)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}

