from rembg import remove
from PIL import Image
import io

def remove_background(input_image):
    # Convert Django uploaded file to PIL Image
    pil_image = Image.open(input_image)
    # Use rembg to remove the background
    output_image = remove(pil_image)
    # Convert PIL Image back to bytes
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr