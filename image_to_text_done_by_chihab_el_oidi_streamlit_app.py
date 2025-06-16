
"""# Image-To-Text : Streamlit Demo"""

from transformers import pipeline
import streamlit as st
from PIL import Image

# Initialize the model only once using Streamlit cache
@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="ydshieh/vit-gpt2-coco-en")

captioner = load_model()

st.title('Image to Text')

def generate_caption(upload_file):
    """Generate caption from uploaded image"""
    if upload_file is None:
        return None, "Please upload an image first"

    try:
        image = Image.open(upload_file)
        result = captioner(image)[0]['generated_text']
        return image, result
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# File uploader outside the form for better UI flow
upload_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display image immediately after upload
if upload_file is not None:
    st.image(upload_file, caption="Uploaded Image", use_column_width=True)

# Separate caption generation button
if st.button('Extract Caption', disabled=(upload_file is None)):
    if upload_file:
        image, caption = generate_caption(upload_file)
        if image:
            st.subheader('Generated Caption:')
            st.success(caption)
    else:
        st.warning("Please upload an image first")
