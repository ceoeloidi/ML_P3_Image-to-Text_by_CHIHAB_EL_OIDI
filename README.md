# Image-to-Text Application done using Gradio and Streamlit

This project provides two implementations for generating text captions from images using a pre-trained Vision Transformer (ViT) and GPT-2 model from Hugging Face Transformers.

![Demo](https://github.com/ceoeloidi/ML_P3_Image-to-Text_by_CHIHAB_EL_OIDI/blob/44f370358a1f466c8a78aba93a51fb72fabe477b/demo/P34.gif) 

![Video](https://github.com/ceoeloidi/ML_P3_Image-to-Text_by_CHIHAB_EL_OIDI/blob/48e88cb885d9f3301d664a4d7fbd97f45e8f944c/ML_P3_Image-to-Text_by_CHIHAB_EL_OIDI_Gradio_Demo.mov)

  ## Try it Yourself

  

  - Gradio app: https://ceoeloidi-image-to-text-by-chihab-el-oidi-gradio.hf.space
    
  - Streamlit app: https://image-to-textbychihabeloidi.streamlit.app



![logo_FS](https://github.com/user-attachments/assets/6657add7-916a-4aff-a1a8-419b6aa9bf0f) 



## Table of Contents

- [Try it Yourself](#try-it-yourself)
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Create and activate a virtual environment:](#1-create-and-activate-a-virtual-environment)
  - [2. Install required packages:](#2-install-required-packages)
- [Running Demos](#running-demos)
  - [Gradio Demo](#gradio-demo)
  - [1. Try it yourself](#1try-it-yourself)
  - [2. Setup and Code](#2setup-and-code)
  - [Initialize the image captioning pipeline](#initialize-the-image-captioning-pipeline)
  - [Create Gradio interface](#create-gradio-interface)
  - [3. Result Demo](#3result-demo)
  - [4. Demo Animated](#4demo-animated)
- [Dependencies](#dependencies)


## Overview

The application uses the ydshieh/vit-gpt2-coco-en model which combines:

 - Vision Transformer (ViT) for image feature extraction
 - GPT-2 decoder for text generation

The project includes two different interface implementations:

  1. Gradio Demo: Simple interface for quick testing

  2. Streamlit Demo: More polished interface with better error handling


## Development Environment Setup

### Prerequisites

  - Python 3.7+
  - pip package manager

## Installation

#### 1. Create and activate a virtual environment:

        python -m venv venv
        source venv/bin/activate  # Linux/MacOS
        venv\Scripts\activate    # Windows

#### 2. Install required packages:
   
        pip install torch transformers gradio streamlit Pillow

## Running Demos

### Gradio Demo

    jupyter notebook "Image-To-Text done by CHIHAB EL OIDI.ipynb"

### 1.	Try it yourself
    
- https://ceoeloidi-image-to-text-by-chihab-el-oidi-gradio.hf.space
        
### 2.	Setup and Code

    from transformers import pipeline
    import gradio as gr
    from PIL import Image

  ## Initialize the image captioning pipeline
      captioner = pipeline(“image-to-text”, model=”ydshieh/vit-gpt2-coco-en”)

      def generate_caption(image):
      “””Generate caption from uploaded image”””
      if image is None:
        return None, “Please upload an image”
    
      ## Open image and generate caption
        img = Image.open(image)
        result = captioner(img)[0][‘generated_text’]
        return img, result  # Return both image and text

  ## Create Gradio interface
      with gr.Blocks(title=”Image To Text”) as app:
      gr.Markdown(“#Image to Text”)  # Optional header for display

      with gr.Row():
        with gr.Column():
            upload_file = gr.Image(type=”filepath”, label=”Upload Image”)
            submit = gr.Button(“Extract Caption”)
        
        with gr.Column():
            output_image = gr.Image(label=”Uploaded Image”, interactive=False)
            output_text = gr.Textbox(label=”Generated Caption”)

    submit.click(
        fn=generate_caption,
        inputs=upload_file,
        outputs=[output_image, output_text]
    )
    app.launch(share=True)  # Run the app




  ### 3.	Result Demo

  ![4](https://github.com/user-attachments/assets/27384cf4-3608-4cec-ac96-6227446cf6d0)


  ### 4.	Demo Animated

    
  ![Demo](https://github.com/ceoeloidi/ML_P3_Image-to-Text_by_CHIHAB_EL_OIDI/blob/44f370358a1f466c8a78aba93a51fb72fabe477b/demo/P34.gif)


## Dependencies

    torch==2.0.1
    torchvision==0.15.2
    matplotlib==3.7.1
    numpy==1.24.3
    tqdm==4.65.0
    pycocotools==2.0.7
    Pillow==9.5.0

### Done by CHIHAB EL OIDI
