# Image-to-Text Application done using Gradio and Streamlit

This project provides two implementations for generating text captions from images using a pre-trained Vision Transformer (ViT) and GPT-2 model from Hugging Face Transformers.

<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe id="js_video_iframe" src="https://jumpshare.com/embed/gmMDfSCZ9j5Oq4DqWoTQ" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

  ## Try it Yourself

  

  - Gradio app: https://ceoeloidi-image-to-text-by-chihab-el-oidi-gradio.hf.space
    
  - Streamlit app: https://image-to-textbychihabeloidi.streamlit.app



![logo_FS](https://github.com/user-attachments/assets/6657add7-916a-4aff-a1a8-419b6aa9bf0f) 



## Table of Contents

- [Try it Yourself](#try-it-yourself)
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Mount Google Drive in Colab:](#1-mount-google-drive-in-colab)
  - [2. Install required packages:](#2-install-required-packages)
- [Dataset Configuration](#dataset-configuration)
  - [Download the Tiny COCO dataset](#--download-the-tiny-coco-dataset)
  - [Place it in your Google Drive](#--place-it-in-your-google-drive-at)
  - [Expected directory structure](#--expected-directory-structure)
- [Code Structure](#code-structure)
  - [Setup & Configuration](#--setup--configuration)
  - [Dataset Preparation](#--dataset-preparation)
  - [Data Augmentation](#--data-augmentation)
  - [Model Architecture](#--model-architecture)
  - [Training Process](#--training-process)
  - [Evaluation & Retrieval](#evaluation--retrieval)
- [Usage](#usage)
- [Results](#results)
- [Saving & Loading Models](#saving--loading-models)
  - [Save trained model](#save-trained-model)
  - [Load for inference](#load-for-inference)
- [Customization](#customization)
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

    
  <div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe id="js_video_iframe" src="https://jumpshare.com/embed/gmMDfSCZ9j5Oq4DqWoTQ" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

  
### - Dataset Preparation

    - Verifies dataset existence and structure  
    - Uses COCO API for annotation handling
    - Creates dataset wrapper:
            
            class TinyCocoSSL(torchvision.datasets.CocoDetection):
                def __getitem__(self, index):
                img, _ = super().__getitem__(index)
                return self.transform(img), self.transform(img)



### - Data Augmentation

    - Applies stochastic transformations:
    
        transforms.Compose([
            transforms.RandomResizedCrop(config.img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



### - Model Architecture

        class SSLModel(nn.Module):
            def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                ...,    
                nn.Linear(128*16*16, 256)
            )

### - Training Process

 - Uses NT-Xent contrastive loss:
    
            def simple_contrastive_loss(z1, z2):
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)
                sim_matrix = torch.mm(z1, z2.T) / config.temperature
                return F.cross_entropy(sim_matrix, targets)


 - AdamW optimizer with learning rate 3e-4
 - Epoch progress tracking with tqdm
 - Loss visualization after each epoch


### - Evaluation & Retrieval

 - Generate image embeddings:
    
            with torch.no_grad():
                for idx in range(len(ssl_dataset)):
                    emb = model(view1.unsqueeze(0).to(device))
                    embeddings.append(emb.cpu())


 - Similarity search function:
    
            def find_similar(query_idx, num_results=3):
                query_emb = embeddings[query_idx]
                similarities = torch.mm(embeddings, query_emb.unsqueeze(1)).squeeze()
                _, indices = torch.topk(similarities, num_results+1)




## Usage

#### - Run all notebook cells sequentially
#### - Monitor training progress through loss plots
#### - After training, retrieve similar images:
    
        for _ in range(config.num_retrieval):
            random_query = random.randint(0, len(ssl_dataset) - 1)
            find_similar(random_query)


## Results

 - Final loss after 24 epochs: 2.05
 - Embedding dimension: 256
 - Sample retrieval shows semantically similar images


## Saving & Loading Models

#### Save trained model:

        torch.save(model.state_dict(), 'tiny_coco_ssl.pth')


#### Load for inference:

        model = SSLModel().to(device)
        model.load_state_dict(torch.load('tiny_coco_ssl.pth'))
        model.eval()


## Customization

#### Adjust these parameters for experimentation:

 - `config.img_size`: Input image size
 - `config.batch_size`: Training batch size
 - `config.temperature`: Contrastive loss temperature
 - Encoder architecture in SSLModel class
 - Augmentation pipeline in ContrastiveTransform


## Dependencies

    torch==2.0.1
    torchvision==0.15.2
    matplotlib==3.7.1
    numpy==1.24.3
    tqdm==4.65.0
    pycocotools==2.0.7
    Pillow==9.5.0

### Done by CHIHAB EL OIDI
