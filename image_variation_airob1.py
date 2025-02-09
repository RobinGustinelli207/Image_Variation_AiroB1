# -*- coding: utf-8 -*-
"""Image Variation AiroB1
"""

import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline
import gradio as gr
from PIL import Image

# Vérification du GPU ou CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Charger le modèle sur le bon device (GPU ou CPU)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

def generate_image(init_image, prompt, strength=0.75):
    """Génère une image à partir d'une image de départ et d'un prompt."""
    init_image = init_image.convert("RGB")
    init_image = init_image.resize((512, 512))

    # Génération de l'image avec le pipeline
    result = pipe(prompt=prompt, image=init_image, strength=strength).images[0]
    return result

# Interface Gradio
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(type="pil", label="Image de départ"),
        gr.Textbox(label="Prompt"),
        gr.Slider(0.1, 1.0, value=0.75, label="Strength"),
    ],
    outputs=gr.Image(label="Image générée"),
    title="Stable Diffusion Image-to-Image",
    description="Téléchargez une image, entrez un prompt, et générez une nouvelle image modifiée par IA.",
)

# Lancer l'application Gradio sur Render
iface.launch(server_port=int(os.environ.get("PORT", 7860)), server_name="0.0.0.0")
