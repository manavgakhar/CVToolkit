from torchvision import models, transforms
import torch
from PIL import Image
import streamlit as st

@st.cache()
def predict(image_path):
    
    #loading model
    model = models.resnet101(pretrained=True)
    
    # applying transformations
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])
    
    
    img = Image.open(image_path)
    batch = torch.unsqueeze(transform(img), 0)

    model.eval()
    out = model(batch)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
