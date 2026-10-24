import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import random
from medmnist import PneumoniaMNIST

st.set_page_config(page_title="AI Medical Image Analysis", page_icon="🫁", layout="centered")

st.title("🫁 AI Chest X-Ray & Grad-CAM Diagnostic Portal")
st.markdown(
    "Automated deep learning triage for **Pneumonia Detection** with **Grad-CAM thermal localization** "
    "highlighting acute pulmonary opacities."
)

# 1. Define Enhanced CNN Architecture
class EnhancedMedicalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

# 2. Load or Auto-Train Model (Never fails if .pth is missing!)
@st.cache_resource
def load_or_train_resources():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    test_ds = PneumoniaMNIST(split="test", transform=transform, download=True)
    model = EnhancedMedicalCNN()

    if os.path.exists("chest_pneumonia_model.pth"):
        model.load_state_dict(torch.load("chest_pneumonia_model.pth", map_location=torch.device("cpu")))
    else:
        # Rapid 2-epoch auto-training if weights file wasn't uploaded
        train_ds = PneumoniaMNIST(split="train", transform=transform, download=True)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(2):
            for imgs, lbls in train_loader:
                lbls = lbls.squeeze().long()
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()
        torch.save(model.state_dict(), "chest_pneumonia_model.pth")

    model.eval()
    return model, test_ds

with st.spinner("Initializing AI Diagnostic Engine & Loading Medical Dataset..."):
    model, test_ds = load_or_train_resources()

# 3. Grad-CAM Function
def compute_gradcam(model, img_tensor):
    activations, gradients = [], []

    def forward_hook(module, inp, out):
        activations.append(out)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = model.conv3.register_forward_hook(forward_hook)
    h2 = model.conv3.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    probs = torch.softmax(output, dim=1).squeeze()
    pred_idx = torch.argmax(probs).item()

    model.zero_grad()
    output[0, pred_idx].backward()

    grads = gradients[0].detach().numpy()[0]
    fmaps = activations[0].detach().numpy()[0]
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(fmaps.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmaps[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    h1.remove()
    h2.remove()
    return cam, probs

# 4. Interactive Controls
st.subheader("1. Select a Patient Scan or Upload Custom X-Ray")
col1, col2 = st.columns(2)

with col1:
    if st.button("🧪 Load Healthy Patient Sample", use_container_width=True):
        normal_indices = [i for i in range(len(test_ds)) if test_ds[i][1].item() == 0]
        idx = random.choice(normal_indices)
        st.session_state["sample_tensor"] = test_ds[idx][0]
        st.session_state["sample_title"] = f"Test Patient #{idx} (Ground Truth: Normal)"

with col2:
    if st.button("🦠 Load Pneumonia Patient Sample", use_container_width=True):
        pneu_indices = [i for i in range(len(test_ds)) if test_ds[i][1].item() == 1]
        idx = random.choice(pneu_indices)
        st.session_state["sample_tensor"] = test_ds[idx][0]
        st.session_state["sample_title"] = f"Test Patient #{idx} (Ground Truth: Pneumonia)"

uploaded_file = st.file_uploader("Or upload any Chest Radiograph from your device (JPG/PNG):", type=["png", "jpg", "jpeg"])

input_tensor = None
scan_title = ""

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("L")
    custom_tf = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = custom_tf(pil_img)
    scan_title = f"Uploaded File: {uploaded_file.name}"
elif "sample_tensor" in st.session_state:
    input_tensor = st.session_state["sample_tensor"]
    scan_title = st.session_state["sample_title"]

# 5. Prediction & Side-by-Side Grad-CAM
if input_tensor is not None:
    st.divider()
    st.subheader(f"2. Diagnostic Results — {scan_title}")

    batch_tensor = input_tensor.unsqueeze(0)
    cam, probs = compute_gradcam(model, batch_tensor)

    normal_p = float(probs[0]) * 100
    pneumonia_p = float(probs[1]) * 100

    raw_img = (input_tensor.squeeze().numpy() * 0.5) + 0.5
    raw_up = cv2.resize(raw_img, (220, 220))
    raw_rgb = np.uint8(255 * np.stack([raw_up]*3, axis=-1))

    cam_up = cv2.resize(cam, (220, 220))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(raw_rgb, 0.55, heatmap, 0.45, 0)

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(raw_rgb, caption="Original Chest Radiograph", use_container_width=True)
    with img_col2:
        st.image(overlay, caption="Grad-CAM Disease Localization", use_container_width=True)

    if pneumonia_p > 50.0:
        st.error(f"⚠️ **AI Diagnosis: PNEUMONIA DETECTED** (Confidence: {pneumonia_p:.2f}%)")
    else:
        st.success(f"✅ **AI Diagnosis: NORMAL / CLEAR LUNGS** (Confidence: {normal_p:.2f}%)")

    st.write("#### Confidence Breakdown:")
    st.write(f"Normal (Clear Lungs): `{normal_p:.1f}%`")
    st.progress(int(normal_p))
    st.write(f"Pneumonia (Pathology): `{pneumonia_p:.1f}%`")
    st.progress(int(pneumonia_p))
