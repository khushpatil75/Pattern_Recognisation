import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import Image
import json

IMG_SIZE = 128  # must match the trained model size

# Load Models
cnn = tf.keras.models.load_model("models/cnn_model.pth")
vit = tf.keras.models.load_model("models/vit_model.pth")
hybrid = tf.keras.models.load_model("models/hybrid_model.pth")

models_dict = {
    "CNN": cnn, 
    "Vision Transformer": vit, 
    "Hybrid": hybrid
}

# Load Labels
with open("labels_map.json", "r", encoding="utf-8") as f:
    labels_map = {v: k for k, v in json.load(f).items()}

st.title("✍️ Devnagari Handwritten Character Recognition")

st.write("Draw a Devnagari character inside the box below:")

canvas = st_canvas(
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=256,
    width=256,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):
    if canvas.image_data is not None:
        # Preprocess for CNN / ViT / Hybrid (3-channel expected)
        img = Image.fromarray(canvas.image_data.astype("uint8")).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

        st.subheader("🔍 Prediction Results:")

        results = []
        for name, model in models_dict.items():
            pred = model.predict(img, verbose=0)[0]
            cls = np.argmax(pred)
            label = labels_map[cls]
            confidence = pred[cls] * 100
            results.append([name, label, confidence])
            st.write(f"**{name} ➜ {label} ({confidence:.2f}%)**")

        # Best model suggestion
        best = max(results, key=lambda x: x[2])
        st.success(f"**Best Model: {best[0]} predicts → {best[1]}** ({best[2]:.2f}%)")
