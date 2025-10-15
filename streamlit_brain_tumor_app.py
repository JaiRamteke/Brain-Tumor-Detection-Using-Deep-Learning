"""
Streamlit App: AI-Powered Brain Tumor Classification

Drop this file into your project folder and run:
    streamlit run streamlit_brain_tumor_app.py

Requirements (install via pip):
    pip install streamlit tensorflow pillow numpy matplotlib opencv-python

Notes:
- Place your trained model weights inside the `models/` folder with these filenames:
    - models/inceptionv3_weights.h5
    - models/alexnet_weights.h5
    - models/mlp_weights.h5
  If weights are not present, the app shows a placeholder and will not perform real predictions.
- The app includes a Grad-CAM implementation for visual explanations when using InceptionV3.

Author: Jai Ramteke
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess
from tensorflow.keras import Model

import streamlit as st
import matplotlib.pyplot as plt
import cv2

# -------------------------- Configuration --------------------------
IMAGE_SIZE = 128
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
CLASS_MAP = {0: 'glioma', 1: 'meningioma', 2: 'no_tumor', 3: 'pituitary_tumor'}

# Model filenames (expected)
MODEL_FILES = {
    'InceptionV3 (Transfer Learning)': MODEL_DIR / 'inceptionv3_weights.h5',
    'AlexNet (Custom CNN)': MODEL_DIR / 'alexnet_weights.h5',
    'MLP (Baseline)': MODEL_DIR / 'mlp_weights.h5'
}

# -------------------------- Utilities --------------------------
@st.cache_resource
def load_keras_model(weights_path: Optional[Path], base_model_name: str = 'inception') -> Optional[tf.keras.Model]:
    """Load model if weights exist. For InceptionV3 we rebuild architecture then load weights (weights should match architecture).
    If weights_path is None or file doesn't exist, return None.
    """
    if weights_path is None or not weights_path.exists():
        return None

    try:
        if base_model_name.lower().startswith('inception'):
            base = InceptionV3(include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), pooling='avg')
            base.trainable = False
            model = tf.keras.Sequential([
                base,
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax')
            ])
            model.load_weights(str(weights_path))
            return model
        else:
            # For custom AlexNet / MLP, we will attempt to load the saved model directly
            model = load_model(str(weights_path))
            return model
    except Exception as e:
        st.error(f"Error loading model from {weights_path}: {e}")
        return None


def preprocess_img(pil_img: Image.Image, target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE), for_model: str = 'generic') -> np.ndarray:
    img = pil_img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32')
    if for_model.lower().startswith('inception'):
        arr = inception_preprocess(arr)
    else:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model: tf.keras.Model, processed_img: np.ndarray) -> Tuple[int, float]:
    preds = model.predict(processed_img)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds, axis=1)[0])
    return class_idx, confidence


# -------------------------- Grad-CAM --------------------------
def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str) -> np.ndarray:
    # Ensure model ends with softmax output
    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(heatmap: np.ndarray, original_img: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = heatmap_color * alpha + original_img
    overlayed = overlayed / np.max(overlayed)
    return np.uint8(255 * overlayed)

# -------------------------- Streamlit UI --------------------------
st.set_page_config(page_title='AI Brain Tumor Classifier', layout='wide', initial_sidebar_state='expanded')

st.title('🧠 AI-Powered Brain Tumor Classifier — Streamlit')
st.markdown("""
Upload a brain MRI image (T2w preferred). The app will preprocess the image,
run it through the selected deep learning model, and show prediction + confidence.
For InceptionV3 we also provide Grad-CAM visual explanations.
""")

# Sidebar: Model selection and model loading
st.sidebar.header('Model Selection')
model_choice = st.sidebar.selectbox('Choose model', list(MODEL_FILES.keys()))

weights_path = MODEL_FILES[model_choice]
col1, col2 = st.columns([2, 3])

with st.sidebar.expander('Model Info', expanded=True):
    st.write(f"**Selected:** {model_choice}")
    st.write(f"**Weights path:** {weights_path}")
    if weights_path.exists():
        st.success('Weights file found')
    else:
        st.warning('Weights file **not found**. Place model weights in the models/ folder to enable real predictions.')

# Load model (if available)
base_name = 'inception' if 'inception' in model_choice.lower() else ('alexnet' if 'alex' in model_choice.lower() else 'mlp')
model = load_keras_model(weights_path if weights_path.exists() else None, base_model_name=base_name)

# Main: image upload
uploaded_file = st.file_uploader('Upload an MRI image (jpg/png):', type=['jpg', 'jpeg', 'png'])

if uploaded_file is None:
    st.info('Upload an MRI image to see predictions. You can also try the sample image below.')
    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        if st.button('Use sample image'):
            # create a blank sample image (placeholder) — in real project, include a sample file
            sample_img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color=(120, 120, 120))
            uploaded_file = st.experimental_set_query_params(sample='1')  # small hack to bypass
            st.session_state['__sample_img__'] = sample_img
            uploaded_file = None
    st.stop()

# Read PIL image
try:
    pil_img = Image.open(uploaded_file)
except Exception:
    # if sample stored
    pil_img = st.session_state.get('__sample_img__', None)
    if pil_img is None:
        st.error('Failed to read the uploaded image. Please upload a valid image file.')
        st.stop()

# Show original image and basic info
col_left, col_right = st.columns([2, 3])
with col_left:
    st.subheader('Input Image')
    st.image(pil_img, use_column_width=True)
    st.write('Format:', pil_img.format)
    st.write('Size:', pil_img.size)

# Preprocess and predict
with col_right:
    st.subheader('Prediction')
    processed = preprocess_img(pil_img, target_size=(IMAGE_SIZE, IMAGE_SIZE), for_model=base_name)

    if model is None:
        st.warning('Model weights not available — app cannot run real predictions.\nPlease place trained weights in the models/ folder.')
        st.info('You can still save this processed image for testing with offline scripts.')
        if st.button('Save processed image locally'):
            out_path = Path('processed_samples')
            out_path.mkdir(exist_ok=True)
            arr = (processed[0] * 255).astype('uint8') if base_name != 'inception' else ((processed[0] - processed[0].min()) / (processed[0].ptp()) * 255).astype('uint8')
            Image.fromarray(arr).save(out_path / 'sample_processed.jpg')
            st.success(f'Saved processed image to {out_path / "sample_processed.jpg"}')
    else:
        class_idx, confidence = predict(model, processed)
        pred_label = CLASS_MAP.get(class_idx, str(class_idx))
        st.metric('Predicted Class', pred_label.title(), delta=None)
        st.metric('Confidence', f"{confidence * 100:.2f}%")

        # Show soft probabilities
        probs = model.predict(processed)[0]
        prob_table = {CLASS_MAP[i].title(): float(probs[i]) for i in range(len(probs))}
        st.write('Prediction Probabilities:')
        st.table(prob_table)

        # Grad-CAM: only for InceptionV3-like models (we expect a conv layer)
        if base_name == 'inception':
            st.subheader('Grad-CAM Explanation')
            try:
                # find a conv layer name (heuristic)
                conv_layer_name = None
                for layer in reversed(model.layers[0].layers if hasattr(model.layers[0], 'layers') else model.layers):
                    if 'conv' in layer.name and len(layer.output_shape) == 4:
                        conv_layer_name = layer.name
                        break
                if conv_layer_name is None:
                    st.warning('Could not locate a convolutional layer to compute Grad-CAM.')
                else:
                    heatmap = make_gradcam_heatmap(processed, model, last_conv_layer_name=conv_layer_name)
                    original_arr = np.array(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))).astype('uint8')
                    overlay = overlay_heatmap_on_image(heatmap, original_arr)

                    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
                    ax[0].imshow(original_arr)
                    ax[0].set_title('Preprocessed Input')
                    ax[0].axis('off')
                    ax[1].imshow(overlay)
                    ax[1].set_title('Grad-CAM Overlay')
                    ax[1].axis('off')
                    st.pyplot(fig)
            except Exception as e:
                st.error(f'Grad-CAM generation failed: {e}')

# Footer: quick tips and save model
st.markdown('---')
left, right = st.columns([3, 1])
with left:
    st.write('**Tips:**')
    st.write('- Use T2-weighted MRI scans for better performance.')
    st.write('- Crop the brain region if possible — reduces background noise.')
    st.write('- For production, consider serving model via TensorFlow Serving and the Streamlit app as a frontend.')
with right:
    if st.button('Show project structure'):
        st.code("""
        project-root/
        ├── models/                       # Put your .h5 model weights here
        ├── streamlit_brain_tumor_app.py  # This Streamlit app
        ├── notebooks/                     # Jupyter notebooks used for training
        └── data/                          # (Optional) sample images or datasets
        """)

st.caption('If you want, I can: (a) generate the Flask/Streamlit deployment Dockerfile, (b) add user auth, or (c) build a simple frontend to accept batch uploads — tell me which.')
