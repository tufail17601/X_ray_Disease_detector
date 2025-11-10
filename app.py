import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---- Page Setup ----
st.set_page_config(
    page_title="X-Ray Classification App",
    page_icon="🩻",
    layout="wide",
)

# ---- Custom CSS ----
st.markdown("""
    <style>
        /* Main background */
        .main {
            background-color: #f8fafc;
            padding: 2rem;
        }

        /* Headings */
        h1, h2, h3 {
            text-align: center;
            color: #1e293b;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Image box */
        .img-container {
            display: flex;
            justify-content: center;
            padding: 15px;
            border-radius: 12px;
            background: #ffffff;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }

        /* Result card */
        .result-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        }

        .prediction-text {
            font-size: 1.2rem;
            color: #0f172a;
        }

        .confidence-text {
            font-size: 1rem;
            color: #475569;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.title("🩻 About the App")
st.sidebar.info(
    """
    This app uses a **VGG16-based CNN model** to classify **X-ray images** into 
    categories such as **Normal**, **Pneumonia**, or **Tuberculosis**.
    Upload an image and view instant predictions.
    """
)
st.sidebar.markdown("**👨‍💻 Developed by:** Muhammad Tufail")

# ---- Load model ----
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("xray_model.h5")
    return model

model = load_model()

# ---- Class names ----
class_names = ['Normal', 'Pneumonia', 'Tuberculosis']

# ---- App Title ----
st.title("🩻 AI-Powered X-Ray Classification System")
st.markdown("Upload a chest X-ray image below to get a real-time classification result.")

# ---- File Upload ----
uploaded_file = st.file_uploader("📤 Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # ---- Columns Layout ----
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("<div class='img-container'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded X-ray", width=320)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        with st.spinner('🔍 Analyzing the image... Please wait...'):
            # Preprocess image
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)
            pred_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

        # ---- Display Result ----
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("🎯 Prediction Result")
        st.markdown(
            f"<p class='prediction-text'>Predicted Class: <b>{class_names[pred_idx]}</b></p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p class='confidence-text'>Confidence: <b>{confidence*100:.2f}%</b></p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- Sidebar Chart ----
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions[0], color=["#34a0a4", "#da1111", "#52b788"])
    ax.set_ylabel("Confidence")
    ax.set_title("Model Confidence")
    st.sidebar.pyplot(fig)

else:
    st.info("👆 Please upload an image to begin classification.")
