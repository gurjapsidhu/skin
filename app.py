import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# Load the trained model
MODEL_PATH = "skin_cancer_cnn.keras"  # Update with your actual model path
model = load_model(MODEL_PATH)

# Streamlit UI Configuration
st.set_page_config(page_title="Skin Cancer Detection", page_icon="🚑", layout="wide")

# Custom CSS for Modern UI
st.markdown(
    """
    <style>
        .stButton>button { background-color: #FF4B4B; color: white; font-size: 16px; }
        .stFileUploader>div>button { background-color: #FF914D; }
        .css-1d391kg { text-align: center; }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Awareness & Prevention"])

if page == "Home":
    # App Title
    st.title("🔬 AI-Powered Skin Cancer Detection")
    st.write("Upload an image to detect if the lesion is benign or malignant.")

    # File Uploader
    uploaded_file = st.file_uploader("Upload a Skin Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        # Preprocess Image
        img = np.array(image)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)
        
        # Make Prediction
        with st.spinner("Analyzing Image..."):
            time.sleep(2)
            prediction = model.predict(img)[0][0]
            
            if prediction < 0.5:
                label = "Benign (Non-Cancerous)"
                confidence = round((1 - prediction) * 100, 2)
                st.success(f"🎉 Prediction: {label} ({confidence}% Confidence)")
                st.balloons()
            else:
                label = "Malignant (Cancerous)"
                confidence = round(prediction * 100, 2)
                st.error(f"⚠️ Prediction: {label} ({confidence}% Confidence)")
                st.markdown("### 🚨 Immediate medical attention is advise
                st.markdown("<style>body {background-color: #ffdddd;}</style>", unsafe_allow_html=True)
                st.markdown("<h2 style='color:red;'>⚠️ Cancer Detected! Consult a doctor immediately.</h2>", unsafe_allow_html=True)
        
        # Button for analytics
        if st.button("📊 View Prediction Analytics"):
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            labels = ['Benign', 'Malignant']
            values = [1 - prediction, prediction]
            axs[0, 0].bar(labels, values, color=['green', 'red'])
            axs[0, 0].set_title("Bar Chart")
            axs[0, 1].pie(values, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])
            axs[0, 1].set_title("Pie Chart")
            axs[1, 0].hist(values, bins=5, color='blue', alpha=0.7)
            axs[1, 0].set_title("Histogram")
            axs[1, 1].scatter(labels, values, color=['green', 'red'])
            axs[1, 1].set_title("Scatter Plot")
            st.pyplot(fig)

elif page == "Awareness & Prevention":
    # Awareness Page
    st.title("📢 Skin Cancer Awareness & Prevention")
    st.write("Skin cancer is one of the most common cancers worldwide. Understanding its causes and prevention methods can save lives.")
    
    st.header("📌 What is Skin Cancer?")
    st.write("Skin cancer occurs when abnormal cells grow uncontrollably in the skin. The two major types are basal cell carcinoma and malignant melanoma.")
    
    st.header("📊 Global Statistics")
    st.write("Skin cancer cases are rising worldwide. Here’s a statistical overview:")
    
    fig, ax = plt.subplots()
    countries = ["USA", "India", "Australia", "UK", "Canada"]
    cases = [50000, 30000, 70000, 25000, 20000]
    ax.bar(countries, cases, color=['red', 'orange', 'blue', 'green', 'purple'])
    ax.set_xlabel("Country")
    ax.set_ylabel("Reported Cases")
    ax.set_title("Skin Cancer Cases by Country")
    st.pyplot(fig)
    
    st.header("☀️ Causes of Skin Cancer")
    st.write("- Overexposure to UV radiation from the sun or tanning beds\n- Genetic factors\n- Weak immune system\n- Exposure to harmful chemicals\n- Repeated exposure to radiation")
    
    st.header("🛡️ Prevention Tips")
    st.write("- Avoid excessive sun exposure, especially between 10 AM - 4 PM\n- Always wear sunscreen with SPF 30+\n- Wear protective clothing and sunglasses\n- Avoid tanning beds\n- Get regular skin check-ups")
    
    st.success("Early detection saves lives! If you notice any unusual changes in your skin, consult a doctor immediately.")
