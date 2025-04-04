import streamlit as st
import numpy as np
import cv2
import requests
from PIL import Image
import io

# Set up page title and layout
st.set_page_config(page_title="Pneumonia Detection AI", layout="wide")

# Title and description
st.title("🫁 Pneumonia Detection AI")
st.write("Upload a *CT scan* to detect early-stage pneumonia and visualize results with AI explanations.")

# File upload section
uploaded_file = st.file_uploader("Upload a CT scan (PNG, JPG, or DICOM)", type=["png", "jpg", "jpeg", "dcm"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded CT Scan", use_column_width=True)

    # Convert image for API request
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    # Send the image to the AI model (Backend API)
    with st.spinner("Processing... Please wait."):
        api_url = "http://backend-api-url/predict"  # Replace with actual backend API URL
        response = requests.post(api_url, files={"file": image_bytes})

        if response.status_code == 200:
            result = response.json()
            prediction = result.get("prediction", "Unknown")
            confidence = result.get("confidence", "N/A")
            heatmap_url = result.get("heatmap_url", None)

            # Display results
            st.subheader("🩺 AI Diagnosis")
            st.write(f"*Prediction:* {prediction}")
            st.write(f"*Confidence:* {confidence:.2f}%")

            # Display heatmap if available
            if heatmap_url:
                st.subheader("📍 AI Heatmap")
                st.image(heatmap_url, caption="Highlighted pneumonia-affected areas")

            # Chatbot integration
            st.subheader("💬 AI Chatbot (ClinicalT5)")
            query = st.text_input("Ask AI a medical question (e.g., 'What does this result mean?'):")
            if st.button("Ask AI"):
                chatbot_url = "http://backend-api-url/chatbot"
                chatbot_response = requests.post(chatbot_url, json={"query": query})
                if chatbot_response.status_code == 200:
                    answer = chatbot_response.json().get("response", "No response")
                    st.write(f"*AI Response:* {answer}")

        else:
            st.error("⚠ Error processing the image. Please try again.")

# Footer
st.markdown("---")
st.write("🔬 *Developed for Early-Stage Pneumonia Detection Hackathon*")
