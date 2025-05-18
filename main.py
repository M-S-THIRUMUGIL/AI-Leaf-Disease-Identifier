import streamlit as st
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
import time

# --- Page Config ---
st.set_page_config(page_title="AI Leaf Disease Identifier", layout="centered")

# --- Title ---
st.markdown("<h1 style='text-align: center; color: white;'>AI Leaf Disease Identifier</h1>", unsafe_allow_html=True)

# --- Navigation Bar ---
navigation = st.sidebar.radio("Navigation", ["Home", "Upload & Detect", "About"])

# --- Background Image with Mobile Responsiveness ---
background_image_url = "https://www.pixelstalk.net/wp-content/uploads/images1/Free-HD-Leaf-Photos.jpg"
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url({background_image_url});
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stAppViewContainer"]::before {{
content: "";
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
background-color: rgba(0, 0, 0, 0.7);
z-index: 0;
}}

@media only screen and (max-width: 768px) {{
[data-testid="stAppViewContainer"] {{
    background-size: contain;
}}
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Cached Model Loaders ---
@st.cache_resource
def load_disease_model():
    with st.spinner("🔄 Loading disease detection model..."):
        return load_model('Training/model/Leaf Deases(96,88).h5')

@st.cache_resource
def load_leaf_filter_model():
    return mobilenet_v2.MobileNetV2(weights="imagenet")

model = load_disease_model()
leaf_filter_model = load_leaf_filter_model()

# --- Label Names ---
label_name = ['Apple scab','Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Cherry Powdery mildew',
'Cherry healthy','Corn Cercospora leaf spot Gray leaf spot', 'Corn Common rust', 'Corn Northern Leaf Blight','Corn healthy', 
'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy','Peach Bacterial spot','Peach healthy', 'Pepper bell Bacterial spot', 
'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy']

# --- Sample Solutions Dictionary ---
disease_solutions = {label: "Temporarily unavailable" for label in label_name}
disease_side_effects = {label: "Temporarily unavailable" for label in label_name}

# --- Leaf Image Check ---
def is_leaf_image(img_rgb):
    img_resized = cv.resize(img_rgb, (224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)
    preds = leaf_filter_model.predict(img_preprocessed)
    decoded_preds = decode_predictions(preds, top=5)[0]
    keywords = ["leaf", "plant", "foliage", "tree", "flower", "botanical"]
    return any(any(word in label.lower() for word in keywords) for (_, label, _) in decoded_preds)

# --- Home Page ---
if navigation == "Home":
    st.write("""
    Welcome to the **AI Leaf Disease Identifier** 🌿

    This AI-powered web application helps farmers, researchers, and agriculture enthusiasts detect diseases in crop leaves.
    
    ### 📸 What you can do:
    - Upload a clear leaf image.
    - Get an instant diagnosis of the disease (if any).
    - View suggestions for solutions and possible side effects (coming soon).

    ### 🌱 Leaf types supported:
    - 🍎 Apple
    - 🍒 Cherry
    - 🌽 Corn
    - 🍇 Grape
    - 🍑 Peach
    - 🌶️ Pepper
    - 🥔 Potato
    - 🍓 Strawberry
    - 🍅 Tomato

    💡 **Upcoming Feature:** Auto-suggested organic and chemical treatments, along with multilingual support!
    """)

# --- Upload and Detection Page ---
elif navigation == "Upload & Detect":
    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

        if st.button("Clear Image"):
            st.experimental_rerun()

        with st.spinner("Analyzing image..."):
            time.sleep(1.5)
            if not is_leaf_image(img_rgb):
                st.error("🚫 This doesn't appear to be a leaf or plant image. Please upload a clear leaf photo.")
            else:
                st.success("🌿 Leaf detected. Running disease detection...")
                resized_img = cv.resize(img_rgb, (150, 150))
                normalized_image = np.expand_dims(resized_img / 255.0, axis=0)
                predictions = model.predict(normalized_image)

                confidence = predictions[0][np.argmax(predictions)] * 100
                predicted_label = label_name[np.argmax(predictions)]

                if confidence >= 80:
                    st.subheader(f"✅ Prediction: {predicted_label}")
                    st.info(f"🛠️ Solution: {disease_solutions[predicted_label]}")
                    st.warning(f"⚠️ Side Effects: {disease_side_effects[predicted_label]}")

                    # Show top 3 predictions
                    st.write("🔍 Top Predictions:")
                    top_3_indices = predictions[0].argsort()[-3:][::-1]
                    for i in top_3_indices:
                        st.write(f"{label_name[i]}: {predictions[0][i]*100:.2f}%")
                else:
                    st.warning("⚠️ Low confidence. Try a clearer image.")

# --- About Page ---
elif navigation == "About":
    st.write("""
    **About this App**

    This AI-powered tool helps farmers and researchers detect diseases in crop leaves.
    It uses deep learning and transfer learning models to identify diseases with high confidence.

    More features coming soon: 🌱
    - Real-time solution database
    - Downloadable reports
    - Crop-specific insights
    - Regional disease analysis
    - Multilingual support
    """)
