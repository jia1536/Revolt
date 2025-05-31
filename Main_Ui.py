import streamlit as st
import os
import json
import base64
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib
from PIL import Image
from crop_labels import crop_labels

# Page configuration
st.set_page_config(
    page_title="AgriRevolt",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Base styles */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.94);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-top: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: rgba(246, 249, 242, 0.97);
        padding: 1.5rem 1rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2E7D32;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #388E3C;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    /* Sliders */
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Cards for results */
    .result-card {
        background-color: #E8F5E9;
        color: #4E342E;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #4CAF50;
    }
    
    .disease-card {
        background-color: #E8F5E9;
        color: #4E342E;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #FF5722;
    }
    
    .remedy-card {
        background-color: #F1F8E9;
        color: #4E342E;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #8BC34A;
    }
    
    /* Progress indicators */
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #2196F3;
        color: #0D47A1;
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 3rem;
        color: #424242;
        font-size: 0.9rem;
    }
    
    /* Markdown text colors - improved contrast */
    .markdown-text-container p {
        color: #008000;
    }
    
    .markdown-text-container ul li {
        color: #008000;
    }
    
    /* Card text color improvements */
    .disease-card p, .remedy-card p, .result-card p {
        color: #008000;
    }
    
    /* Info section colors */
    .info-section {
        background-color: #FAFAFA;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        color: #008000;
        border: 1px solid #E0E0E0;
    }
    
    /* Utility classes */
    .text-center { text-align: center; }
    .mb-0 { margin-bottom: 0; }
    .mt-1 { margin-top: 1rem; }
    .mt-2 { margin-top: 2rem; }
    .mt-3 { margin-top: 3rem; }
    .animate-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: .7; }
    }
</style>
""", unsafe_allow_html=True)
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def display_header():
    base64_img = get_base64_image("logo.png")

    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; padding: 10px;">
            <img src="data:image/png;base64,{base64_img}" 
                 style="border-radius: 10%; width: 150px; height: 150px; 
                        object-fit: cover; border: 3px solid #4CAF50;">
        </div>
        """, unsafe_allow_html=True)


    with col2:
        st.markdown("""
            <div style='line-height: 1.2;'>
                <h1 style='margin-bottom: 0; font-size: 5rem; color: #2E7D32;'>AgriRevolt</h1>
                <p style='font-size: 1.2rem; color: #4CAF50; font-style: italic; margin-top: 0;'>Your Smart Farming Companion — Powered by AI</p>
            </div>
        """, unsafe_allow_html=True)
        # Call it in your app

# Sidebar elements with improved UX
with st.sidebar:
    st.markdown(""" <div><h2> 🌱 Revolutionizing Agriculture: Smart Farming using Machine and Deep Learning</h2></div>"""
                , unsafe_allow_html=True)
    st.markdown(""" <div> <h3>Welcome, Smart Farmer!</h3> </div>""", unsafe_allow_html=True)
    
    
    # Tool selection
    st.markdown("### Choose Your Tool:")
    model_type = st.radio(
        "",
        options=[
            "🩺 Diagnose Plant Disease",
            "🌾 Crop Recommendation"
        ],
        index=0
    )
    
    st.markdown("""
        <div class="info-box">
            <strong>💡 Pro Tip:</strong><br>
            For best disease diagnosis results, take clear photos in natural light with the entire leaf visible.
        </div>
    """, unsafe_allow_html=True)
    
# Model paths
disease_model_path = 'plant_disease.keras'
crop_model_path = 'model.pkl'
disease_model_exists = os.path.exists(disease_model_path)
crop_model_exists = os.path.exists(crop_model_path)

# Remedies
remedies_dict = {}
try:
    with open('remedies.json', 'r') as f:
        remedies_dict = json.load(f)
except FileNotFoundError:
    st.warning("⚠️ remedies.json not found. Remedies may be unavailable.")

# Class labels
class_names = [
    "Apple__Apple_scab", "Apple__Black_rot", "Apple__Cedar_apple_rust", "Apple__healthy",
    "Blueberry__healthy", "Cherry_(including_sour)__Powdery_mildew", "Cherry_(including_sour)__healthy",
    "Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)__Common_rust_",
    "Corn_(maize)__Northern_Leaf_Blight", "Corn_(maize)__healthy", "Grape__Black_rot",
    "Grape__Esca_(Black_Measles)", "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape__healthy",
    "Orange__Haunglongbing_(Citrus_greening)", "Peach__Bacterial_spot", "Peach__healthy",
    "Pepper,_bell__Bacterial_spot", "Pepper,_bell__healthy", "Potato__Early_blight",
    "Potato__Late_blight", "Potato__healthy", "Raspberry__healthy", "Soybean__healthy",
    "Squash__Powdery_mildew", "Strawberry__Leaf_scorch", "Strawberry__healthy",
    "Tomato__Bacterial_spot", "Tomato__Early_blight", "Tomato__Late_blight",
    "Tomato__Leaf_Mold", "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites Two-spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato__healthy"
]

# Format disease name for better readability
def format_disease_name(disease_name):
    parts = disease_name.split('__')
    plant = parts[0].replace('_', ' ')
    
    if len(parts) > 1:
        condition = parts[1].replace('_', ' ')
        if condition.lower() == 'healthy':
            return f"{plant} (Healthy)"
        else:
            return f"{plant}: {condition}"
    return disease_name

# Main content area
display_header()

# Disease Diagnosis
if model_type.startswith("🩺"):
    st.markdown("## 🔬 Plant Disease Diagnosis")
    
    st.markdown("""
        <div class="info-box">
            <strong>How it works:</strong> Upload a clear image of a plant leaf showing symptoms. Our AI system will
            analyze the image and identify potential diseases, then provide treatment recommendations.
        </div>
    """, unsafe_allow_html=True)
    
    if not disease_model_exists:
        st.error("⚠️ Disease model not found! Please add 'plant_disease.keras'.")
    else:
        try:
            model = load_model(disease_model_path)
            
            # Single column image upload layout
            st.markdown("### Upload Leaf Image")
            uploaded_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png"],
                help="Take a clear, well-lit photo of the affected leaf"
            )
            
            st.markdown("""
                <div class="info-section">
                    <strong>Supported plants:</strong> Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, 
                    Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
                </div>
            """, unsafe_allow_html=True)
            
            if uploaded_file:
                img = Image.open(uploaded_file).convert("RGB")
                st.markdown("### Image Preview")
                st.image(img, use_container_width =True, caption="Uploaded Leaf Image")
                
                analyze_button = st.button("🔍 Analyze Image", use_container_width=True)
                
                if analyze_button:
                    with st.spinner("🧬 Analyzing leaf pattern..."):
                        # Progress bar for better UX during processing
                        progress_bar = st.progress(0)
                        for i in range(100):
                            # Simulate processing time
                            import time
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Actual prediction
                        img = img.resize((224, 224))
                        img_array = image.img_to_array(img)
                        img_array = np.expand_dims(img_array, axis=0) / 255.0
                        
                        prediction = model.predict(img_array)
                        predicted_class = class_names[np.argmax(prediction)]
                        
                        # Clear progress bar after completion
                        progress_bar.empty()
                        
                        # Display results in a modern card
                        st.markdown("""
                            <div class="disease-card">
                                <h3 style="margin-top: 0;">Diagnosis Results</h3>
                                <p><strong>Identified Condition:</strong> {}</p>
                            </div>
                        """.format(format_disease_name(predicted_class)), unsafe_allow_html=True)
                        
                        # Show remedies if available
                        remedy = remedies_dict.get(predicted_class, "No specific remedy available for this condition.")
                        
                        st.markdown("""
                            <div class="remedy-card">
                                <h3 style="margin-top: 0;">🌱 Treatment Recommendations</h3>
                                <p>{}</p>
                            </div>
                        """.format(remedy), unsafe_allow_html=True)
                        
                        # Related resources
                        st.markdown("### 📚 Related Resources")
                        resource_col1, resource_col2 = st.columns(2)
                        
                        with resource_col1:
                            st.markdown("""
                                <div style="padding: 15px; border-radius: 10px; border: 1px solid #E0E0E0;">
                                    <h4 style="margin-top: 0; color: #2E7D32;">📋 Prevention Tips</h4>
                                    <ul>
                                        <li>Practice crop rotation</li>
                                        <li>Maintain proper plant spacing</li>
                                        <li>Water at the base, not on leaves</li>
                                        <li>Remove infected plant debris</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        with resource_col2:
                            st.markdown("""
                                <div style="padding: 15px; border-radius: 10px; border: 1px solid #E0E0E0;">
                                    <h4 style="margin-top: 0; color: #2E7D32;">🔎 Similar Diseases</h4>
                                    <p>Other conditions with similar symptoms that might require different treatments:</p>
                                    <ul>
                                        <li>Nutrient deficiencies</li>
                                        <li>Environmental stress</li>
                                        <li>Related fungal/bacterial infections</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background-color: #F5F5F5; border-radius: 10px; padding: 40px 20px; text-align: center;">
                        <div style="font-size: 3rem; margin-bottom: 20px;">📸</div>
                        <p style="color: #212121;">Upload a photo of your plant to get started</p>
                    </div>
                """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error: {e}")

# Crop Recommendation
elif model_type.startswith("🌾"):
    st.markdown("## 🌱 Smart Crop Recommendation")
    
    st.markdown("""
        <div class="info-box">
            <strong>How it works:</strong> Enter your soil parameters and local climate conditions. 
            Our AI system will analyze this data to recommend the most suitable crops for your land.
        </div>
    """, unsafe_allow_html=True)
    
    if not crop_model_exists:
        st.error("⚠️ Crop recommendation model not found! Please add 'model.pkl'.")
    else:
        try:
            crop_model = joblib.load(crop_model_path)
            
            st.markdown("### Enter Your Farm Parameters")
            
            # Tabs for different parameter categories
            soil_tab, climate_tab = st.tabs(["📊 Soil Parameters", "🌤️ Climate Data"])
            
            with soil_tab:
                col1, col2 = st.columns(2)
                with col1:
                    N = st.slider("Nitrogen (N)", 0, 150, 50, 
                                 help="Amount of Nitrogen in soil (kg/ha)")
                    P = st.slider("Phosphorus (P)", 0, 150, 50, 
                                 help="Amount of Phosphorus in soil (kg/ha)")
                with col2:
                    K = st.slider("Potassium (K)", 0, 200, 50, 
                                 help="Amount of Potassium in soil (kg/ha)")
                    ph = st.slider("pH", 0.0, 14.0, 6.5, 0.1, 
                                  help="Soil pH level (0-14)")
                
                # Soil health indicator
                soil_health = (N/150 + P/150 + (14-abs(7-ph))/7)/3 * 100
                st.markdown(f"""
                    <div style="margin-top: 20px;">
                        <p><strong>Soil Health Indicator:</strong></p>
                        <div style="background-color: #e0e0e0; border-radius: 10px; height: 10px;">
                            <div style="width: {soil_health}%; background-color: {'#4CAF50' if soil_health > 60 else '#FFC107' if soil_health > 40 else '#F44336'}; 
                                        height: 10px; border-radius: 10px;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #424242;">
                            <span>Poor</span>
                            <span>Moderate</span>
                            <span>Excellent</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            with climate_tab:
                col1, col2 = st.columns(2)
                with col1:
                    temp = st.slider("Average Temperature (°C)", 0.0, 50.0, 25.0, 0.5, 
                                    help="Average temperature throughout growing season")
                    humidity = st.slider("Humidity (%)", 0, 100, 65, 
                                        help="Average relative humidity percentage")
                with col2:
                    rainfall = st.slider("Annual Rainfall (mm)", 0.0, 3000.0, 1000.0, 50.0, 
                                        help="Expected annual rainfall in millimeters")
            
            # More visually appealing button
            predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
            with predict_col2:
                predict_button = st.button("🔮 Recommend Best Crops", use_container_width=True)
            
            if predict_button:
                with st.spinner("Analyzing farm parameters..."):
                    # Progress bar for better UX
                    progress_bar = st.progress(0)
                    for i in range(100):
                        # Simulate processing time
                        import time
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    input_data = pd.DataFrame({
                        'N': [N], 'P': [P], 'K': [K],
                        'temperature': [temp],
                        'humidity': [humidity],
                        'ph': [ph],
                        'rainfall': [rainfall]
                    })
                    
                    # Clear progress bar after completion
                    progress_bar.empty()
                    
                    result = crop_model.predict(input_data)
                    crop_index = int(result[0])
                    crop_name = crop_labels[crop_index]
                    
                    # Display recommendation with images and info
                    st.markdown("""
                        <div class="result-card">
                            <h3 style="margin-top: 0;">🌱 Crop Recommendation</h3>
                            <div style="display: flex; align-items: center;">
                                <div style="font-size: 3rem; margin-right: 20px;">🌿</div>
                                <div>
                                    <p style="font-size: 1.5rem; font-weight: 600; margin-bottom: 5px; color: #2E7D32;">
                                        {crop_name}
                                    </p>
                                    <p>Best match for your soil and climate conditions</p>
                                </div>
                            </div>
                        </div>
                    """.replace("{crop_name}", crop_name.capitalize()), unsafe_allow_html=True)
                    
                    # Additional crop information
                    st.markdown("### Crop Information")
                    info_col1, info_col2 = st.columns(2)
                    
                    with info_col1:
                        st.markdown("""
                            <div style="padding: 15px; border-radius: 10px; border: 1px solid #E0E0E0;">
                                <h4 style="margin-top: 0; color: #2E7D32;">Growing Requirements</h4>
                                <ul>
                                    <li><strong>Growing Season:</strong> Depends on variety</li>
                                    <li><strong>Watering Needs:</strong> Medium</li>
                                    <li><strong>Sunlight:</strong> Full sun</li>
                                    <li><strong>Spacing:</strong> Crop-specific</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with info_col2:
                        st.markdown("""
                            <div style="padding: 15px; border-radius: 10px; border: 1px solid #E0E0E0;">
                                <h4 style="margin-top: 0; color: #2E7D32;">Market Potential</h4>
                                <ul>
                                    <li><strong>Market Demand:</strong> Medium-High</li>
                                    <li><strong>Price Stability:</strong> Moderate</li>
                                    <li><strong>Processing Options:</strong> Multiple</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Alternative crops
                    st.markdown("### Alternative Options")
                    alt_col1, alt_col2, alt_col3 = st.columns(3)
                    
                    with alt_col1:
                        st.markdown("""
                            <div style="text-align: center; padding: 15px; border-radius: 10px; border: 1px solid #E0E0E0;">
                                <div style="font-size: 2rem;">🌽</div>
                                <p style="font-weight: 600; color: #2E7D32;">Corn</p>
                                <p>Compatibility: 80%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with alt_col2:
                        st.markdown("""
                            <div style="text-align: center; padding: 15px; border-radius: 10px; border: 1px solid #E0E0E0;">
                                <div style="font-size: 2rem;">🍅</div>
                                <p style="font-weight: 600; color: #2E7D32;">Tomato</p>
                                <p>Compatibility: 75%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with alt_col3:
                        st.markdown("""
                            <div style="text-align: center; padding: 15px; border-radius: 10px; border: 1px solid #E0E0E0;">
                                <div style="font-size: 2rem;">🥔</div>
                                <p style="font-weight: 600; color: #2E7D32;">Potato</p>
                                <p>Compatibility: 70%</p>
                            </div>
                        """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error loading crop model: {e}")

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 Plant Doctor Assistant | Your Smart Farming Companion</p>
</div>
""", unsafe_allow_html=True)