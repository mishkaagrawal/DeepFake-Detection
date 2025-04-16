import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from convnext import ConvNeXt  # Assuming this is your custom ConvNeXt class

# Load the model (caching the resource to prevent reloading every time)
@st.cache_resource
def ConvNeXt_model():
    model_conv = ConvNeXt(num_classes=2)
    state_dict = torch.load('convnext.pth', map_location=torch.device('cpu'))  # Ensure proper path
    model_conv.load_state_dict(state_dict)
    model_conv.eval()
    return model_conv

# Load the model
model = ConvNeXt_model()

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model-expected input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")  # Step 1
    img_tensor = transform(image).unsqueeze(0)         # Step 2 & 3
    
    img_tensor = img_tensor.to('cpu')                 # Match model device
    
    with torch.no_grad():
        outputs = model(img_tensor)                    # Step 4
        predicted = outputs.argmax(dim=1).item()

    return "Real ✅" if predicted == 1 else "Deep Fake ❌"

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption='Uploaded Image', use_column_width=True)
#     st.write("Classifying...")
#     label = predict_image(image)
#     st.write(f"Prediction: **{label}**")



# Apply custom CSS for styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
            background-image: url("https://images.unsplash.com/photo-1535223289827-42f1e9919769");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            color: white;
        }

        header, footer, .reportview-container .main footer {
            visibility: hidden;
            height: 0;
        }

        .main-title {
            font-size: 40px;
            text-align: center;
            font-weight: 600;
            margin-bottom: 20px;
            color: #ffffff;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
        }

        .box {
            background-color: rgba(0, 0, 0, 0.5);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }

        .result-box {
            background-color: rgba(255, 255, 255, 0.1);
            border: 2px solid #ffffff33;
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #fff;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🕵️ Deepfake Image Detector with ConvNeXt</div>', unsafe_allow_html=True)

# Upload box
st.markdown('<div class="box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📁 Upload an image for analysis:", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

# Show image and placeholder result
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Placeholder output — replace this part later with your ML model
    result = predict_image(uploaded_file)
    st.markdown(f'<div class="result-box">Result: {result}</div>', unsafe_allow_html=True)