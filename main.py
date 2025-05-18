import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import io
import csv
import os

# Set the page configuration
st.set_page_config(page_title="Bird Sound Classifier", page_icon="🐦", layout="wide")

# Load your pre-trained model (replace with the correct path)
model = load_model("bird_song_classifier_1.h5")

# Define a mapping for category IDs to bird names
CATEGORY_MAPPING = {
    0: 'Andean Guan_sound',
    1: 'Andean Tinamou_sound',
    2: 'Australian Brushturkey_sound',
    3: 'Band-tailed Guan_sound',
    4: 'Barred Tinamou_sound',
    5: 'Bartletts Tinamou_sound',
    6: 'Baudo Guan_sound',
    7: 'Bearded Guan_sound',
    8: 'Berlepschs Tinamou_sound',
    9: 'Biak Scrubfowl_sound',
    10: 'Black Tinamou_sound',
    11: 'Black-billed Brushturkey_sound',
    12: 'Black-capped Tinamou_sound',
    13: 'Black-fronted Piping Guan_sound',
    14: 'Blue-throated Piping Guan_sound',
    15: 'Brazilian Tinamou_sound',
    16: 'Brown Tinamou_sound',
    17: 'Brushland Tinamou_sound',
    18: 'Buff-browed Chachalaca_sound',
    19: 'Cauca Guan_sound',
    20: 'Chaco Chachalaca_sound',
    21: 'Chestnut-bellied Guan_sound',
    22: 'Chestnut-headed Chachalaca_sound',
    23: 'Chestnut-winged Chachalaca_sound',
    24: 'Chilean Tinamou_sound',
    25: 'Choco Tinamou_sound',
    26: 'Cinereous Tinamou_sound',
    27: 'Collared Brushturkey_sound',
    28: 'Colombian Chachalaca_sound',
    29: 'Common Ostrich_sound',
    30: 'Crested Guan_sound',
    31: 'Curve-billed Tinamou_sound',
    32: 'Darwins Nothura_sound',
    33: 'Dusky Megapode_sound',
    34: 'Dusky-legged Guan_sound',
    35: 'Dwarf Cassowary_sound',
    36: 'Dwarf Tinamou_sound',
    37: 'East Brazilian Chachalaca_sound',
    38: 'Elegant Crested Tinamou_sound',
    39: 'Emu_sound',
    40: 'Great Spotted Kiwi_sound',
    41: 'Great Tinamou_sound',
    42: 'Greater Rhea_sound',
    43: 'Grey Tinamou_sound',
    44: 'Grey-headed Chachalaca_sound',
    45: 'Grey-legged Tinamou_sound',
    46: 'Highland Tinamou_sound',
    47: 'Hooded Tinamou_sound',
    48: 'Huayco Tinamou_sound',
    49: 'Lesser Nothura_sound',
    50: 'Lesser Rhea_sound',
    51: 'Little Chachalaca_sound',
    52: 'Little Spotted Kiwi_sound',
    53: 'Little Tinamou_sound',
    54: 'Maleo_sound',
    55: 'Malleefowl_sound',
    56: 'Marail Guan_sound',
    57: 'Melanesian Megapode_sound',
    58: 'Micronesian Megapode_sound',
    59: 'Moluccan Megapode_sound',
    60: 'New Guinea Scrubfowl_sound',
    61: 'Nicobar Megapode_sound',
    62: 'North Island Brown Kiwi_sound',
    63: 'Northern Cassowary_sound',
    64: 'Okarito Kiwi_sound',
    65: 'Orange-footed Scrubfowl_sound',
    66: 'Ornate Tinamou_sound',
    67: 'Pale-browed Tinamou_sound',
    68: 'Patagonian Tinamou_sound',
    69: 'Philippine Megapode_sound',
    70: 'Plain Chachalaca_sound',
    71: 'Puna Tinamou_sound',
    72: 'Quebracho Crested Tinamou_sound',
    73: 'Red-billed Brushturkey_sound',
    74: 'Red-faced Guan_sound',
    75: 'Red-legged Tinamou_sound',
    76: 'Red-throated Piping Guan_sound',
    77: 'Red-winged Tinamou_sound',
    78: 'Rufous-bellied Chachalaca_sound',
    79: 'Rufous-headed Chachalaca_sound',
    80: 'Rufous-vented Chachalaca_sound',
    81: 'Rusty Tinamou_sound',
    82: 'Rusty-margined Guan_sound',
    83: 'Scaled Chachalaca_sound',
    84: 'Slaty-breasted Tinamou_sound',
    85: 'Small-billed Tinamou_sound',
    86: 'Solitary Tinamou_sound',
    87: 'Somali Ostrich_sound',
    88: 'Southern Brown Kiwi_sound',
    89: 'Southern Cassowary_sound',
    90: 'Speckled Chachalaca_sound',
    91: 'Spixs Guan_sound',
    92: 'Spotted Nothura_sound',
    93: 'Sula Megapode_sound',
    94: 'Taczanowskis Tinamou_sound',
    95: 'Tanimbar Megapode_sound',
    96: 'Tataupa Tinamou_sound',
    97: 'Thicket Tinamou_sound',
    98: 'Tongan Megapode_sound',
    99: 'Trinidad Piping Guan_sound',
    100: 'Undulated Tinamou_sound',
    101: 'Vanuatu Megapode_sound',
    102: 'Variegated Tinamou_sound',
    103: 'Wattled Brushturkey_sound',
    104: 'West Mexican Chachalaca_sound',
    105: 'White-bellied Chachalaca_sound',
    106: 'White-bellied Nothura_sound',
    107: 'White-browed Guan_sound',
    108: 'White-crested Guan_sound',
    109: 'White-throated Tinamou_sound',
    110: 'White-winged Guan_sound',
    111: 'Yellow-legged Tinamou_sound'
}

# Function to extract features from the audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Function to load bird information from description.txt file
@st.cache_data
def load_bird_info(file_path):
    bird_info = {}
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                category = int(row["Number"])
                bird_info[category] = {
                    "name": row["Common Name"],
                    "scientific_name": row["Scientific Name"],
                    "description": row["Description"]
                }
            except ValueError as e:
                st.error(f"Error parsing row: {row}")
                st.error(f"Exception: {e}")
    return bird_info

# Load bird information from description.txt
BIRD_INFO = load_bird_info("description.txt")

# Streamlit application interface
st.markdown("<h1 style='text-align: center;'>Bird Sound Classification</h1>", unsafe_allow_html=True)
st.write("Upload an audio file (WAV or MP3) to classify the bird sound. The model will predict the bird species based on the sound!")

# File uploader with enhanced UI
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"], label_visibility="collapsed")

if uploaded_file is not None:
    try:
        # Display progress bar with dynamic updates
        progress_bar = st.progress(0)
        progress_bar.progress(20)
        
        # Convert the uploaded file to an in-memory file for librosa
        audio_file = io.BytesIO(uploaded_file.read())

        # Extract features from the uploaded audio file
        features = extract_features(audio_file)

        progress_bar.progress(60)

        # Reshape the features to match the input shape of the model
        features = features.reshape(1, -1)

        # Make prediction using the model
        prediction = model.predict(features)
        
        progress_bar.progress(100)

        # Get the predicted class
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Get bird information
        bird_info = BIRD_INFO.get(predicted_class, {"name": "Unknown", "scientific_name": "Unknown", "description": "No description available."})

        # Display the result in a styled card
        st.markdown(f"""
            <div style="background-color: #f0f8ff; border-radius: 10px; padding: 20px; margin-top: 20px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
                <h3 style="text-align: center;">Prediction Result</h3>
                <h4 style="text-align: center; color: #2c3e50;">{bird_info['name']}</h4>
                <p style="text-align: center; font-size: 1.1em; color: #34495e;">Scientific Name: <strong>{bird_info['scientific_name']}</strong></p>
                <p style="text-align: center; font-size: 1em; color: #7f8c8d;">{bird_info['description']}</p>
            </div>
        """, unsafe_allow_html=True)

        # Add the audio playback feature
        st.markdown("<h5>Listen to the uploaded sound:</h5>", unsafe_allow_html=True)
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

        
    except Exception as e:
        st.error(f"Error processing the file: {e}")

# Add a footer
st.markdown("""
    <footer style="text-align: center; padding: 10px 0; background-color: #2c3e50; color: white;">
        <p>Made with ❤️ by Bird Sound Classifier Team</p>
    </footer>
""", unsafe_allow_html=True)
