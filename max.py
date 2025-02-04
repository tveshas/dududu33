'''import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------------
# 1. MODEL + TRAINING CODE (SAME)
# -------------------------------

data = {
    'Qualifying_Position': [1, 1, 2, 1, 0, 2, 0, 1, 3, 1, 0, 1, 2, 1, 4, 1, 0, 2, 1, 1],
    'FP1_Gap': [0.15, 0.30, 0.12, 0.40, 5.0, 0.25, 4.8, 0.18, 0.35, 0.22, 4.9, 0.15, 0.28, 0.20, 0.45, 0.12, 5.2, 0.32, 0.17, 0.19],
    'FP2_Gap': [0.10, 0.25, 0.08, 0.35, 5.5, 0.20, 5.1, 0.15, 0.30, 0.18, 5.3, 0.12, 0.25, 0.15, 0.40, 0.10, 5.6, 0.28, 0.14, 0.16],
    'FP3_Gap': [0.05, 0.20, 0.05, 0.30, 6.0, 0.15, 5.5, 0.12, 0.25, 0.15, 5.8, 0.10, 0.22, 0.12, 0.35, 0.08, 6.2, 0.25, 0.11, 0.13],
    'Pit_Stops': [1, 2, 1, 3, 4, 1, 5, 2, 3, 1, 4, 1, 2, 1, 3, 1, 5, 2, 1, 2],
    'Finish_Position': [1, 1, 1, 0, 0, 2, 0, 1, 3, 1, 0, 1, 2, 1, 4, 1, 0, 2, 1, 1]
}
df = pd.DataFrame(data)

X = torch.tensor(df.drop('Finish_Position', axis=1).values, dtype=torch.float32)
y = torch.tensor(df['Finish_Position'].values, dtype=torch.long)

class RaceClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = RaceClassifier(input_size=X.shape[1], num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 300
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -------------------------------
# 2. STREAMLIT APP WITH ENHANCED STYLING
# -------------------------------

st.set_page_config(
    page_title="Max Aura Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS styling with Spotify iframe support
page_bg_img = f"""
<style>
/* Previous CSS remains the same */

/* Spotify embed container */
.spotify-embed {{
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
    background: rgba(0, 0, 0, 0.7);
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}}

/* Max quote styling */
.max-quote {{
    color: #FFFFFF;
    font-style: italic;
    text-align: center;
    padding: 15px;
    margin: 20px 0;
    background: rgba(30, 144, 255, 0.2);
    border-left: 4px solid #1E90FF;
    border-radius: 4px;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Banner
st.markdown("""
    <div style='text-align:center; margin:20px 0;'>
        <img src='https://i.pinimg.com/736x/7e/61/72/7e617230ae15f4cb117aef7595f79d0b.jpg' 
        alt='Banner' style='max-width:100%;height:auto;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,0.2);'>
    </div>
""", unsafe_allow_html=True)

# Title and Subtitle with Max context
st.title("Max Aura Predictor")
st.markdown("""
    <div class='max-quote'>
        "We chose Max for our linear regression model because he's more consistent than AWS uptime."
        <br><small>- Data Science Team</small>
    </div>
""", unsafe_allow_html=True)

# Spotify embed
spotify_embed = """
<div class='spotify-embed'>
    <iframe src="https://open.spotify.com/embed/track/4nKRZAONxGgcKCMin730Ai" 
            width="300" 
            height="80" 
            frameborder="0" 
            allowtransparency="true" 
            allow="encrypted-media">
    </iframe>
</div>
"""
st.markdown(spotify_embed, unsafe_allow_html=True)

# Fun facts about Max's predictability
with st.expander("📊 Why Max Verstappen?"):
    st.markdown("""
        * More predictable than your morning coffee routine
        * Fastest laps so consistent, they're boring the statisticians
        * DNFs are rarer than a Mercedes 1-2 in 2023
        * Gap to P2 is usually bigger than the F1 TV subscription fee
        * He's just simply, simply lovely!
    """)

# Input columns
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    q_pos = st.number_input("Qualifying Position", min_value=0, max_value=20, value=1)
with col2:
    fp1_gap = st.number_input("FP1 Gap (s)", value=0.25, format="%.3f")
with col3:
    fp2_gap = st.number_input("FP2 Gap (s)", value=0.20, format="%.3f")
with col4:
    fp3_gap = st.number_input("FP3 Gap (s)", value=0.15, format="%.3f")
with col5:
    pit_stops = st.number_input("Pit Stops", min_value=0, max_value=10, value=2)

predict_button = st.button("Predict Race Outcome", key="predict")

if predict_button:
    inputs = torch.tensor([[q_pos, fp1_gap, fp2_gap, fp3_gap, pit_stops]], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        prediction = predicted.item()

    if prediction == 0:
        st.error("🚨 Predicted Result: DNF or DNS (As rare as Max making a mistake) 🚨")
    else:
        if prediction == 1:
            st.success(f"🏆 Predicted Finish Position: P1 - That's how we do it!")
        else:
            st.success(f"🏆 Predicted Finish Position: P{prediction}")

if st.checkbox("Show Training Loss Curve (as flat as Max's competition)"):
    dummy_loss = [5/(i+1) for i in range(epochs)]
    fig, ax = plt.subplots(facecolor='#0E1117')
    ax.plot(dummy_loss, color='#1E90FF')
    ax.set_title("Training Loss Curve", color='white')
    ax.set_xlabel("Epochs", color='white')
    ax.set_ylabel("Loss", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2)
    st.pyplot(fig) '''
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Model and training code remains the same
data = {
    'Qualifying_Position': [1, 1, 2, 1, 0, 2, 0, 1, 3, 1, 0, 1, 2, 1, 4, 1, 0, 2, 1, 1],
    'FP1_Gap': [0.15, 0.30, 0.12, 0.40, 5.0, 0.25, 4.8, 0.18, 0.35, 0.22, 4.9, 0.15, 0.28, 0.20, 0.45, 0.12, 5.2, 0.32, 0.17, 0.19],
    'FP2_Gap': [0.10, 0.25, 0.08, 0.35, 5.5, 0.20, 5.1, 0.15, 0.30, 0.18, 5.3, 0.12, 0.25, 0.15, 0.40, 0.10, 5.6, 0.28, 0.14, 0.16],
    'FP3_Gap': [0.05, 0.20, 0.05, 0.30, 6.0, 0.15, 5.5, 0.12, 0.25, 0.15, 5.8, 0.10, 0.22, 0.12, 0.35, 0.08, 6.2, 0.25, 0.11, 0.13],
    'Pit_Stops': [1, 2, 1, 3, 4, 1, 5, 2, 3, 1, 4, 1, 2, 1, 3, 1, 5, 2, 1, 2],
    'Finish_Position': [1, 1, 1, 0, 0, 2, 0, 1, 3, 1, 0, 1, 2, 1, 4, 1, 0, 2, 1, 1]
}
df = pd.DataFrame(data)

X = torch.tensor(df.drop('Finish_Position', axis=1).values, dtype=torch.float32)
y = torch.tensor(df['Finish_Position'].values, dtype=torch.long)

class RaceClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = RaceClassifier(input_size=X.shape[1], num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 300
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Enhanced Streamlit app with animations
st.set_page_config(
    page_title="Max Aura Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS with animations
page_bg_img = """
<style>
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes elegantReveal {
    0% {
        opacity: 0;
        transform: translateY(30px) scale(0.95);
        letter-spacing: 5px;
    }
    50% {
        opacity: 0.5;
        transform: translateY(15px) scale(0.97);
        letter-spacing: 2px;
    }
    100% {
        opacity: 1;
        transform: translateY(0) scale(1);
        letter-spacing: normal;
    }
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% {
        transform: translateY(0);
    }
    40% {
        transform: translateY(-20px);
    }
    60% {
        transform: translateY(-10px);
    }
}

.banner-img {
    animation: fadeIn 1s ease-out;
}

.title {
    animation: elegantReveal 1.8s cubic-bezier(0.4, 0, 0.2, 1);
    text-transform: uppercase;
    font-weight: 300;
    letter-spacing: normal;
}

.max-quote {
    color: #FFFFFF;
    font-style: italic;
    text-align: center;
    padding: 15px;
    margin: 20px 0;
    background: rgba(30, 144, 255, 0.2);
    border-left: 4px solid #1E90FF;
    border-radius: 4px;
    animation: fadeIn 1.5s ease-out;
}

.input-section {
    animation: fadeIn 2s ease-out;
}

.spotify-embed {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
    background: rgba(0, 0, 0, 0.7);
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    animation: fadeIn 2.5s ease-out;
}

.prediction {
    animation: bounce 1s ease-out;
}

.stButton button {
    animation: fadeIn 2s ease-out;
}

.css-1offfwp {
    animation: fadeIn 2s ease-out;
}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Banner with animation class
st.markdown("""
    <div class='banner-img' style='text-align:center; margin:20px 0;'>
        <img src='https://i.pinimg.com/736x/7e/61/72/7e617230ae15f4cb117aef7595f79d0b.jpg' 
        alt='Banner' style='max-width:100%;height:auto;border-radius:8px;box-shadow:0 4px 6px rgba(0,0,0,0.2);'>
    </div>
""", unsafe_allow_html=True)

# Title with animation class
st.markdown("<h1 class='title'>Max Aura Predictor</h1>", unsafe_allow_html=True)

# Max quote with animation
st.markdown("""
    <div class='max-quote'>
        "We chose Max for our linear regression model because he's more consistent than AWS uptime."
        <br><small>- Data Science Team</small>
    </div>
""", unsafe_allow_html=True)

# Spotify embed with animation
spotify_embed = """
<div class='spotify-embed'>
    <iframe src="https://open.spotify.com/embed/track/4nKRZAONxGgcKCMin730Ai" 
            width="300" 
            height="80" 
            frameborder="0" 
            allowtransparency="true" 
            allow="encrypted-media">
    </iframe>
</div>
"""
st.markdown(spotify_embed, unsafe_allow_html=True)

# Fun facts with animation
with st.expander("📊 Why Max Verstappen?"):
    st.markdown("""
        * More predictable than your morning coffee routine
        * Fastest laps so consistent, they're boring the statisticians
        * DNFs are rarer than a Mercedes 1-2 in 2023
        * Gap to P2 is usually bigger than the F1 TV subscription fee
        * He's just simply, simply lovely!
    """)

# Input section with animation class
st.markdown("<div class='input-section'>", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    q_pos = st.number_input("Qualifying Position", min_value=0, max_value=20, value=1)
with col2:
    fp1_gap = st.number_input("FP1 Gap (s)", value=0.25, format="%.3f")
with col3:
    fp2_gap = st.number_input("FP2 Gap (s)", value=0.20, format="%.3f")
with col4:
    fp3_gap = st.number_input("FP3 Gap (s)", value=0.15, format="%.3f")
with col5:
    pit_stops = st.number_input("Pit Stops", min_value=0, max_value=10, value=2)
st.markdown("</div>", unsafe_allow_html=True)

predict_button = st.button("Predict Race Outcome", key="predict")

if predict_button:
    inputs = torch.tensor([[q_pos, fp1_gap, fp2_gap, fp3_gap, pit_stops]], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        prediction = predicted.item()

    st.markdown("<div class='prediction'>", unsafe_allow_html=True)
    if prediction == 0:
        st.error("🚨 Predicted Result: DNF or DNS (As rare as Max making a mistake) 🚨")
    else:
        if prediction == 1:
            st.success(f"🏆 Predicted Finish Position: P1 - That's how we do it!")
        else:
            st.success(f"🏆 Predicted Finish Position: P{prediction}")
    st.markdown("</div>", unsafe_allow_html=True)

if st.checkbox("Show Training Loss Curve (as flat as Max's competition)"):
    dummy_loss = [5/(i+1) for i in range(epochs)]
    fig, ax = plt.subplots(facecolor='#0E1117')
    ax.plot(dummy_loss, color='#1E90FF')
    ax.set_title("Training Loss Curve", color='white')
    ax.set_xlabel("Epochs", color='white')
    ax.set_ylabel("Loss", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)