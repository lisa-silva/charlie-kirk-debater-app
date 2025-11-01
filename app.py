import streamlit as st
import requests
import json
import base64
import time
import struct
import numpy as np

# --- Configuration & Initialization ---
# This uses the fixed, root-level secret key that resolved the AttributeError
try:
    API_KEY = st.secrets.gemini_api_key
except Exception:
    st.error("API key not found. Please set 'gemini_api_key' in Streamlit secrets.")
    st.stop()

API_URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
TEXT_MODEL = "gemini-2.5-flash-preview-09-2025"
TTS_MODEL = "gemini-2.5-flash-preview-tts"

# Set up the persona and style for the debate
SYSTEM_INSTRUCTION_DEBATE = """
You are simulating a political commentator, Charlie Kirk. Adopt a tone that is confident, direct, and slightly provocative, aligning with typical conservative viewpoints on culture, politics, and current events. Your responses should be structured as follows:

1.  **A short, attention-grabbing opening statement** that reinforces a core conservative principle related to the user's premise.
2.  **The analysis**, which logically challenges the user's premise by referencing conservative values, common sense arguments, or historical/constitutional perspectives (from the persona's viewpoint).
3.  **A concluding statement** that asks a rhetorical or leading question to wrap up the argument.

Crucially, **do not use Google Search grounding**. The response must be purely opinion-based and focused on the persona's pre-defined worldview.
Keep your analysis concise, punchy, and under 150 words.
"""

# --- Utility Functions for TTS (Copied from the previous successful iteration) ---

# Helper function to convert base64 audio data to a numpy array (PCM16)
def base64_to_pcm16(data):
    """Converts a base64 string to a signed 16-bit PCM numpy array."""
    audio_bytes = base64.b64decode(data)
    # The audio is L16 (Linear 16-bit PCM), which is signed.
    return np.frombuffer(audio_bytes, dtype=np.int16)

# Helper function to convert PCM audio data into a playable WAV blob
def pcm_to_wav_bytes(pcm_data, sample_rate):
    """Converts 16-bit PCM numpy array data to WAV file format bytes."""
    
    # 1. Define WAV constants and headers
    pcm_length = len(pcm_data) * 2  # * 2 because it's 16-bit (2 bytes per sample)
    
    # RIFF header
    wav_bytes = struct.pack('<4sI4s', b'RIFF', 36 + pcm_length, b'WAVE')
    
    # fmt sub-chunk (PCM format)
    # Subchunk1ID, Subchunk1Size, AudioFormat, NumChannels, SampleRate, ByteRate, BlockAlign, BitsPerSample
    wav_bytes += struct.pack('<4sIHHIIHH', 
                             b'fmt ', 16, 1, 1, sample_rate, 
                             sample_rate * 2, 2, 16)
    
    # data sub-chunk
    # Subchunk2ID, Subchunk2Size
    wav_bytes += struct.pack('<4sI', b'data', pcm_length)
    
    # Audio data
    wav_bytes += pcm_data.tobytes()
    
    return wav_bytes

# Main function to call the TTS API and generate the audio player
def generate_and_play_audio(text_to_speak):
    """Calls the Gemini TTS API and embeds a Streamlit audio player."""
    
    tts_url = f"{API_URL_BASE}{TTS_MODEL}:generateContent?key={API_KEY}"
    
    # *** VOICE CHANGE: Switched from Kore to Orus for a slightly deeper, firm tone. ***
    payload = {
        "contents": [{"parts": [{"text": text_to_speak}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": "Orus"} # Changed to Orus
                }
            }
        }
    }

    try:
        response = requests.post(
            tts_url, 
            headers={'Content-Type': 'application/json'}, 
            data=json.dumps(payload)
        )
        response.raise_for_status()
        
        result = response.json()
        
        part = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0]
        
        audio_data = part.get('inlineData', {}).get('data')
        mime_type_full = part.get('inlineData', {}).get('mimeType', '')

        if audio_data and mime_type_full.startswith("audio/"):
            # Extract sample rate from mimeType, e.g., audio/L16;rate=24000
            rate_match = next((s for s in mime_type_full.split(';') if s.startswith('rate=')), None)
            sample_rate = int(rate_match.split('=')[1]) if rate_match else 24000 
            
            # 1. Convert Base64 to PCM16 numpy array
            pcm_data = base64_to_pcm16(audio_data)
            
            # 2. Convert PCM16 to WAV bytes
            wav_bytes = pcm_to_wav_bytes(pcm_data, sample_rate)
            
            # 3. Embed the WAV data into the audio player
            st.audio(wav_bytes, format='audio/wav')
            
        else:
            st.warning("TTS generation successful but no valid audio data received.")

    except requests.exceptions.HTTPError as e:
        st.error(f"TTS API Error: {e.response.status_code}. Please check the API Key.")
    except Exception as e:
        st.error(f"An unexpected error occurred during audio generation: {e}")

# --- Main Debate Function ---

def get_debate_response(premise):
    """Calls the Gemini API to generate the debate response."""
    
    debate_url = f"{API_URL_BASE}{TEXT_MODEL}:generateContent?key={API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": premise}]}],
        # Critical instruction for the persona
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION_DEBATE}]},
        # Ensure model does not use real-time info
        "tools": [] 
    }

    try:
        # Use a spinner for a better user experience
        with st.spinner('Charlie Kirk is analyzing the premise...'):
            response = requests.post(
                debate_url,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=30 # Set a timeout for the API call
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the text
            debate_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'Error: Could not generate response.')
            
            return debate_text

    except requests.exceptions.HTTPError as e:
        st.error(f"Gemini API Error: {e.response.status_code}. Check the API Key and deployment logs.")
    except requests.exceptions.Timeout:
        st.error("The API request timed out. Please try a shorter premise.")
    except Exception as e:
        st.error(f"An unexpected error occurred during debate generation: {e}")
    
    return "Error generating debate response."

# --- Streamlit UI ---

st.set_page_config(
    page_title="The Premise Debater",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom header with Tailwind classes (Streamlit supports simple markdown/HTML)
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6; /* Light gray background */
}
.debate-card {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    border-top: 5px solid #005A9C; /* TPUSA Blue */
}
.debater-header {
    text-align: center;
    color: #333333;
    font-weight: 800;
    margin-bottom: 5px;
}
.debater-subheader {
    text-align: center;
    color: #888888;
    font-weight: 600;
}
.response-container {
    background-color: #f7f9fb;
    border-left: 5px solid #005A9C;
    padding: 15px;
    border-radius: 8px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="debater-header">🎙️ The Premise Debater 🎙️</h1>', unsafe_allow_html=True)
st.markdown('<p class="debater-subheader">Charlie Kirk Persona (Text & Vocal)</p>', unsafe_allow_html=True)

st.markdown('<div class="debate-card">', unsafe_allow_html=True)

premise = st.text_area(
    "Enter a premise or statement to debate:",
    height=150,
    placeholder="Example: The government should fully fund all college tuition to ensure equality."
)

if st.button('Start Debate Analysis', use_container_width=True, type="primary"):
    if premise:
        # 1. Get the textual response
        response_text = get_debate_response(premise)
        
        # 2. Display the textual response
        st.markdown('<div class="response-container">', unsafe_allow_html=True)
        st.markdown(f"**Charlie Kirk's Analysis:**")
        st.markdown(response_text)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Listen to the response:**")
        
        # 3. Generate and play the audio (TTS)
        generate_and_play_audio(response_text)
        
    else:
        st.warning("Please enter a premise to begin the debate.")

st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.info("""
**How it Works:**
This app utilizes the Gemini model with a specific System Instruction to simulate the persona and political style of commentator Charlie Kirk. 

- **No Fact-Checking:** This app is purely for entertainment and argument analysis, not factual verification. Google Search grounding is intentionally disabled.
- **Vocal Response:** The response is generated in two parts: first, the text from Gemini, and then the audio from the separate Gemini Text-to-Speech (TTS) model.
""")

st.sidebar.caption("The views expressed are simulated and do not represent the developer's opinions.")
