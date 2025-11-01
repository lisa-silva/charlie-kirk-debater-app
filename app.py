import streamlit as st
import time
import requests
import json
import base64
import numpy as np
import io

# --- 1. CONFIGURATION AND SECRETS ---

# The app.py file must access the API key from st.secrets at the root level.
try:
    # This key is used for both text generation and TTS.
    API_KEY = st.secrets.gemini_api_key
except AttributeError:
    st.error("API key not found. Please set 'gemini_api_key' in Streamlit secrets.")
    st.stop()

# Base URL for the non-streaming Gemini API endpoint
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# The voice name we settled on (Charon: Informative/Polished)
VOICE_NAME = "Charon" 

# --- 2. GEMINI API HANDLERS ---

# Utility function for base64 to ArrayBuffer conversion (needed for audio context)
def base64ToArrayBuffer(base64):
    """Converts a base64 string to a raw ArrayBuffer."""
    decoded = base64.b64decode(base64)
    return decoded.buffer

def pcmToWav(pcm16, sampleRate=24000):
    """
    Converts raw 16-bit PCM audio data (Int16Array) into a standard WAV Blob.
    
    The Gemini TTS API returns audio/L16 (linear PCM, 16-bit, signed).
    We need to add the WAV header for browser playback.
    """
    # 44 bytes for the WAV header
    buffer = new ArrayBuffer(44 + pcm16.byteLength)
    view = new DataView(buffer)

    # RIFF chunk descriptor
    writeString(view, 0, 'RIFF')
    view.setUint32(4, 36 + pcm16.byteLength, true)
    writeString(view, 8, 'WAVE')

    # FMT sub-chunk
    writeString(view, 12, 'fmt ')
    view.setUint32(16, 16, true) # Sub-chunk size (16 for PCM)
    view.setUint16(20, 1, true)  # Audio format (1 for PCM)
    view.setUint16(22, 1, true)  # Number of channels (1)
    view.setUint32(24, sampleRate, true) # Sample rate
    view.setUint32(28, sampleRate * 2, true) # Byte rate (SampleRate * NumChannels * 2)
    view.setUint16(32, 2, true)  # Block align (NumChannels * 2)
    view.setUint16(34, 16, true) # Bits per sample (16)

    # DATA sub-chunk
    writeString(view, 36, 'data')
    view.setUint32(40, pcm16.byteLength, true)

    # Write the PCM data
    offset = 44
    for i in range(pcm16.size):
        view.setInt16(offset, pcm16[i], true)
        offset += 2

    return Blob([buffer], { type: 'audio/wav' })

def writeString(view, offset, string):
    """Helper function to write a string to a DataView."""
    for i in range(len(string)):
        view.setUint8(offset + i, string.charCodeAt(i))

# --- Text-to-Speech Function ---

def generate_tts_audio(text_to_speak):
    """Calls the Gemini TTS API to generate base64 audio data."""
    
    tts_model = "gemini-2.5-flash-preview-tts"
    url = f"{API_BASE_URL}{tts_model}:generateContent?key={API_KEY}"
    
    # System instruction to ensure a specific tone for the debate
    system_prompt = f"Speak the following response with an authoritative, firm, and informative tone."
    
    payload = {
        "contents": [
            {"parts": [{"text": text_to_speak}]}
        ],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": VOICE_NAME}
                }
            },
        },
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }

    # Exponential Backoff for API call
    for i in range(3):
        try:
            response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status() 
            result = response.json()
            
            # Extract audio data
            part = result['candidates'][0]['content']['parts'][0]
            audio_data = part['inlineData']['data']
            mime_type = part['inlineData']['mimeType']
            
            # The API returns audio/L16;rate=24000
            if audio_data and mime_type and mime_type.startswith("audio/L16"):
                return audio_data, mime_type
            else:
                st.error("TTS generation failed or returned invalid audio data.")
                return None, None
            
        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 503] and i < 2:
                # Retry on rate limit or server errors
                time.sleep(2 ** i)
            else:
                st.error(f"API Error (TTS): {e}")
                return None, None
        except Exception as e:
            st.error(f"An unexpected error occurred during TTS: {e}")
            return None, None
    return None, None

# --- Text Generation Function ---

def generate_text_response(premise, debate_history):
    """Calls the Gemini API for the debate text."""
    
    url = f"{API_BASE_URL}{MODEL_NAME}:generateContent?key={API_KEY}"
    
    # System instruction to define the Charlie Kirk persona
    system_prompt = (
        "You are Charlie Kirk, a conservative radio talk show host and activist. "
        "Your goal is to debate and challenge the user's premise. "
        "Your responses must be highly articulate, confident, and direct. "
        "Use a persuasive and firm tone. Structure your response into 3-4 distinct points, "
        "each starting with a bold headline, and end with a direct question to keep the debate going. "
        "Use simple, declarative sentences suitable for a radio commentary style."
    )
    
    # Format the chat history for the API call
    contents = []
    # Add the initial user premise
    contents.append({"role": "user", "parts": [{"text": f"Premise to challenge: {premise}"}]})
    
    # Add subsequent debate history
    for role, text in debate_history:
        contents.append({"role": role, "parts": [{"text": text}]})
    
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }

    # Exponential Backoff for API call
    for i in range(3):
        try:
            response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status() 
            result = response.json()
            
            # Extract generated text
            if result['candidates'][0]['content']['parts'][0]['text']:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "The model did not return a response."
            
        except requests.exceptions.HTTPError as e:
            if response.status_code in [429, 500, 503] and i < 2:
                time.sleep(2 ** i)
            else:
                st.error(f"API Error (Text Gen): {e}")
                return "An API error occurred. Please check your key and try again."
        except Exception as e:
            st.error(f"An unexpected error occurred during text generation: {e}")
            return "An unexpected error occurred. Check the console for details."
    return "Failed to get a response after multiple retries."


# --- 3. STREAMLIT UI COMPONENTS ---

# Function to play audio from base64 data using Streamlit components
def play_audio_from_base64(base64_audio_data, mime_type):
    """Encodes audio data into a playable format for Streamlit."""
    # Assuming base64_audio_data is the raw PCM data string
    # We need to construct a proper data URL for HTML5 audio
    
    # Since Streamlit's audio element is simple, we'll convert L16 PCM to WAV on the fly
    # Note: Streamlit's built-in audio player is usually simpler, but this ensures playback.
    
    # Decode base64 to bytes
    audio_bytes = base64.b64decode(base64_audio_data)
    
    # The Streamlit environment doesn't easily allow running the pcmToWav JavaScript 
    # function directly. A simplified approach is to convert to a WAV in Python:
    
    if mime_type.startswith("audio/L16"):
        # Extract sample rate from mime type if possible, default to 24000
        try:
            rate_str = mime_type.split('rate=')[1]
            sample_rate = int(rate_str)
        except:
            sample_rate = 24000
            
        # Convert bytes to a 16-bit numpy array
        pcm_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Write WAV header
        buffer = io.BytesIO()
        write_wav_header(buffer, pcm_data.size * 2, sample_rate)
        buffer.write(pcm_data.tobytes())
        wav_bytes = buffer.getvalue()
        
        # Convert WAV bytes to base64 for embedding
        b64_wav = base64.b64encode(wav_bytes).decode('utf-8')
        audio_src = f"data:audio/wav;base64,{b64_wav}"
        
        # Use Streamlit's HTML method for direct audio play
        st.markdown(f"""
            <audio controls autoplay>
                <source src="{audio_src}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
        """, unsafe_allow_html=True)
    else:
        st.error("Unsupported audio MIME type for playback.")


def write_wav_header(f, data_size, sample_rate, channels=1, bit_depth=16):
    """Writes a standard WAV file header to a file-like object."""
    import struct
    
    f.write(struct.pack('<4s', b'RIFF'))
    f.write(struct.pack('<I', 36 + data_size))
    f.write(struct.pack('<4s', b'WAVE'))
    f.write(struct.pack('<4s', b'fmt '))
    f.write(struct.pack('<I', 16))
    f.write(struct.pack('<H', 1))  # PCM format
    f.write(struct.pack('<H', channels))
    f.write(struct.pack('<I', sample_rate))
    f.write(struct.pack('<I', sample_rate * channels * bit_depth // 8))
    f.write(struct.pack('<H', channels * bit_depth // 8))
    f.write(struct.pack('<H', bit_depth))
    f.write(struct.pack('<4s', b'data'))
    f.write(struct.pack('<I', data_size))


# --- 4. STREAMLIT APP LOGIC ---

# Initialize chat history (to manage the debate context)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Title and App Setup
st.set_page_config(page_title="The Vocal Premise Debater", layout="centered")
st.title("🗣️ The Vocal Premise Debater")
st.markdown("---")
st.info("I am ready to debate any premise you propose. Drop your idea into the box below to start the conversation.")

# Display all prior messages in the chat interface
for role, text in st.session_state.messages:
    if role == "model":
        with st.chat_message("Charlie Kirk (AI)", avatar="🎙️"):
            st.markdown(text)
    else:
        with st.chat_message("User", avatar="👤"):
            st.markdown(text)

# Chat input container
if prompt := st.chat_input("Enter your premise or counter-argument here..."):
    # 1. Display User Message
    with st.chat_message("User", avatar="👤"):
        st.markdown(prompt)
    
    # Add user message to state
    st.session_state.messages.append(("user", prompt))
    
    # 2. Prepare for Model Response
    with st.chat_message("Charlie Kirk (AI)", avatar="🎙️"):
        with st.spinner("Charlie is formulating his counter-argument..."):
            
            # Separate the initial premise from the subsequent debate turns for the API call
            premise = st.session_state.messages[0][1] if st.session_state.messages else ""
            debate_history = st.session_state.messages[1:] 

            # 3. Generate Text Response
            response_text = generate_text_response(premise, debate_history)
            
            # 4. Generate TTS Audio and Play
            with st.spinner("Generating Voice Response..."):
                audio_data_b64, mime_type = generate_tts_audio(response_text)

            # 5. Display Text and Audio
            if audio_data_b64 and mime_type:
                # Play the generated audio
                play_audio_from_base64(audio_data_b64, mime_type)
            
            # Display the text response
            st.markdown(response_text)
            
            # Add model message to state
            st.session_state.messages.append(("model", response_text))
            
# Reset button for starting a new debate
if st.button("Start New Debate (Clear History)", key="reset_button"):
    st.session_state.messages = []
    st.experimental_rerun()
