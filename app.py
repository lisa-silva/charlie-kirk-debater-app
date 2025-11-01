import streamlit as st
import requests
import struct
import numpy as np

# --- 1. CONFIGURATION ---

# IMPORTANT: API Key is pulled securely from Streamlit secrets
try:
    # Ensure this key matches the one in .streamlit/secrets.toml
    API_KEY = st.secrets.tool_auth.gemini_api_key
    if not API_KEY:
        st.error("API key not found. Please set 'gemini_api_key' in Streamlit secrets.")
        st.stop()
except AttributeError:
    st.error("Configuration error. Make sure 'gemini_api_key' is set under the [tool_auth] section in your secrets.toml.")
    st.stop()

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent"
MODEL_NAME = "gemini-2.5-flash-preview-tts"
DEBATER_VOICE = "Charon"
DEBATER_PERSONA = "a political commentator named Charlie Kirk. You are known for your firm, clear, and fast-paced delivery. You must respond as if you are passionately challenging the user's premise, using concise, persuasive arguments and speaking only for 1-2 sentences."
GROUNDING_TOOL = {"google_search": {}}

st.set_page_config(layout="centered", page_title="Premise Debater App")


# --- 2. UTILITY FUNCTIONS ---

def pcm_to_wav(pcm16, sample_rate, num_channels=1, bits_per_sample=16):
    """
    Converts raw 16-bit PCM (NumPy array) data into a WAV file format (bytes).
    This function replaces the previous incorrect JavaScript ArrayBuffer logic.
    """
    
    # Ensure input is a NumPy array of type int16 (signed 16-bit PCM)
    if not isinstance(pcm16, np.ndarray) or pcm16.dtype != np.int16:
        pcm16 = np.frombuffer(pcm16, dtype=np.int16)

    # WAV header size is 44 bytes
    data_size = pcm16.nbytes
    chunk_size = 36 + data_size
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)

    wav_header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', chunk_size, b'WAVE', b'fmt ', 16, 1, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample, b'data', data_size
    )

    # Combine header and audio data
    return wav_header + pcm16.tobytes()

@st.cache_data(show_spinner=False)
def get_debater_audio(user_premise):
    """
    Calls the Gemini API (TTS and Text) to get a vocalized and grounded response.
    """
    # 1. Generate text response and audio data simultaneously
    
    # System instruction focuses on the Charlie Kirk persona
    system_instruction = f"You are {DEBATER_PERSONA}"

    # The user query instructs the model to both generate text and audio
    user_query = f"Challenge the following premise in a concise, 1-2 sentence response. Keep your arguments sharp and direct, using your persona: '{user_premise}'"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "responseModalities": ["TEXT", "AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": DEBATER_VOICE}
                }
            }
        },
        "tools": [GROUNDING_TOOL], # Enable Google Search for grounding
    }

    headers = {
        'Content-Type': 'application/json'
    }

    # Attempt API call with exponential backoff (simplified for this context)
    try:
        response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
    except requests.exceptions.RequestException as e:
        st.error(f"API Request failed: {e}")
        return None, "Error: Failed to connect to API."

    candidate = result.get('candidates', [{}])[0]
    
    # 2. Extract Text
    text_part = next((p for p in candidate.get('content', {}).get('parts', []) if 'text' in p), {})
    generated_text = text_part.get('text', "Sorry, I couldn't generate a text response.")

    # 3. Extract Audio (PCM data)
    audio_part = next((p for p in candidate.get('content', {}).get('parts', []) if 'inlineData' in p), {})
    inline_data = audio_part.get('inlineData', {})
    
    if inline_data and 'data' in inline_data:
        # The API returns base64-encoded PCM audio data
        pcm_base64 = inline_data['data']
        pcm_bytes = np.frombuffer(st.runtime.legacy_caching.rehydrate(pcm_base64), dtype=np.int16)
        
        # 4. Convert PCM bytes to WAV format
        # The sample rate for flash-preview-tts is 24000 Hz
        wav_data = pcm_to_wav(pcm_bytes, sample_rate=24000)
        
        return wav_data, generated_text
    
    return None, generated_text


# --- 3. STREAMLIT UI ---

st.title("🎙️ The Vocal Premise Debater")
st.markdown(
    """
    **Persona:** Charlie Kirk (Voice: Charon).
    Enter a premise or a statement, and the Debater will challenge it using
    grounded arguments and speak the response aloud.
    """
)

# Input area
premise_input = st.text_area(
    "Enter your premise (e.g., 'The current deficit spending is beneficial for future economic growth.'):",
    key="premise_text",
    height=100,
    placeholder="Type your controversial premise here..."
)

# Button to trigger the debate
if st.button("Challenge My Premise", type="primary", use_container_width=True):
    if not premise_input:
        st.warning("Please enter a premise to be challenged.")
    else:
        with st.spinner("Engaging Debater... Generating voice and arguments..."):
            audio_bytes, text_response = get_debater_audio(premise_input)

        st.subheader("Debater's Response:")
        st.info(text_response)

        if audio_bytes:
            st.audio(audio_bytes, format='audio/wav')
        else:
            st.error("Could not generate audio. Please check the API response in the console for details.")

st.markdown("---")
st.markdown(f"")
