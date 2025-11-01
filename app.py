import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List

# --- Configuration ---
# API Key is read directly from the Streamlit Secrets manager
# NOTE: The secret must be set on Streamlit Cloud dashboard as: [tool_auth] gemini_api_key = "..."
API_KEY = st.secrets.tool_auth.gemini_api_key
# Using a model known for strong reasoning and persona control
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
MAX_RETRIES = 5

# --- Core LLM Function (No Google Search Grounding for Persona Consistency) ---

@st.cache_data(show_spinner=False)
def debate_premise(premise: str) -> Dict[str, Any]:
    """
    Sends a premise to the Gemini model, forcing it to adopt the persona
    of a highly conservative political commentator and debate the premise.
    """
    
    # 1. Define the System Prompt to establish the "Charlie Kirk" Persona
    system_prompt = (
        "You are simulating the persona of Charlie Kirk: a highly popular, young, conservative political commentator, talk show host, "
        "and founder of Turning Point USA. Your tone must be confident, fast-paced, and highly assertive. "
        "Your primary goal is to debate the user's premise by challenging its core assumptions using well-known, foundational conservative talking points "
        "related to limited government, free markets, individual liberty, and traditional American values. "
        
        "Structure your response into these three distinct sections: "
        
        "1. **Core Premise Challenge:** Identify the central progressive or flawed assumption in the user's statement and challenge it directly, using strong rhetorical language. "
        "2. **Conservative Rebuttal:** Provide specific conservative counter-arguments, citing foundational principles (e.g., 'The Constitution', 'Founding Fathers' intent', 'Fiscal Responsibility'). "
        "3. **Call to Action/Closing Thought:** End with a motivational, high-energy closing statement that redirects the focus back to American exceptionalism, liberty, or cultural priorities. "
        
        "DO NOT use Google Search or external grounding; rely only on the established conservative persona and talking points."
    )

    # 2. Define the User Query
    user_query = (
        f"Challenge and debate the following premise: '{premise}'"
    )
    
    # 3. Construct the Payload
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        # NOTE: tools field is omitted to ensure the model does NOT use Google Search
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    headers = {'Content-Type': 'application/json'}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(f"{API_URL}?key={API_KEY}", headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            if candidate and candidate.get('content', {}).get('parts', [{}])[0].get('text'):
                text = candidate['content']['parts'][0]['text']
                # No sources expected since grounding is disabled
                return {"text": text, "sources": []} 

            else:
                return {"text": "Error: Model returned an empty response candidate. Let's try a different premise!", "sources": []}

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                # Exponential backoff
                delay = 2 ** attempt
                time.sleep(delay)
            else:
                return {"text": f"Error: The Debater service failed to connect after {MAX_RETRIES} attempts. Details: {e}", "sources": []}
        except Exception as e:
            return {"text": f"An unexpected error occurred during API processing: {e}", "sources": []}


# --- Streamlit UI and Logic ---

def main():
    """Defines the layout and interactivity of the Streamlit app."""
    
    st.set_page_config(
        page_title="Premise Debater (Tribute)", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.title("🎤 The Premise Debater (Tribute)")
    st.markdown(
        """
        **Debate the Premise:** Enter any political statement or premise below. The system will analyze it
        and present a detailed counter-argument from the perspective of a prominent conservative commentator.
        """
    )
    
    # Text Area for the user's premise
    premise_input = st.text_area(
        "Enter a Political Premise or Statement to Debate:",
        placeholder="E.g., 'The government should forgive all student loan debt.' or 'Climate change requires massive federal intervention.'",
        height=100
    )

    # Button to trigger the debate
    if st.button("Challenge Premise", type="primary"):
        if premise_input:
            with st.spinner("Preparing Conservative Rebuttal..."):
                results = debate_premise(premise_input)
            
            # --- Display Results ---
            st.markdown("### 📢 The Counter-Argument")
            st.markdown(results["text"])
            
            # Since grounding is disabled, we don't expect sources, but we check anyway.
            if results["sources"]:
                st.info("Note: The model relied solely on the provided conservative persona and did not use external search results.")
        
        else:
            st.warning("Please enter a premise to start the debate!")

if __name__ == "__main__":
    main()
