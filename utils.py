# utils.py (Streamlit Cloud Optimized)
import streamlit as st
from google import genai
from google.genai import types

# Initialize GenAI client with Streamlit secrets
try:
    genai_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("❌ Missing GEMINI_API_KEY in Streamlit secrets")
    raise ValueError("GEMINI_API_KEY not found in Streamlit secrets")
except Exception as e:
    st.error(f"🔧 Configuration Error: {str(e)}")
    raise

def summarize_common(
    system_prompt: str,
    parts: list[types.Part],
    analysis_mode: str = "Quick Scan",
    model_name: str = "gemini-1.5-pro-latest"
) -> str:
    """
    Unified content generation function with error handling
    """
    token_config = {
        "Quick Scan": 1024,
        "Detailed Analysis": 2048,
        "Deep Dive": 4096
    }
    
    try:
        response = genai_client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=token_config.get(analysis_mode, 2048),
                system_instruction=system_prompt
            )
        )
        return response.text
    
    except Exception as e:
        st.error("⚠️ Failed to generate content")
        raise RuntimeError(f"API Error: {str(e)}") from e
