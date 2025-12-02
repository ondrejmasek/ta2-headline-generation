import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os


@st.cache_resource
def load_model():
    """
    Load the private HF model using the token stored in Streamlit secrets.
    Cached so the model loads only once.
    """
    hf_token = st.secrets["HF_TOKEN"]
    repo_id = st.secrets["HF_REPO_ID"]

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        use_auth_token=hf_token
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        repo_id,
        use_auth_token=hf_token
    )

    model.eval()
    return tokenizer, model

# Page configuration
st.set_page_config(
    page_title="Czech Headline Generator",
    page_icon="📰",
    layout="wide"
)

# ============================================================================
# PLACEHOLDER MODEL FUNCTION
# TODO: Replace this with a real fine-tuned model (e.g., T5 from Hugging Face)
# ============================================================================
def generate_headlines(text, num_headlines, min_words, max_words):
    tokenizer, model = load_model()

    if not text.strip():
        return [""] * num_headlines

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    min_tokens = max(min_words, 3)
    max_new_tokens = max(max_words + 4, min_tokens + 2)

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    num_beams = max(4, num_headlines)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_headlines,
            min_length=min_tokens,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

    cleaned = []
    for h in decoded:
        words = h.strip().split()
        if len(words) > max_words:
            words = words[:max_words]
        cleaned.append(" ".join(words))

    return cleaned


# ============================================================================
# AUTHENTICATION
# ============================================================================
# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Show login form if not authenticated
if not st.session_state.authenticated:
    st.title("🔐 Login")
    st.write("Enter the password to access the Czech headline generation application.")
    
    password = st.text_input("Password", type="password", key="password_input")
    
    if st.button("Sign In"):
        # Use secrets in production, fallback to hardcoded password for local testing
        try:
            correct_password = st.secrets["PASSWORD"]
        except (KeyError , FileNotFoundError):
            print(":(")
        
        if password == correct_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password! Please try again.")
    
    st.stop()  # Don't render anything below if not authenticated


# ============================================================================
# MAIN APP (shown only after successful authentication)
# ============================================================================
st.title("📰 Czech Headline Generator")
st.write("Demo application for generating Czech headlines from articles using an AI model.")

# Logout button in sidebar
with st.sidebar:
    st.write("### Settings")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Text input area
    article_text = st.text_area(
        "Paste article text here",
        height=300,
        placeholder="Paste the full Czech article text here for which you want to generate headlines...",
        help="Paste Czech article text. The model will generate several headline variants."
    )

with col2:
    st.write("### Model Parameters")
    
    # Number of headlines to generate
    num_headlines = st.slider(
        "Number of generated headlines",
        min_value=1,
        max_value=10,
        value=3,
        help="How many headline variants to generate"
    )
    
    # Minimum word count
    min_words = st.number_input(
        "Minimum headline word count",
        min_value=3,
        max_value=20,
        value=5,
        help="Minimum headline length in words"
    )
    
    # Maximum word count
    max_words = st.number_input(
        "Maximum headline word count",
        min_value=5,
        max_value=30,
        value=12,
        help="Maximum headline length in words"
    )
    
    # Validate that max >= min
    if max_words < min_words:
        st.warning("⚠️ Maximum must be greater than or equal to minimum!")
    
    st.write("")  # Spacing
    
    # Generate button
    generate_button = st.button(
        "🚀 Generate!",
        type="primary",
        use_container_width=True
    )

# Generation logic
if generate_button:
    # Validate input
    if not article_text.strip():
        st.warning("⚠️ Please paste article text before generating headlines!")
    elif max_words < min_words:
        st.error("❌ Maximum word count must be greater than or equal to minimum word count!")
    else:
        # Show loading state
        with st.spinner("🔄 Generating headlines..."):
            # Call the placeholder model function
            headlines = generate_headlines(
                text=article_text,
                num_headlines=num_headlines,
                min_words=min_words,
                max_words=max_words
            )
        
        # Display results
        st.success("✅ Headlines generated successfully!")
        
        st.write("### 📋 Generated Headlines")
        
        # Display headlines in a nice container
        for idx, headline in enumerate(headlines, 1):
            with st.container():
                st.markdown(f"""
                <div style="
                    padding: 15px;
                    margin: 10px 0;
                    background-color: #f0f2f6;
                    border-left: 4px solid #1f77b4;
                    border-radius: 5px;
                ">
                    <strong>Headline {idx}:</strong><br/>
                    {headline}
                </div>
                """, unsafe_allow_html=True)

# Footer
st.write("---")
st.caption("💡 Tip: This application currently uses a placeholder model. The real Czech headline generation model will be integrated later.")
