import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

# Global device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        token=hf_token,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        repo_id,
        token=hf_token,
    )

    model.to(device)
    model.eval()
    return tokenizer, model

# Page configuration
st.set_page_config(
    page_title="Czech Headline Generator",
    page_icon="📰",
    layout="wide"
)

# ============================================================================
# HEADLINE GENERATION FUNCTION
# ============================================================================
def generate_headlines(text, num_headlines, min_words, max_words):
    tokenizer, model = load_model()

    text = text.strip()
    if not text:
        return [""] * num_headlines

    # basic range sanity
    min_words = max(3, int(min_words))
    max_words = max(min_words + 1, int(max_words))

    # rough mapping words -> tokens (Czech words are often 1–2 tokens)
    # Add buffer to ensure complete sentences
    approx_min_tokens = int(min_words * 1.0)
    approx_max_new_tokens = int(max_words * 2.5)  # Increased buffer for complete headlines

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    # Decoding strategy:
    # - 1 headline  -> deterministic beam search for best quality
    # - >1 headline -> sampling for diversity
    if num_headlines == 1:
        do_sample = False
        num_beams = 4
        num_return_sequences = 1
        top_k = None
        top_p = None
        temperature = None
    else:
        do_sample = True
        num_beams = 1
        num_return_sequences = num_headlines
        top_k = 50
        top_p = 0.95
        temperature = 0.9

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            min_length=approx_min_tokens,
            max_new_tokens=approx_max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            early_stopping=False,  # Changed to False to avoid premature cutoff
            eos_token_id=tokenizer.eos_token_id,  # Ensure proper sentence ending
        )

    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Clean up and ensure we don't have incomplete words
    cleaned = []
    for h in decoded:
        h = " ".join(h.split()).strip()  # collapse extra spaces/newlines
        
        # Optional: truncate to max_words if significantly over, but preserve complete words
        words = h.split()
        if len(words) > max_words + 3:  # Only truncate if significantly over limit
            h = " ".join(words[:max_words+2])  # Adjusted to +2 to better fit max_words
        
        cleaned.append(h)

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
        "Minimum headline word count (approx.)",
        min_value=3,
        max_value=20,
        value=5,
        help="Minimum headline length in words"
    )
    
    # Maximum word count
    max_words = st.number_input(
        "Maximum headline word count (approx.)",
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
        # Show loading state with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Loading model...")
        progress_bar.progress(25)
        
        status_text.text("🔄 Processing article text...")
        progress_bar.progress(50)
        
        status_text.text("🔄 Generating headlines...")
        progress_bar.progress(75)
        
        # Call the model function
        headlines = generate_headlines(
            text=article_text,
            num_headlines=num_headlines,
            min_words=min_words,
            max_words=max_words
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Generation complete!")
        
        # Clear progress indicators after a brief moment
        import time
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
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
