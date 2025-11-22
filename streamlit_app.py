import streamlit as st
import time

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
def generate_headlines(text: str, num_headlines: int, min_words: int, max_words: int) -> list[str]:
    """
    Placeholder function for headline generation.
    
    In the future, this will be replaced with a real Czech headline generation model.
    For example, a fine-tuned T5 or BART model from Hugging Face.
    
    Args:
        text: The article text to generate headlines for
        num_headlines: Number of headlines to generate
        min_words: Minimum number of words in each headline
        max_words: Maximum number of words in each headline
    
    Returns:
        List of generated headlines
    """
    # Simulate model processing time
    time.sleep(5)
    
    # Return dummy headlines for demo purposes
    headlines = []
    for i in range(num_headlines):
        word_count = min_words + (i % (max_words - min_words + 1))
        headlines.append(f"Demo Czech headline number {i+1} ({word_count} words)")
    
    return headlines


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
        if password == "slunicko":
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
        max_value=5,
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
