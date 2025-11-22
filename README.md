# 📰 Czech Headline Generator

A demo Streamlit application for generating Czech headlines from article text using AI models.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ta2-headline-generation.streamlit.app/)

## Features

- 🔐 Password-protected access
- 📝 Multi-line text input for Czech articles
- ⚙️ Configurable generation parameters (number of headlines, word count limits)
- 🎨 Clean, responsive UI
- 🚀 Ready for model integration

## How to run locally

1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Open your browser to `http://localhost:8501`

4. Login with password: `slunicko`

## Deployment to Streamlit Cloud

1. Push your code to GitHub

2. Go to [share.streamlit.io](https://share.streamlit.io/)

3. Configure secrets in your app settings:
   ```toml
   PASSWORD = "your_secure_password_here"
   ```

4. The app will auto-deploy from the main branch

## Future Development

Currently uses a placeholder function for headline generation. The next step is to integrate a real Czech headline generation model (e.g., fine-tuned T5 or BART from Hugging Face).

