import streamlit as st
import pickle

# EX :- win big now | win free urgent offer limited limited urgent urgent free beyond baby physical environmental none meeting foreign low | unknownmail.cc
# EX :- project update | team sync president series today already involve lose control brother issue week blood firm personal let next | company.com


# Page configuration
st.set_page_config(page_title="Email Spam Detector", page_icon="📧")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Loading the model and vectorizer
try:
    model = pickle.load(open('spam_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'spam_model.pkl' ya 'vectorizer.pkl' file nahi mili. Pehle model train karke save karein.")

# UI Header
st.title("📧 Email Spam Classifier")
st.write("Apna email subject aur text niche enter karein check karne ke liye.")

# Input Section
with st.container():
    domain = st.text_input("Email Domain", placeholder="Write your email domain...")
    subject = st.text_input("Subject", placeholder="E.g. Congratulations! You won a prize")
    message = st.text_area("Email Content", placeholder="Write your email body here...", height=150)

# Prediction Logic
if st.button("Predict Now"):
    if message.strip() == "":
        st.warning("Please enter the email text to analyze.")
    else:
        # Combine subject and message (Common practice in spam detection)
        full_text = subject + " " + message + " " + domain
        
        # 1. Preprocess/Transform using vectorizer
        data = vectorizer.transform([full_text])
        
        # 2. Prediction
        prediction = model.predict(data)[0]
        
        # 3. Display Result
        st.divider()
        if prediction == 1: # Assuming 1 is Spam
            st.error("🚨 This is a SPAM email!")
        else:
            st.success("✅ This is a HAM (Safe) email.")

# Footer
st.caption("Built with Python & Streamlit")