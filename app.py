import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import os
import time
import matplotlib.pyplot as plt
import pytesseract
from pdf2image import convert_from_bytes

# ✅ Path to Tesseract (update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Admin\Desktop\python projects\Tesseract-OCR\tesseract.exe"

# Function to set the background image
def set_background(image_file):
    """Sets a semi-transparent background image in Streamlit."""
    if not os.path.exists(image_file):
        st.error("Background image not found. Check the file path.")
        return
    
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    background_css = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{encoded_string}") no-repeat center center fixed;
        background-size: cover;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.3);
        z-index: -1;
    }}
    .block-container {{
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        margin-top: 80px;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

# Set background image
image_path = r"C:\Users\Admin\Pictures\resume ranking\backgroundAicte.png"
set_background(image_path)

# ✅ Extract text with OCR fallback
def extract_text_from_pdf(file):
    text = ""
    try:
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.warning(f"⚠️ Error reading PDF with PyPDF2: {e}")

    # OCR only if no text found
    if not text.strip():
        try:
            st.info(f"Using OCR for scanned resume: {file.name}")
            file.seek(0)
            images = convert_from_bytes(file.read())
            for img in images:
                text += pytesseract.image_to_string(img)
        except Exception as e:
            st.error(f"⚠️ OCR failed: {e}")

    return text

# ✅ Rank resumes and return Top 5
def rank_resumes(job_description, resumes, top_n=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    documents = [job_description] + resumes
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked_resumes = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
    return ranked_resumes[:top_n], similarity_scores

# ✅ File validation (size + pages)
def validate_resume(file, max_size_mb=10, max_pages=2):
    """Reject resumes larger than max_size_mb or more than max_pages."""
    file_size_mb = file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        st.warning(f"❌ {file.name} skipped: File size {file_size_mb:.2f} MB exceeds {max_size_mb} MB limit.")
        return False

    try:
        pdf = PdfReader(file)
        num_pages = len(pdf.pages)
        if num_pages > max_pages:
            st.warning(f"❌ {file.name} skipped: Resume has {num_pages} pages (limit: {max_pages}).")
            return False
    except Exception as e:
        st.error(f"⚠️ Could not read {file.name}: {e}")
        return False

    return True

# Page Title
st.markdown(
    "<h1 style='text-align: center; color: var(--text-color, black);'>Resume Ranking System</h1>",
    unsafe_allow_html=True,
)

# Dark Mode Toggle
dark_mode = st.checkbox("🌙 Dark Mode")
if dark_mode:
    st.markdown("""
        <style>
        body, .stApp { background-color: #121212 !important; color: white !important; }
        .block-container { background: rgba(50, 50, 50, 0.9) !important; }
        .stTextInput, .stTextArea, .stFileUploader { color: white !important; }
        .stProgress { background-color: #444 !important; }
        </style>
    """, unsafe_allow_html=True)

# Upload PDFs
uploaded_files = st.file_uploader("Upload resumes (PDF)", accept_multiple_files=True)
job_desc = st.text_area("Enter Job Description")

# Validate uploaded files
valid_files = []
if uploaded_files:
    st.subheader("Uploaded Files")
    for file in uploaded_files:
        st.write(f"📄 {file.name} ({file.size / 1024:.2f} KB)")
        file.seek(0)  # Reset pointer for validation
        if validate_resume(file):
            file.seek(0)  # Reset pointer again for later use
            valid_files.append(file)

# Rank resumes
if st.button("Rank Resumes"):
    if valid_files and job_desc:
        with st.spinner("Ranking resumes, please wait..."):
            time.sleep(2)

            # Extract text only from valid resumes
            resumes_text = [extract_text_from_pdf(file) for file in valid_files]
            
            # Rank (only top 5)
            rankings, scores = rank_resumes(job_desc, resumes_text, top_n=5)
            
            st.subheader("🏆 Top 5 Resumes")
            result_data = []
            for i, (index, score) in enumerate(rankings):
                st.write(f"Rank {i+1}: {valid_files[index].name} (Score: {score:.2f})")
                st.progress(int(score * 100))
                result_data.append([valid_files[index].name, score])
            
            df = pd.DataFrame(result_data, columns=["Resume Name", "Score"])
            
            # Bar Chart (Top 5 only)
            st.subheader("📊 Resume Ranking Visualization (Top 5)")
            fig, ax = plt.subplots()
            ax.barh([valid_files[i].name for i, _ in rankings], 
                    [score for _, score in rankings],
                    color=['blue', 'purple', 'orange', 'green', 'red'][:len(rankings)])
            ax.set_xlabel("Similarity Score")
            ax.set_title("Top 5 Resume Ranking Results")
            ax.invert_yaxis()
            st.pyplot(fig)
            
            # Download Top 5
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Top 5 Results", data=csv, file_name="top5_resumes.csv", mime="text/csv")
    else:
        st.warning("Please upload valid resumes (<=10MB, <=2 pages) and enter a job description.")
