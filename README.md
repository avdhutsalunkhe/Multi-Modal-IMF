# 📚 Multi-Modal IMF Document RAG

This project is a **multi-modal Retrieval-Augmented Generation (RAG)** system built for the **Big AIR Lab** assignment.

It reads a complex IMF PDF report (such as the Qatar Article IV) and breaks it into:

- Text chunks  
- Numeric tables  
- Figures (images) with OCR  

Then it allows you to **chat with the document**, with grounded answers that include **citations** (page number, content type, and relevance).

---

## 🔧 1. Tech Stack

- **Python** (3.10 recommended)
- **Streamlit** - interactive UI
- **PyMuPDF** - PDF parsing
- **Tesseract OCR** - extract text from images/charts
- **Sentence-Transformers (MiniLM)** - embedding model
- **FAISS** - vector database for fast retrieval
- **Transformers (FLAN-T5 or others)** - answer generation LLM

---

## 📁 2. Project Structure

```text
multi-model_assignment/
├── app.py                   # Streamlit chat UI
├── config.py                # Global config for paths & models
├── document_processor.py    # Extracts text, tables, and images (OCR)
├── process_document.py      # Phase 1–2: PDF ingestion + chunking
├── create_embeddings.py     # Phase 3: Create embeddings + FAISS index
├── vector_store.py          # FAISS vector store management
├── llm_qa.py                # LLM-based QA system
├── run_pipeline.py          # (Optional) Full pipeline runner
├── requirements.txt         # Python dependencies
└── data/
    ├── raw/                 # Input PDFs (e.g., qatar_test_doc.pdf)
    ├── processed/           # Extracted chunks (JSON)
    ├── images/              # Extracted chart/figure images (PNG)
    └── vector_store/        # FAISS index + metadata
```
# 🧠 Multi-Modal IMF RAG

A powerful document intelligence pipeline for analyzing IMF reports (or any PDF) using multimodal reasoning - combining OCR, text extraction, and vector search with FAISS.

---

## 🛠️ Setup Instructions

### 🔧 1. Clone the Repository
git clone https://github.com/avdhutsalunkhe/Multi-Model-IMF.git
cd multi-modal-imf-rag

text

---

### 🧱 2. Create a Virtual Environment
**Windows:**
python -m venv venv
venv\Scripts\activate

text

**Mac / Linux:**
python3 -m venv venv
source venv/bin/activate

text

---

### 📦 3. Install Required Python Packages
pip install --upgrade pip
pip install -r requirements.txt

text

---

### 🔍 4. Install Tesseract OCR (Required for Image/Chart Text Extraction)

**🪟 Windows Installation**
1. Download the installer from [UB Mannheim’s Tesseract page](https://github.com/UB-Mannheim/tesseract/wiki).
2. Install it (default path: `C:\Program Files\Tesseract-OCR`)
3. Update `document_processor.py`:
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

text
4. Verify installation:
tesseract --version

text
If you see a version number, Tesseract is successfully installed 🎉

---

### 📥 5. Add Your PDF Document
Place your PDF inside:
data/raw/

text
**Example:**
data/raw/qatar_test_doc.pdf

text
You may rename the file - just make sure it stays in the `raw` folder.

---

### 🚀 6. Run the Processing Pipeline

**Step 1 - Extract Text, Tables & Images (OCR)**
python process_document.py

text
Expected output:
Extracted 78 text chunks
Extracted 370 tables
Extracted 13 images with OCR
Total chunks: 461

Generated: data/processed/extracted_chunks.json

text

**Step 2 - Create Embeddings + Build FAISS Index**
python create_embeddings.py

text
This step:
- Loads extracted chunks  
- Embeds them using MiniLM  
- Builds a FAISS vector index  

Output:
FAISS index with 461 vectors
COMPLETE

text

---

### 💬 7. Run the Streamlit Chat Application
After embeddings are ready, launch:
streamlit run app.py

text

Expected console output:
Local URL: http://localhost:8501

text
Open in your browser 🚀

You’ll see:
- Sidebar for selecting the IMF report  
- Option to upload your own PDF  
- Model & document info  
- Interactive chat interface  

---

### 🧠 8. How to Use the App
**In the sidebar:**
- ✅ Choose a document (use the included Qatar IMF file or upload your own)
- 📤 If uploading:
  - Click **Process & Index This PDF**
  - Wait for extraction, embeddings, and FAISS index creation

**Try asking:**
- “Summarize Qatar’s recent economic performance.”
- “What are the key risks mentioned in the report?”
- “Show the fiscal balance for 2024.”
- “What does the inflation chart show?”

---

## 📘 License
This project is licensed under the [GNU General Public License v3.0](LICENSE).

---

## 👥 Contributors
- **Avdhut Salunkhe** - Author


---
