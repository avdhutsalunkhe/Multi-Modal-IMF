import streamlit as st
import os


import re
import pandas as pd


from vector_store import VectorStore
from llm_qa import LLMQA, SimpleQA
from document_processor import DocumentProcessor
import config


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Multi-Modal RAG – Document QA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- CUSTOM STYLES --------------------
st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #111827 0, #020617 50%, #000000 100%);
            color: #e5e7eb;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }


        .main-title h1 {
            font-size: 2.4rem;
            font-weight: 700;
            color: #e5e7eb;
            margin-bottom: 0.2rem;
        }
        .main-subtitle {
            font-size: 0.95rem;
            color: #9ca3af;
        }


        .chat-user {
            background: #1f2937;
            border-radius: 0.9rem;
            padding: 0.75rem 1rem;
            margin-bottom: 0.4rem;
            border: 1px solid #374151;
        }
        .chat-assistant {
            background: #020617;
            border-radius: 0.9rem;
            padding: 0.75rem 1rem;
            margin-bottom: 0.4rem;
            border: 1px solid #4b5563;
        }


        section[data-testid="stSidebar"] {
            background: #020617;
            border-left: 1px solid #111827;
        }
        .sidebar-header {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }
        .metric-label {
            font-size: 0.8rem;
            color: #9ca3af;
        }
        .metric-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: #e5e7eb;
        }
        .doc-card {
            border-radius: 0.9rem;
            border: 1px solid #1f2937;
            padding: 0.75rem 0.9rem;
            background: #020617;
            margin-bottom: 0.6rem;
        }
        .doc-card-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #e5e7eb;
        }
        .doc-card-body {
            font-size: 0.8rem;
            color: #9ca3af;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- helper to render table text as st.table ----------
def render_table_from_text(table_text: str):
    """
    Convert an extracted table text block into a nice grid and show it.
    Heuristic:
    - First non-empty line = header
    - Split on 2+ spaces OR 1+ spaces (fallback)
    - Each following line = row
    """


    # Clean & split into lines
    lines = [ln.strip() for ln in table_text.split("\n") if ln.strip()]


    if len(lines) < 2:
        st.text(table_text)
        return


    # ---------- 1. Header ----------
    header_line = lines[0]


    # Try: split by 2+ spaces
    headers = re.split(r"\s{2,}", header_line)
    # If that fails (only 1 column), fallback to single spaces
    if len(headers) == 1:
        headers = re.split(r"\s+", header_line)


    headers = [h.strip() for h in headers if h.strip()]


    # If still empty or only 1 header, just show plain text
    if len(headers) <= 1:
        st.text(table_text)
        return


    # ---------- 2. Rows ----------
    data_rows = []
    for ln in lines[1:]:
        # First try 2+ spaces
        parts = re.split(r"\s{2,}", ln)
        if len(parts) == 1:
            # fallback: 1+ spaces
            parts = re.split(r"\s+", ln)


        parts = [p.strip() for p in parts]


        # Pad / trim to match header length
        if len(parts) < len(headers):
            parts = parts + [""] * (len(headers) - len(parts))
        elif len(parts) > len(headers):
            parts = parts[: len(headers)]


        data_rows.append(parts)


    # ---------- 3. Build DataFrame ----------
    df = pd.DataFrame(data_rows, columns=headers)


    # ---------- 4. Style like your screenshot ----------
    styled = (
        df.style.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#8DB243"),  # green header
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("border", "1px solid #4B5563"),
                        ("text-align", "center"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border", "1px solid #4B5563"),
                        ("padding", "4px 6px"),
                    ],
                },
                {
                    "selector": "table",
                    "props": [
                        ("border-collapse", "collapse"),
                        ("font-size", "12px"),
                    ],
                },
            ]
        )
    )


    st.table(styled)


# --------- IMF Numeric Table Helper ---------
def render_imf_numeric_table(block: str):
    """
    Parse an IMF-style one-line table like the Page 42 example
    into a real table and display it with st.table.


    Assumes:
    - Year header: 2020 2021 ... 2029
    - Each row: label (words) then 10 numeric values.
    """
    years_pattern = r"2020 2021 2022 2023 2024 2025 2026 2027 2028 2029"
    m = re.search(years_pattern, block)
    if not m:
        # Not this type of table -> just show the text
        st.write(block)
        return


    years = block[m.start(): m.end()].split()
    tail = block[m.end():].strip()


    tokens = tail.split()
    number_re = re.compile(r"^-?\d+(\.\d+)?$")


    rows = []
    i = 0
    n_years = len(years)


    while i < len(tokens):
        # 1) collect label words until first number
        label_tokens = []
        while i < len(tokens) and not number_re.match(tokens[i]):
            # skip footnote markers like "1/" or "2/"
            if re.match(r"^\d+\/$", tokens[i]):
                i += 1
                continue
            label_tokens.append(tokens[i])
            i += 1


        if not label_tokens:
            break


        label = " ".join(label_tokens)


        # 2) collect exactly n_years numbers
        values = []
        while i < len(tokens) and len(values) < n_years:
            if number_re.match(tokens[i]):
                values.append(tokens[i])
            i += 1


        if len(values) != n_years:
            # incomplete row -> stop
            break


        rows.append([label] + values)


    if not rows:
        st.write(block)
        return


    # Build list of dicts for st.table
    columns = ["Item"] + years
    data = [dict(zip(columns, row)) for row in rows]


    st.table(data)


# --------- IMF-style Table Helpers ---------
def parse_imf_style_table(block: str) -> pd.DataFrame:
    """
    Parse an IMF-style table block like the one from Page 42 into a DataFrame.


    Assumes:
    - Years 2020..2029 appear consecutively and act as column headers.
    - Each row: label words, then 10 numeric values.
    """


    # 1) Find the header with years 2020..2029
    years_pattern = r"2020 2021 2022 2023 2024 2025 2026 2027 2028 2029"
    m = re.search(years_pattern, block)
    if not m:
        raise ValueError("Year header not found in block")


    years = block[m.start() : m.end()].split()
    # Remaining text after the year header
    tail = block[m.end() :].strip()


    # 2) Tokenize the remaining part
    tokens = tail.split()
    rows = []
    i = 0
    n_years = len(years)


    number_re = re.compile(r"^-?\d+(\.\d+)?$")


    while i < len(tokens):
        # Collect label words until we hit the first numeric token
        label_tokens = []
        while i < len(tokens) and not number_re.match(tokens[i]):
            # skip footnote markers like "1/" or "2/"
            if re.match(r"^\d+\/$", tokens[i]):
                i += 1
                continue
            label_tokens.append(tokens[i])
            i += 1


        if not label_tokens:
            break  # no more labels, stop


        label = " ".join(label_tokens)


        # Now collect exactly n_years numeric values
        values = []
        while i < len(tokens) and len(values) < n_years:
            if number_re.match(tokens[i]):
                values.append(tokens[i])
            # skip non-numeric junk
            i += 1


        if len(values) != n_years:
            # incomplete row; stop
            break


        rows.append([label] + values)


    # 3) Build DataFrame
    df = pd.DataFrame(rows, columns=["Item"] + years)
    return df


def render_imf_style_table_or_text(block: str):
    """
    Try to parse an IMF-style table. If parsing fails, just print the text.
    """
    try:
        df = parse_imf_style_table(block)
        st.dataframe(df, use_container_width=True)
    except Exception:
        # fallback: show plain text
        st.write(block)


# -------------------- SESSION STATE --------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_type" not in st.session_state:
    st.session_state.doc_type = None  # "qatar" or "uploaded"
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None
if "qatar_loaded" not in st.session_state:
    st.session_state.qatar_loaded = False
if "uploaded_processed" not in st.session_state:
    st.session_state.uploaded_processed = False
if "qa_loaded" not in st.session_state:
    st.session_state.qa_loaded = False


# -------------------- LOAD LLM ONCE --------------------
if not st.session_state.qa_loaded:
    with st.spinner(f"🚀 Loading LLM model: {config.LLM_MODEL}"):
        try:
            st.session_state.qa_system = LLMQA(model_name=config.LLM_MODEL)
        except Exception:
            st.warning("LLM model failed to load. Falling back to SimpleQA.", icon="⚠️")
            st.session_state.qa_system = SimpleQA()
        st.session_state.qa_loaded = True


# -------------------- SIDEBAR: DOCUMENT SELECTION --------------------
with st.sidebar:
    st.markdown('<div class="sidebar-header">📂 Document Selection</div>', unsafe_allow_html=True)


    doc_mode = st.radio(
        "Choose document source",
        ["Qatar IMF (preprocessed)", "Upload PDF (any topic)"],
        index=0,
    )


    # Qatar IMF – preprocessed FAISS index
    if doc_mode == "Qatar IMF (preprocessed)":
        st.markdown(
            '<div class="doc-card">'
            '<div class="doc-card-title">Qatar IMF Article IV Report</div>'
            '<div class="doc-card-body">'
            "Preprocessed multi-modal index (text, tables, OCR figures) "
            "built from the official IMF staff report for Qatar."
            "</div></div>",
            unsafe_allow_html=True,
        )


        if not st.session_state.qatar_loaded:
            base = config.VECTOR_STORE_PATH
            faiss_file1 = base
            faiss_file2 = base + ".faiss"


            if os.path.exists(faiss_file1) or os.path.exists(faiss_file2):
                with st.spinner("🔄 Loading preprocessed vector index…"):
                    try:
                        vector_store = VectorStore(model_name=config.EMBEDDING_MODEL)
                        vector_store.load(config.VECTOR_STORE_PATH)
                        st.session_state.vector_store = vector_store
                        st.session_state.doc_type = "qatar"
                        st.session_state.doc_name = "Qatar IMF Article IV (preprocessed)"


                        chunks = vector_store.chunks
                        text_count = sum(1 for c in chunks if c["type"] == "text")
                        table_count = sum(1 for c in chunks if c["type"] == "table")
                        image_count = sum(1 for c in chunks if c["type"] == "image")
                        st.session_state.doc_stats = {
                            "total": len(chunks),
                            "text": text_count,
                            "table": table_count,
                            "image": image_count,
                        }


                        st.session_state.qatar_loaded = True
                        st.session_state.uploaded_processed = False
                    except Exception as e:
                        st.error(f"Error loading preprocessed index: {e}")
            else:
                st.error("Preprocessed FAISS index not found. Run the pipeline first.", icon="❌")
                st.markdown(
                    """
                    ```bash
                    python process_document.py
                    python create_embeddings.py
                    ```
                    """,
                    unsafe_allow_html=False,
                )


    # Upload PDF mode
    else:
        st.markdown(
            '<div class="doc-card">'
            '<div class="doc-card-title">Custom PDF (any topic)</div>'
            '<div class="doc-card-body">'
            "Upload any structured PDF (report, paper, policy document). "
            "We will extract text, tables, and images (OCR), build embeddings, "
            "and enable QA over this specific file."
            "</div></div>",
            unsafe_allow_html=True,
        )


        uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])


        if uploaded_file is not None:
            process_clicked = st.button("⚙️ Process & Index This PDF")


            if process_clicked:
                with st.spinner("🔍 Reading PDF, extracting multi-modal content…"):
                    raw_upload_dir = os.path.join(config.RAW_DATA_DIR, "uploads")
                    os.makedirs(raw_upload_dir, exist_ok=True)
                    pdf_path = os.path.join(raw_upload_dir, uploaded_file.name)


                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.read())


                    processor = DocumentProcessor(pdf_path)
                    chunks = processor.process_document()
                    processor.close()


                    text_count = sum(1 for c in chunks if c["type"] == "text")
                    table_count = sum(1 for c in chunks if c["type"] == "table")
                    image_count = sum(1 for c in chunks if c["type"] == "image")


                    with st.spinner("📦 Creating embeddings & vector index for uploaded PDF…"):
                        vs = VectorStore(model_name=config.EMBEDDING_MODEL)
                        # create in-memory index (no save_path)
                        vs.create_embeddings(chunks)


                    st.session_state.vector_store = vs
                    st.session_state.doc_type = "uploaded"
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.doc_stats = {
                        "total": len(chunks),
                        "text": text_count,
                        "table": table_count,
                        "image": image_count,
                    }
                    st.session_state.uploaded_processed = True
                    st.session_state.qatar_loaded = False
                    st.session_state.chat_history = []


                    st.success("✅ PDF processed and indexed. You can start asking questions now!")


        else:
            st.info("Upload a PDF to enable QA for a new topic.")


    # Sidebar – system info & metrics
    st.markdown("---")
    st.markdown('<div class="sidebar-header">📊 System Status</div>', unsafe_allow_html=True)


    if st.session_state.vector_store and st.session_state.doc_stats:
        stats = st.session_state.doc_stats


        st.markdown('<div class="metric-label">Current Document</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{st.session_state.doc_name}</div>',
            unsafe_allow_html=True,
        )


        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="metric-label">Total Chunks</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{stats["total"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Embedding Model</div>', unsafe_allow_html=True)
            st.caption(config.EMBEDDING_MODEL)
        with c2:
            st.markdown('<div class="metric-label">LLM</div>', unsafe_allow_html=True)
            st.caption(config.LLM_MODEL)
            st.markdown('<div class="metric-label">Multi-modal Coverage</div>', unsafe_allow_html=True)
            st.caption(
                f"Text: {stats['text']} · Tables: {stats['table']} · Images: {stats['image']}"
            )


        if st.button("🧹 Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("No document indexed yet. Use the controls above to load or process a PDF.")


# -------------------- MAIN HEADER --------------------
st.markdown('<div class="main-title">', unsafe_allow_html=True)
st.title("Multi-Modal Document RAG QA")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">'
    "Select a document in the sidebar (preprocessed IMF report or your own PDF) "
    "and ask natural language questions. Answers are grounded in the document’s "
    "text, tables, and OCR-extracted figures, with rich citations."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("")


# -------------------- MAIN CHAT INTERFACE --------------------
vector_store = st.session_state.vector_store
qa_system = st.session_state.qa_system


if vector_store and qa_system and st.session_state.doc_stats:
    st.markdown("---")


    # Previous conversation
    for message in st.session_state.chat_history:
        role = message["role"]
        bubble_class = "chat-user" if role == "user" else "chat-assistant"
        with st.chat_message(role):
            st.markdown(
                f'<div class="{bubble_class}">{message["content"]}</div>',
                unsafe_allow_html=True,
            )
            if "citations" in message:
                with st.expander("📎 View Citations"):
                    for cite in message["citations"]:
                        st.markdown(
                            f"**{cite['source']}** "
                            f"| Type: `{cite['type']}` "
                            f"| Page: `{cite['page']}` "
                            f"| Relevance: `{cite['relevance_score']:.3f}`"
                        )
                        if cite["type"] == "text":
                            content = cite["content"]
                            # If it contains the 2020–2029 header, treat as IMF numeric table
                            if "2020 2021 2022 2023 2024 2025 2026 2027 2028 2029" in content:
                                render_imf_numeric_table(content)
                            else:
                                st.write(content)
                        elif cite["type"] == "table":
                            render_table_from_text(cite["content"])
                        elif cite["type"] == "image":
                            img_path = cite.get("image_path")
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, caption=cite["source"])
                            if cite.get("content"):
                                with st.expander("OCR Text"):
                                    st.write(cite["content"])
                        st.markdown("---")


    # New user query
    query = st.chat_input(
        f"Ask a question about: {st.session_state.doc_name or 'the selected document'}…"
    )


    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(
                f'<div class="chat-user">{query}</div>',
                unsafe_allow_html=True,
            )


        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving context and generating answer…"):
                search_results = vector_store.search(query, k=5)
                result = qa_system.generate_answer_with_citations(query, search_results)


                st.markdown(
                    f'<div class="chat-assistant">{result["answer"]}</div>',
                    unsafe_allow_html=True,
                )


                with st.expander("📎 View Citations"):
                    for cite in result["citations"]:
                        st.markdown(
                            f"**{cite['source']}** "
                            f"| Type: `{cite['type']}` "
                            f"| Page: `{cite['page']}` "
                            f"| Relevance: `{cite['relevance_score']:.3f}`"
                        )
                        if cite["type"] == "text":
                            content = cite["content"]
                            # If it contains the 2020–2029 header, treat as IMF numeric table
                            if "2020 2021 2022 2023 2024 2025 2026 2027 2028 2029" in content:
                                render_imf_numeric_table(content)
                            else:
                                st.write(content)
                        elif cite["type"] == "table":
                            render_table_from_text(cite["content"])
                        elif cite["type"] == "image":
                            img_path = cite.get("image_path")
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, caption=cite["source"])
                            if cite.get("content"):
                                with st.expander("OCR Text"):
                                    st.write(cite["content"])
                        st.markdown("---")


        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "citations": result["citations"],
            }
        )


else:
    st.markdown("---")
    st.info(
        "No document is active yet. In the sidebar, either:\n\n"
        "- Load the preprocessed Qatar IMF report, or\n"
        "- Upload and process a new PDF.",
        icon="ℹ️",
    )