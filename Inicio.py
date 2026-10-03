import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="TF-IDF Match",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# ESTILOS — inspirado en la referencia: coral, peach, cards
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --coral: #ff5b4d;
    --coral-2: #ff7651;
    --orange: #ffad3d;
    --peach: #fff0e9;
    --cream: #fff8f4;
    --ink: #171717;
    --muted: #756b68;
    --purple: #bba7df;
    --line: #f0d9d2;
    --white: #ffffff;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 82% 2%, rgba(255, 121, 81, .18), transparent 24%),
        radial-gradient(circle at 5% 20%, rgba(255, 173, 61, .10), transparent 20%),
        linear-gradient(180deg, #fff8f4 0%, #fffaf8 52%, #ffffff 100%);
    color: var(--ink);
}

/* Ocultar elementos de Streamlit que rompen el look de app */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Tipografía */
h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.7px !important;
}

h1 {
    font-size: 2.25rem !important;
    line-height: 1.05 !important;
}

h2 {
    font-size: 1.35rem !important;
}

h3 {
    font-size: 1.05rem !important;
}

p, label, li {
    color: #514946 !important;
}

/* Contenedor principal */
.block-container {
    max-width: 1180px !important;
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}

/* Header */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 22px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 11px;
}

.brand-icon {
    width: 42px;
    height: 42px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.25rem;
    background: linear-gradient(135deg, var(--coral), var(--orange));
    box-shadow: 0 8px 18px rgba(255, 91, 77, .22);
}

.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--ink);
}

.brand-sub {
    color: #9a8984;
    font-size: .74rem;
    margin-top: 1px;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    min-height: 250px;
    padding: 28px 32px;
    border-radius: 26px;
    background:
        radial-gradient(circle at 92% 8%, rgba(255,255,255,.25), transparent 23%),
        linear-gradient(135deg, #ff5b4d 0%, #ff7050 47%, #ffab3e 100%);
    box-shadow: 0 18px 38px rgba(229, 90, 70, .20);
    margin-bottom: 22px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -85px;
    bottom: -145px;
    border-radius: 50%;
    background: rgba(255,255,255,.13);
}

.hero-eyebrow {
    color: rgba(255,255,255,.84);
    font-size: .78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.hero-title {
    position: relative;
    z-index: 2;
    max-width: 620px;
    margin-top: 7px;
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.15rem;
    font-weight: 700;
    line-height: 1.08;
}

.hero-copy {
    position: relative;
    z-index: 2;
    max-width: 600px;
    margin-top: 9px;
    color: rgba(255,255,255,.86);
    font-size: .95rem;
    line-height: 1.55;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 18px;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.22);
    color: white;
    font-size: .78rem;
    font-weight: 600;
}

/* Cards */
.card {
    background: rgba(255,255,255,.93);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 9px 25px rgba(70, 43, 35, .055);
    margin-bottom: 18px;
}

.card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 3px;
}

.card-subtitle {
    color: #958681;
    font-size: .79rem;
    margin-bottom: 15px;
}

/* Inputs */
textarea, input[type="text"] {
    background: #fffaf8 !important;
    color: var(--ink) !important;
    border: 1px solid #efd8d1 !important;
    border-radius: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .9rem !important;
}

textarea {
    min-height: 150px !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--coral) !important;
    box-shadow: 0 0 0 3px rgba(255,91,77,.12) !important;
}

textarea::placeholder,
input::placeholder {
    color: #b0a19d !important;
}

/* Botones */
.stButton > button {
    border: 0 !important;
    border-radius: 14px !important;
    min-height: 46px !important;
    padding: .7rem 1.15rem !important;
    background: linear-gradient(135deg, var(--coral), var(--coral-2)) !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: .86rem !important;
    font-weight: 700 !important;
    box-shadow: 0 9px 19px rgba(255,91,77,.20) !important;
    transition: .18s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 24px rgba(255,91,77,.28) !important;
    background: linear-gradient(135deg, #ef4d42, #ff684b) !important;
}

[data-testid="stDownloadButton"] button {
    border-radius: 14px !important;
    min-height: 44px !important;
    background: #191919 !important;
    color: white !important;
    border: 0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}

[data-testid="stDownloadButton"] button:hover {
    background: #333 !important;
}

/* Métricas */
.metric-card {
    background: linear-gradient(145deg, #fff2ec, #fffaf8);
    border: 1px solid #f1d8d0;
    border-radius: 17px;
    padding: 17px 18px;
    min-height: 105px;
}

.metric-label {
    color: #9a817a;
    font-size: .72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
}

.metric-value {
    color: var(--ink);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 700;
    margin-top: 5px;
}

.metric-note {
    color: #a08d87;
    font-size: .72rem;
    margin-top: 2px;
}

/* Score */
.score-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(145deg, #ff6550, #ff8b55);
    color: white;
    border-radius: 20px;
    padding: 22px;
    min-height: 190px;
    box-shadow: 0 14px 28px rgba(255,91,77,.18);
}

.score-card:after {
    content: "";
    position: absolute;
    width: 170px;
    height: 170px;
    right: -75px;
    top: -70px;
    border-radius: 50%;
    background: rgba(255,255,255,.15);
}

.score-label {
    color: rgba(255,255,255,.84);
    font-size: .76rem;
    font-weight: 700;
}

.score-number {
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.3rem;
    line-height: 1;
    font-weight: 700;
    margin: 6px 0;
}

.score-status {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,.18);
    font-size: .72rem;
    font-weight: 600;
}

.score-bar {
    height: 7px;
    border-radius: 99px;
    background: rgba(255,255,255,.28);
    margin-top: 19px;
    overflow: hidden;
}

.score-fill {
    width: 72%;
    height: 100%;
    border-radius: 99px;
    background: white;
}

/* Respuesta */
.answer-card {
    background: linear-gradient(145deg, #f1e8ff, #fff2ec);
    border: 1px solid #e5d9ef;
    border-radius: 20px;
    padding: 22px;
}

.answer-badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: #ffffff;
    color: #73599d;
    font-size: .7rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.answer-text {
    color: #241f29;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    line-height: 1.4;
}

/* Resultado por documento */
.result-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 11px 12px;
    border: 1px solid #f0e0db;
    border-radius: 13px;
    background: #fffdfc;
    margin: 7px 0;
}

.result-rank {
    min-width: 34px;
    text-align: center;
    border-radius: 9px;
    padding: 5px 7px;
    background: #fff0eb;
    color: #dc4e43;
    font-size: .72rem;
    font-weight: 700;
}

.result-name {
    flex: 1;
    color: #38302d;
    font-size: .82rem;
    font-weight: 600;
}

.result-score {
    color: #b34d43;
    font-family: 'Space Grotesk', sans-serif;
    font-size: .82rem;
    font-weight: 700;
}

/* Tabla */
[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid #efddd7 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #8c7c77 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--coral) !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: var(--coral) !important;
}

/* Expander */
div[data-testid="stExpander"] {
    background: #fffdfc !important;
    border: 1px solid #efddd7 !important;
    border-radius: 14px !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 14px !important;
}

/* Divider */
hr {
    border-color: #f0ded8 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1818 !important;
}

[data-testid="stSidebar"] * {
    color: #eee !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="app-header">
    <div class="brand">
        <div class="brand-icon">✦</div>
        <div>
            <div class="brand-name">TF-IDF Match</div>
            <div class="brand-sub">Question & answer intelligence</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Text intelligence · English</div>
    <div class="hero-title">Find the document that best matches your question.</div>
    <div class="hero-copy">
        Compare your question against a collection of documents using TF-IDF,
        stemming and cosine similarity.
    </div>
    <div class="hero-pill">✦ Smart matching &nbsp;·&nbsp; Snowball stemming</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# STEMMER
# ============================================================
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    return [stemmer.stem(t) for t in tokens]

# ============================================================
# ENTRADA
# ============================================================
left, right = st.columns([1.35, 0.65], gap="large")

with left:
    st.markdown("""
    <div class="card">
        <div class="card-title">Your documents</div>
        <div class="card-subtitle">One document per line · English only</div>
    """, unsafe_allow_html=True)

    text_input = st.text_area(
        "Documentos",
        "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together.",
        label_visibility="collapsed",
        height=170,
    )

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class="card">
        <div class="card-title">How it works</div>
        <div class="card-subtitle">Three simple steps</div>
        <div class="uso-tag">01 · Normalize</div>
        <div class="uso-tag">02 · TF-IDF</div>
        <div class="uso-tag">03 · Similarity</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <div class="card-title">Your question</div>
    <div class="card-subtitle">Write the question you want to match against the documents</div>
""", unsafe_allow_html=True)

question = st.text_input(
    "Pregunta",
    "Who is playing?",
    label_visibility="collapsed",
)

run = st.button("✦  CALCULATE MATCH", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PROCESAMIENTO
# ============================================================
if run:
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:
        st.warning("Add at least one document to continue.")
        st.stop()

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_and_stem,
        stop_words="english",
        token_pattern=None
    )

    try:
        X = vectorizer.fit_transform(documents)
    except ValueError:
        st.error("The documents do not contain enough usable English terms.")
        st.stop()

    df_tfidf = pd.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=[f"Doc {i+1}" for i in range(len(documents))]
    )

    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, X).flatten()

    best_idx = similarities.argmax()
    best_doc = documents[best_idx]
    best_score = similarities[best_idx]

    # ========================================================
    # RESULTADO PRINCIPAL
    # ========================================================
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Documents</div>
            <div class="metric-value">{len(documents)}</div>
            <div class="metric-note">documents analyzed</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Vocabulary</div>
            <div class="metric-value">{len(vectorizer.get_feature_names_out())}</div>
            <div class="metric-note">unique stems</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Best match</div>
            <div class="metric-value">Doc {best_idx + 1}</div>
            <div class="metric-note">highest similarity</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Similarity</div>
            <div class="metric-value">{best_score:.3f}</div>
            <div class="metric-note">cosine score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    result_left, result_right = st.columns([0.95, 1.05], gap="large")

    with result_left:
        percentage = int(round(best_score * 100))
        percentage = max(0, min(100, percentage))

        st.markdown(f"""
        <div class="score-card">
            <div class="score-label">MATCH SCORE · DOC {best_idx + 1}</div>
            <div class="score-number">{best_score:.3f}</div>
            <span class="score-status">● Best semantic match</span>
            <div class="score-bar"><div class="score-fill" style="width:{percentage}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

    with result_right:
        st.markdown(f"""
        <div class="answer-card">
            <div class="answer-badge">BEST ANSWER</div>
            <div class="answer-text">{best_doc}</div>
            <div style="margin-top:12px;color:#817580;font-size:.78rem;">
                Document {best_idx + 1} · cosine similarity {best_score:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # TODOS LOS RESULTADOS
    # ========================================================
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    sim_df = pd.DataFrame({
        "Documento": [f"Doc {i+1}" for i in range(len(documents))],
        "Texto": documents,
        "Similitud": similarities
    }).sort_values("Similitud", ascending=False).reset_index(drop=True)

    rank_col, tfidf_col = st.columns([1, 1.15], gap="large")

    with rank_col:
        st.markdown("""
        <div class="card">
            <div class="card-title">Similarity ranking</div>
            <div class="card-subtitle">Documents ordered by cosine similarity</div>
        """, unsafe_allow_html=True)

        for rank, row in sim_df.iterrows():
            original_idx = int(row["Documento"].replace("Doc ", "")) - 1
            score = float(row["Similitud"])

            st.markdown(f"""
            <div class="result-row">
                <div class="result-rank">#{rank + 1}</div>
                <div class="result-name">Doc {original_idx + 1}</div>
                <div class="result-score">{score:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with tfidf_col:
        st.markdown("""
        <div class="card">
            <div class="card-title">TF-IDF matrix</div>
            <div class="card-subtitle">Normalized values after stemming</div>
        """, unsafe_allow_html=True)

        st.dataframe(
            df_tfidf.round(3),
            use_container_width=True,
            height=280
        )

        csv_tfidf = df_tfidf.to_csv(index=True).encode("utf-8")
        st.download_button(
            "↓  Export TF-IDF CSV",
            data=csv_tfidf,
            file_name="tfidf_matrix.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # STEMS COINCIDENTES
    # ========================================================
    q_stems = tokenize_and_stem(question)
    vocab = vectorizer.get_feature_names_out()
    matched = [
        s for s in q_stems
        if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0
    ]

    st.markdown(f"""
    <div class="card">
        <div class="card-title">Question analysis</div>
        <div class="card-subtitle">Stems from the question that appear in the selected document</div>
    """, unsafe_allow_html=True)

    if matched:
        for stem in matched:
            st.markdown(f'<span class="uso-tag">{stem}</span>', unsafe_allow_html=True)
    else:
        st.info("No question stems were found directly in the selected document.")

    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("View processed documents and similarity table"):
        st.dataframe(
            sim_df.style.format({"Similitud": "{:.3f}"}),
            use_container_width=True
        )
else:
    st.markdown("""
    <div class="card" style="text-align:center;padding:30px 22px;">
        <div style="font-size:1.7rem;margin-bottom:8px;">✦</div>
        <div class="card-title">Ready when you are</div>
        <div class="card-subtitle" style="margin-bottom:0;">
            Add your documents, type a question and calculate the best match.
        </div>
    </div>
    """, unsafe_allow_html=True)
