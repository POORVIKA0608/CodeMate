import streamlit as st
import streamlit.components.v1 as components
import os
import shutil
import git
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="CodeMate", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0a0710;
}

#MainMenu, footer, header { visibility: hidden; }

.hero-wrap {
    padding: 3rem 0 1.5rem 0;
    text-align: center;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 4.2rem;
    line-height: 1;
    letter-spacing: -2px;
    background: linear-gradient(95deg, #FFD23D 0%, #FF8A3D 35%, #FF3D81 70%, #C13DFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 200% auto;
    animation: shine 6s ease-in-out infinite;
    margin-bottom: 0.4rem;
}

@keyframes shine {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 100% center; }
}

.hero-sub {
    color: #a3a0b8;
    font-size: 1.15rem;
    font-weight: 500;
    max-width: 560px;
    margin: 0 auto;
}

.flow-row {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    margin: 1.6rem 0 2.4rem 0;
    flex-wrap: wrap;
}

.flow-chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #d8d6e8;
    padding: 0.45rem 1.1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 500;
}

.flow-chip b {
    background: linear-gradient(95deg, #FFD23D, #FF3D81);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.step-label {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: #f4f2ff;
    margin-bottom: 0.6rem;
}

.step-num {
    background: linear-gradient(135deg, #FF3D81, #C13DFF);
    color: white;
    width: 26px;
    height: 26px;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
}

.glass-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(6px);
}

.stCaption, p, label, .stMarkdown { color: #c4c1d6 !important; }

pre, .stCodeBlock, div[data-testid="stCodeBlock"] {
    background-color: #15101f !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

div[data-testid="stCodeBlock"] code, pre code {
    color: #e8e6f0 !important;
}

code:not(pre code) {
    background-color: rgba(255,61,129,0.15) !important;
    color: #FF8A3D !important;
    border-radius: 4px !important;
    padding: 0.15rem 0.4rem !important;
}

.stTextInput > div > div > input,
input[type="text"],
.stTextInput input {
    background-color: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    padding: 0.7rem 1rem !important;
    caret-color: #ffffff !important;
}

.stTextInput > div > div > input::placeholder {
    color: rgba(255,255,255,0.4) !important;
}

.stTextInput input:focus {
    border-color: #FF3D81 !important;
    box-shadow: 0 0 0 1px #FF3D81 !important;
}

.stButton button {
    background: linear-gradient(95deg, #FF3D81 0%, #FF8A3D 100%) !important;
    color: #0F0E17 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.8rem !important;
    transition: transform 0.15s ease, box-shadow 0.2s ease !important;
    width: 100%;
}

.stButton button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 8px 24px rgba(255, 61, 129, 0.35) !important;
}

.stAlert { border-radius: 10px !important; }

div[data-testid="stExpander"] {
    background-color: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}

hr { border-color: rgba(255,255,255,0.1) !important; }

.answer-card {
    background: linear-gradient(135deg, rgba(255,61,129,0.08), rgba(193,61,255,0.06));
    border: 1px solid rgba(255,61,129,0.25);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0 1.6rem 0;
}
</style>
""", unsafe_allow_html=True)

# animated particle network background - dots connecting echoes how RAG links related chunks
# injected into the PARENT document (not the sandboxed iframe) so it renders behind the full app
components.html("""
<script>
(function() {
  const doc = window.parent.document;

  // avoid creating duplicate canvases on Streamlit reruns
  if (doc.getElementById('net-bg')) return;

  const c = doc.createElement('canvas');
  c.id = 'net-bg';
  c.style.position = 'fixed';
  c.style.top = '0';
  c.style.left = '0';
  c.style.width = '100vw';
  c.style.height = '100vh';
  c.style.zIndex = '-1';
  c.style.pointerEvents = 'none';
  doc.body.appendChild(c);

  const ctx = c.getContext('2d');
  let w = c.width = window.parent.innerWidth;
  let h = c.height = window.parent.innerHeight;

  const colors = ['#FF3D81', '#FF8A3D', '#FFD23D', '#C13DFF'];
  const N = 55;
  const pts = Array.from({length: N}, () => ({
    x: Math.random() * w,
    y: Math.random() * h,
    vx: (Math.random() - 0.5) * 0.35,
    vy: (Math.random() - 0.5) * 0.35,
    c: colors[Math.floor(Math.random() * colors.length)]
  }));

  function tick() {
    ctx.clearRect(0, 0, w, h);
    for (const p of pts) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
    }
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 140) {
          ctx.strokeStyle = `rgba(255,255,255,${0.08 * (1 - dist/140)})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.stroke();
        }
      }
    }
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
      ctx.fillStyle = p.c;
      ctx.globalAlpha = 0.65;
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    requestAnimationFrame(tick);
  }
  tick();

  window.parent.addEventListener('resize', () => {
    w = c.width = window.parent.innerWidth;
    h = c.height = window.parent.innerHeight;
  });
})();
</script>
""", height=0)



# ---- cached resources so they only load once ----
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


model = load_embedding_model()
groq_client = load_groq_client()


# ---- ingestion functions (same logic as ingest.py) ----
def get_python_files(repo_path):
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ('.git', 'venv', '__pycache__')]
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def read_file_content(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def chunk_text(text, filepath, chunk_size=40, overlap=5):
    lines = text.split('\n')
    chunks = []
    start = 0
    while start < len(lines):
        end = start + chunk_size
        chunk_lines = lines[start:end]
        chunk_str = '\n'.join(chunk_lines)
        if chunk_str.strip():
            chunks.append({
                "text": chunk_str,
                "file": filepath,
                "start_line": start + 1,
                "end_line": min(end, len(lines))
            })
        start += chunk_size - overlap
    return chunks


def get_embedding(text):
    return model.encode(text).tolist()


def clone_and_index_repo(repo_url):
    """Clone a fresh repo and build a NEW chromadb collection for it"""
    clone_path = "current_repo"

    # remove any previously cloned repo
    if os.path.exists(clone_path):
        shutil.rmtree(clone_path, onerror=lambda f, p, e: os.chmod(p, 0o777) or f(p))

    # clone the new one
    git.Repo.clone_from(repo_url, clone_path)

    # read + chunk all python files
    files = get_python_files(clone_path)
    if not files:
        return None, 0

    all_chunks = []
    for filepath in files:
        content = read_file_content(filepath)
        chunks = chunk_text(content, filepath)
        all_chunks.extend(chunks)

    # use a fresh in-memory chromadb collection each time (no leftover data between repos)
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name="current_repo_chunks")

    documents = [chunk['text'] for chunk in all_chunks]
    metadatas = [{
        "file": chunk['file'],
        "start_line": chunk['start_line'],
        "end_line": chunk['end_line']
    } for chunk in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    # batch-embed all chunks in one call instead of looping one at a time - much faster
    embeddings = model.encode(documents, show_progress_bar=False).tolist()

    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    return collection, len(all_chunks)


def retrieve_relevant_chunks(collection, question, n_results=4):
    question_embedding = get_embedding(question)
    results = collection.query(query_embeddings=[question_embedding], n_results=n_results)
    chunks = []
    for i in range(len(results['documents'][0])):
        chunks.append({
            "text": results['documents'][0][i],
            "file": results['metadatas'][0][i]['file'],
            "start_line": results['metadatas'][0][i]['start_line'],
            "end_line": results['metadatas'][0][i]['end_line']
        })
    return chunks


def build_prompt(question, chunks):
    context = ""
    for chunk in chunks:
        context += f"\n--- From {chunk['file']} (lines {chunk['start_line']}-{chunk['end_line']}) ---\n"
        context += chunk['text']
        context += "\n"

    return f"""You are a helpful coding assistant. Use ONLY the following code context to answer the question.
If the context doesn't contain enough information to answer, say so honestly instead of guessing.

CODE CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def ask_question(collection, question):
    chunks = retrieve_relevant_chunks(collection, question)
    prompt = build_prompt(question, chunks)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content, chunks


# ---- UI ----
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">CodeMate</div>
    <div class="hero-sub">Drop in any public GitHub repo and ask it anything — answers backed by the actual source code.</div>
</div>
<div class="flow-row">
    <div class="flow-chip"><b>Retrieve</b> → relevant code chunks</div>
    <div class="flow-chip"><b>Generate</b> → grounded answers</div>
    <div class="flow-chip"><b>Cite</b> → exact file & line</div>
</div>
""", unsafe_allow_html=True)

# session_state keeps the indexed repo's collection alive between interactions
if "collection" not in st.session_state:
    st.session_state.collection = None
    st.session_state.repo_name = None

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="step-label"><span class="step-num">1</span> Index a repository</div>', unsafe_allow_html=True)
repo_url = st.text_input("GitHub repo URL", placeholder="https://github.com/psf/requests", label_visibility="collapsed").strip()
index_clicked = st.button("🚀  Index this repo", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

if index_clicked and repo_url:
    with st.spinner("Cloning repo + generating embeddings... larger repos can take 2-5 minutes"):
        try:
            collection, num_chunks = clone_and_index_repo(repo_url)
            if collection is None:
                st.error("⚠️ No Python files found in this repo.")
            else:
                st.session_state.collection = collection
                st.session_state.repo_name = repo_url
                st.success(f"✅ Indexed {num_chunks} code chunks from {repo_url}")
        except Exception as e:
            st.error(f"❌ Failed to clone/index repo: {e}")

if st.session_state.collection is not None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="step-label"><span class="step-num">2</span> Ask about <code>{st.session_state.repo_name.split("/")[-1]}</code></div>', unsafe_allow_html=True)
    question = st.text_input("Your question", placeholder="e.g. How does this library handle timeouts?", label_visibility="collapsed")
    ask_clicked = st.button("✨  Ask")
    st.markdown('</div>', unsafe_allow_html=True)

    if ask_clicked and question:
        with st.spinner("Searching code and generating answer..."):
            answer, chunks = ask_question(st.session_state.collection, question)

        st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

        st.markdown("**📚 Sources**")
        for chunk in chunks:
            with st.expander(f"📄 {chunk['file']} (lines {chunk['start_line']}-{chunk['end_line']})"):
                st.code(chunk['text'], language="python")
else:
    st.info("👆 Paste a GitHub repo URL above and click 'Index this repo' to get started.")