import os
import chromadb
from sentence_transformers import SentenceTransformer

# load the embedding model (downloads once, then cached locally)
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_python_files(repo_path):
    """Walk through the repo and find all .py files"""
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        # skip the venv and .git folders
        dirs[:] = [d for d in dirs if d not in ('.git', 'venv', '__pycache__')]
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def read_file_content(filepath):
    """Read the content of a single file"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def chunk_text(text, filepath, chunk_size=40, overlap=5):
    """Split text into overlapping chunks of `chunk_size` lines"""
    lines = text.split('\n')
    chunks = []

    start = 0
    while start < len(lines):
        end = start + chunk_size
        chunk_lines = lines[start:end]
        chunk_str = '\n'.join(chunk_lines)

        if chunk_str.strip():  # skip empty chunks
            chunks.append({
                "text": chunk_str,
                "file": filepath,
                "start_line": start + 1,
                "end_line": min(end, len(lines))
            })

        start += chunk_size - overlap  # move forward, but overlap a bit

    return chunks


def get_embedding(text):
    """Convert text into a vector"""
    return model.encode(text).tolist()


if __name__ == "__main__":
    files = get_python_files("test_repo")
    print(f"Found {len(files)} Python files:")

    all_chunks = []
    for filepath in files:
        content = read_file_content(filepath)
        chunks = chunk_text(content, filepath)
        all_chunks.extend(chunks)

    print(f"\nCreated {len(all_chunks)} total chunks")

    # set up chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="codemate_chunks")

    print("\nEmbedding and storing all chunks... this may take a minute")

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(all_chunks):
        documents.append(chunk['text'])
        embeddings.append(get_embedding(chunk['text']))
        metadatas.append({
            "file": chunk['file'],
            "start_line": chunk['start_line'],
            "end_line": chunk['end_line']
        })
        ids.append(f"chunk_{i}")

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(all_chunks)} chunks...")

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"\nDone! Stored {collection.count()} chunks in ChromaDB.")