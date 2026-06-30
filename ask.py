import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# load the GROQ_API_KEY from .env
load_dotenv()

# load the same embedding model used in ingest.py
model = SentenceTransformer('all-MiniLM-L6-v2')

# connect to the existing ChromaDB we built in ingest.py
client_db = chromadb.PersistentClient(path="./chroma_db")
collection = client_db.get_or_create_collection(name="codemate_chunks")

# set up the Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def get_embedding(text):
    """Convert text into a vector (same function as ingest.py)"""
    return model.encode(text).tolist()


def retrieve_relevant_chunks(question, n_results=4):
    """Embed the question and find the closest matching code chunks"""
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    # results comes back as nested lists, flatten into a clean list of dicts
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
    """Combine retrieved chunks + question into a prompt for the LLM"""
    context = ""
    for chunk in chunks:
        context += f"\n--- From {chunk['file']} (lines {chunk['start_line']}-{chunk['end_line']}) ---\n"
        context += chunk['text']
        context += "\n"

    prompt = f"""You are a helpful coding assistant. Use ONLY the following code context to answer the question. 
If the context doesn't contain enough information to answer, say so honestly instead of guessing.

CODE CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    return prompt


def ask_question(question):
    """Full pipeline: retrieve relevant chunks, then generate an answer"""
    chunks = retrieve_relevant_chunks(question)
    prompt = build_prompt(question, chunks)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    print("\n" + "=" * 60)
    print("ANSWER:")
    print(answer)
    print("\nSOURCES:")
    for chunk in chunks:
        print(f"  - {chunk['file']} (lines {chunk['start_line']}-{chunk['end_line']})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("CodeMate - Ask questions about the 'requests' repo")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Your question: ")
        if question.strip().lower() in ("quit", "exit"):
            break
        ask_question(question)