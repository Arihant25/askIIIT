import os
import numpy as np
import faiss
import ollama
from sentence_transformers import SentenceTransformer


EMBEDDINGS_DIR = "embeddings"
INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_index.meta"

# -----------------
# Load embeddings & build FAISS index
# -----------------


# --- Helper: get embedding files info ---
def get_embedding_files_info():
    files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npz")]
    info = {}
    for f in files:
        path = os.path.join(EMBEDDINGS_DIR, f)
        stat = os.stat(path)
        info[f] = stat.st_mtime
    return info


# --- Load or rebuild index ---
def load_embeddings_and_index():
    texts = []
    embeddings_list = []
    for filename in os.listdir(EMBEDDINGS_DIR):
        if filename.endswith(".npz"):
            data = np.load(os.path.join(EMBEDDINGS_DIR, filename), allow_pickle=True)
            chunks = data["texts"]
            embeds = data["embeddings"]
            texts.extend(chunks)
            embeddings_list.append(embeds)
    if not texts:
        raise ValueError("No embeddings found — run the PDF embedding script first.")
    embeddings = np.vstack(embeddings_list).astype("float32")
    return texts, embeddings


def save_index(index, path):
    faiss.write_index(index, path)


def load_index(path):
    return faiss.read_index(path)


def save_meta(meta, path):
    import json

    with open(path, "w") as f:
        json.dump(meta, f)


def load_meta(path):
    import json

    with open(path, "r") as f:
        return json.load(f)


embedding_files_info = get_embedding_files_info()
need_rebuild = True
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    try:
        saved_meta = load_meta(META_PATH)
        if saved_meta == embedding_files_info:
            # No change, load index
            index = load_index(INDEX_PATH)
            texts, _ = load_embeddings_and_index()  # texts only
            need_rebuild = False
    except Exception:
        pass

if need_rebuild:
    texts, embeddings = load_embeddings_and_index()
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    save_index(index, INDEX_PATH)
    save_meta(embedding_files_info, META_PATH)
    print(f"Rebuilt FAISS index and saved to disk. Loaded {len(texts)} chunks.")
else:
    print(f"Loaded FAISS index from disk. {len(texts)} chunks.")


# -----------------
# Search function
# -----------------
def search(query_embedding, top_k=5):
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    return [(texts[i], float(D[0][j])) for j, i in enumerate(I[0])]


# -----------------
# Embedding model
# -----------------
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
embed_model = SentenceTransformer(MODEL_NAME)

# Conversation history
conversation_history = []

print("\nRAG Chatbot ready! Type 'exit' to quit.")

# -----------------
# Main chatbot loop
# -----------------
while True:
    user_q = input("\nYou: ").strip()
    if user_q.lower() == "exit":
        break

    # Step 1: Embed the question
    q_embedding = embed_model.encode(user_q).astype("float32")

    # Step 2: Retrieve relevant chunks from docs
    results = search(q_embedding, top_k=5)
    context_text = "\n\n".join([r[0] for r in results])

    # Step 3: Append user message to conversation history
    conversation_history.append({"role": "user", "content": user_q})

    # Step 4: Build context + conversation
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant that answers using the provided context "
            "and your memory of this conversation. If the context is not relevant, "
            "you can also use prior messages to infer the answer."
        ),
    }

    # Inject retrieval results into the last user turn
    latest_user_with_context = {
        "role": "user",
        "content": f"Context:\n{context_text}\n\nQuestion: {user_q}",
    }

    # Full message list for Ollama
    messages = [system_message] + conversation_history[:-1] + [latest_user_with_context]

    # Step 5: Stream response from Ollama
    print("\nBot: ", end="", flush=True)
    full_response = ""
    stream = ollama.chat(model="gemma3:1b-it-qat", messages=messages, stream=True)
    for chunk in stream:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        full_response += token
    print()

    # Step 6: Add bot response to conversation history
    conversation_history.append({"role": "assistant", "content": full_response})
