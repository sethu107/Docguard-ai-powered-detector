import torch
import clip
from pdf2image import convert_from_path
import os
import faiss
import numpy as np
import json
import gc

# ----------------------------
# Device Setup
# ----------------------------
device = "cpu"

print(f"Using device: {device}")

# ----------------------------
# Lazy Load CLIP Model
# ----------------------------
model = None
preprocess = None


def get_model():
    global model, preprocess

    if model is None:
        # Smaller model for Render free tier
        model, preprocess = clip.load("RN50", device=device)

    return model, preprocess


# ----------------------------
# Database files
# ----------------------------
emb_file = "embeddings.npy"
idx_file = "file_index.json"
faiss_file = "faiss.index"

# ----------------------------
# Load Database
# ----------------------------
if (
    os.path.exists(emb_file)
    and os.path.exists(idx_file)
    and os.path.exists(faiss_file)
):
    all_embeddings = np.load(emb_file)

    with open(idx_file, "r") as f:
        file_index = json.load(f)

    index = faiss.read_index(faiss_file)

else:
    all_embeddings = np.empty((0, 1024), dtype="float16")

    file_index = []

    # RN50 output size = 1024
    index = faiss.IndexFlatIP(1024)

# ----------------------------
# Helper Functions
# ----------------------------
def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def save_database():
    np.save(emb_file, all_embeddings)

    with open(idx_file, "w") as f:
        json.dump(file_index, f)

    faiss.write_index(index, faiss_file)

# ----------------------------
# Memory Optimized PDF Embedding
# ----------------------------
def pdf_to_embedding(pdf_path, dpi=40):

    model, preprocess = get_model()

    pages = convert_from_path(pdf_path, dpi=dpi)

    embeddings = []

    with torch.no_grad():

        for page in pages:

            img = preprocess(page.convert("RGB")).unsqueeze(0).to(device)

            emb = model.encode_image(img).cpu().numpy()[0]

            embeddings.append(emb)

            # Free memory
            del img
            del page

            gc.collect()

    # Average embedding
    final_embedding = np.mean(embeddings, axis=0)

    final_embedding = np.array([final_embedding], dtype="float32")

    return normalize(final_embedding)

# ----------------------------
# Core Logic
# ----------------------------
def process_pdf(new_pdf):

    global all_embeddings, file_index, index

    messages = []

    base_name = os.path.basename(new_pdf)

    msg = f"📄 Processing file: {base_name}"

    print(msg)
    messages.append(msg)

    # Generate embedding
    new_emb = pdf_to_embedding(new_pdf)

    # ----------------------------
    # First PDF
    # ----------------------------
    if len(file_index) == 0:

        file_index.append(base_name)

        all_embeddings = np.vstack(
            [all_embeddings, new_emb.astype("float16")]
        )

        index.add(new_emb.astype("float32"))

        save_database()

        msg2 = f"✅ First PDF added to database"

        print(msg2)
        messages.append(msg2)

        return messages

    # ----------------------------
    # Similarity Search
    # ----------------------------
    D, I = index.search(new_emb.astype("float32"), k=1)

    similarity = float(D[0][0])

    matched_file = file_index[I[0][0]]

    similarity_percent = similarity * 100

    # ----------------------------
    # Duplicate
    # ----------------------------
    if similarity >= 0.90:

        msg3 = (
            f"⚡ Duplicate Detected\n"
            f"Similar to: {matched_file}\n"
            f"Match: {similarity_percent:.2f}%"
        )

        print(msg3)
        messages.append(msg3)

        return messages

    # ----------------------------
    # Unique PDF
    # ----------------------------
    file_index.append(base_name)

    all_embeddings = np.vstack(
        [all_embeddings, new_emb.astype("float16")]
    )

    index.add(new_emb.astype("float32"))

    save_database()

    msg4 = (
        f"✨ Unique PDF\n"
        f"Added to database successfully"
    )

    print(msg4)
    messages.append(msg4)

    return messages