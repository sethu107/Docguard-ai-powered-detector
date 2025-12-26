import torch
import clip
from pdf2image import convert_from_path
import os
import faiss
import numpy as np
import json
import time
import shutil

# ----------------------------
# Setup CLIP
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Using device: {device}")

# ----------------------------
# File paths
# ----------------------------
emb_file = "embeddings.npy"
idx_file = "file_index.json"
faiss_file = "faiss.index"
database_dir = "pdf_database"
unique_dir = os.path.join(database_dir, "unique")
duplicate_dir = os.path.join(database_dir, "duplicate")

# Create folders if missing
os.makedirs(unique_dir, exist_ok=True)
os.makedirs(duplicate_dir, exist_ok=True)

# ----------------------------
# Load database (or create empty)
# ----------------------------
if os.path.exists(emb_file) and os.path.exists(idx_file) and os.path.exists(faiss_file):
    all_embeddings = np.load(emb_file)
    with open(idx_file, "r") as f:
        file_index = json.load(f)
    index = faiss.read_index(faiss_file)
else:
    all_embeddings = np.empty((0, 512), dtype="float32")
    file_index = []
    index = faiss.IndexFlatIP(512)  # cosine similarity


# ----------------------------
# Helpers
# ----------------------------
def pdf_to_page_embeddings(pdf_path, batch_size=30, dpi=100, num_threads=4):
    pages = convert_from_path(pdf_path, dpi=dpi, thread_count=num_threads)
    tensors = torch.stack([preprocess(p.convert("RGB")) for p in pages])

    batches = [tensors[i:i + batch_size] for i in range(0, len(tensors), batch_size)]

    all_embs = []
    with torch.no_grad():
        for batch in batches:
            emb = model.encode_image(batch.to(device)).cpu().numpy()
            all_embs.append(emb)

    return np.vstack(all_embs).astype("float32")


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def save_database(embeddings, file_index_list, faiss_index):
    np.save(emb_file, embeddings)
    with open(idx_file, "w") as f:
        json.dump(file_index_list, f)
    faiss.write_index(faiss_index, faiss_file)


# ----------------------------
# Core logic for a single PDF
# ----------------------------
def process_pdf(new_pdf: str):
    """
    Process one PDF and return a list of log messages
    (the same style you see in the console) for the UI.
    """
    global all_embeddings, file_index, index

    messages = []

    # 1) Detected file
    msg = f"📄 Detected new file: {new_pdf}"
    print(msg)
    messages.append(msg)

    new_embs = normalize(pdf_to_page_embeddings(new_pdf))
    base_name = os.path.basename(new_pdf)

    # ---------- FIRST PDF EVER ----------
    if len(file_index) == 0:
        ts = time.strftime("%Y%m%d-%H%M%S")
        new_name = f"{os.path.splitext(base_name)[0]}_{ts}.pdf"
        dest = os.path.join(unique_dir, new_name)
        shutil.move(new_pdf, dest)

        file_index.extend([[new_name, page] for page in range(len(new_embs))])
        all_embeddings = np.vstack([all_embeddings, new_embs])
        index.add(new_embs)
        save_database(all_embeddings, file_index, index)

        msg2 = f"✅ Added first PDF to unique → {new_name}"
        print(msg2)
        messages.append(msg2)

        return messages

    # ---------- COMPARE WITH EXISTING ----------
    D, I = index.search(new_embs, k=5)
    arr = []
    for d_row, i_row in zip(D, I):
        top_score = d_row[0]
        top_file = file_index[i_row[0]][0]
        arr.append([top_file, top_score])

    total_elements = len(arr)
    pdf_count = {}
    for pdf_name, score in arr:
        if score >= 0.95:  # similarity threshold
            pdf_count[pdf_name] = pdf_count.get(pdf_name, 0) + 1

    pdf_percentage = {name: (count / total_elements) * 100 for name, count in pdf_count.items()}
    total_percentage = sum(pdf_percentage.values())

    expected_total = 60  # threshold for duplicate decision

    # ---------- DUPLICATE CASE ----------
    if total_percentage >= expected_total:
        for name, percentage in pdf_percentage.items():
            msg_sim = f"⚡ {base_name} is similar to {name} → {percentage:.2f}% match"
            print(msg_sim)
            messages.append(msg_sim)

        similar_pdf = max(pdf_percentage, key=pdf_percentage.get)
        name_no_ext, ext = os.path.splitext(base_name)
        dup_name = f"{name_no_ext}similar-to{similar_pdf}{ext}"
        dest = os.path.join(duplicate_dir, dup_name)
        shutil.move(new_pdf, dest)

        msg3 = f"📂 Stored as duplicate: {dup_name}"
        print(msg3)
        messages.append(msg3)

        return messages

    # ---------- UNIQUE CASE ----------
    ts = time.strftime("%Y%m%d-%H%M%S")
    new_name = f"{os.path.splitext(base_name)[0]}_{ts}.pdf"
    dest = os.path.join(unique_dir, new_name)
    shutil.move(new_pdf, dest)

    file_index.extend([[new_name, page] for page in range(len(new_embs))])
    all_embeddings = np.vstack([all_embeddings, new_embs])
    index.add(new_embs)
    save_database(all_embeddings, file_index, index)

    msg4 = f"✨ {base_name} was unique → added as {new_name}"
    print(msg4)
    messages.append(msg4)

    return messages
