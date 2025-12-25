import torch
import clip
from pdf2image import convert_from_path
import os
import faiss
import numpy as np
import json
import time
import shutil
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flask import Flask, redirect, url_for, render_template, request

# ----------------------------
# Setup CLIP
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Using device: {device}")

# ----------------------------
# Helper functions
# ----------------------------
def pdf_to_page_embeddings(pdf_path, batch_size=30, dpi=100, num_threads=4):
    pages = convert_from_path(pdf_path, dpi=dpi, thread_count=num_threads)
    tensors = torch.stack([preprocess(p.convert("RGB")) for p in pages])

    batches = [tensors[i:i+batch_size] for i in range(0, len(tensors), batch_size)]

    all_embs = []
    with torch.no_grad():
        for batch in batches:
            emb = model.encode_image(batch.to(device)).cpu().numpy()
            all_embs.append(emb)

    return np.vstack(all_embs).astype("float32")

def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def save_database(embeddings, file_index, index):
    np.save(emb_file, embeddings)
    with open(idx_file, "w") as f:
        json.dump(file_index, f)
    faiss.write_index(index, faiss_file)

# ----------------------------
# File paths
# ----------------------------
emb_file = "embeddings.npy"
idx_file = "file_index.json"
faiss_file = "faiss.index"
database_dir = "pdf_database"
unique_dir = os.path.join(database_dir, "unique")
duplicate_dir = os.path.join(database_dir, "duplicate")
watch_dir = "watch_folder"

# Create folders if missing
os.makedirs(unique_dir, exist_ok=True)
os.makedirs(duplicate_dir, exist_ok=True)
os.makedirs(watch_dir, exist_ok=True)

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
# Core logic for handling PDFs
# ----------------------------
results_array = []  # Store results for frontend rendering

def process_pdf(new_pdf):
    global all_embeddings, file_index, index, results_array

    print(f"\n📄 Detected new file: {new_pdf}")
    new_embs = normalize(pdf_to_page_embeddings(new_pdf))
    base_name = os.path.basename(new_pdf)

    if len(file_index) == 0:
        # First PDF ever → just add
        ts = time.strftime("%Y%m%d-%H%M%S")
        new_name = f"{os.path.splitext(base_name)[0]}_{ts}.pdf"
        dest = os.path.join(unique_dir, new_name)
        shutil.move(new_pdf, dest)

        file_index.extend([[new_name, page] for page in range(len(new_embs))])
        all_embeddings = np.vstack([all_embeddings, new_embs])
        index.add(new_embs)
        save_database(all_embeddings, file_index, index)
        results_array.append(f"Added unique: {new_name}")
        print(f"✅ Added first PDF to unique → {new_name}")

    else:
        # Search similarity
        D, I = index.search(new_embs, k=5)
        arr = []
        for d_row, i_row in zip(D, I):
            top_score = d_row[0]
            top_file = file_index[i_row[0]][0]
            arr.append([top_file, top_score])

        # Percentage calculation
        total_elements = len(arr)
        pdf_count = {}
        for pdf_name, score in arr:
            if score >= 0.95:  # similarity threshold
                pdf_count[pdf_name] = pdf_count.get(pdf_name, 0) + 1

        pdf_percentage = {name: (count / total_elements) * 100 for name, count in pdf_count.items()}
        total_percentage = sum(pdf_percentage.values())

        expected_total = 60  # threshold for duplicate decision
        if total_percentage >= expected_total:
            # Duplicate → rename and move
            for name, percentage in pdf_percentage.items():
                print(f"⚡ {base_name} is similar to {name} → {percentage:.2f}% match")

            similar_pdf = max(pdf_percentage, key=pdf_percentage.get)
            name_no_ext, ext = os.path.splitext(base_name)
            dup_name = f"{name_no_ext}similar-to{similar_pdf}{ext}"
            dest = os.path.join(duplicate_dir, dup_name)
            shutil.move(new_pdf, dest)
            results_array.append(f"Duplicate: {dup_name} (similar to {similar_pdf})")
            print(f"📂 Stored as duplicate: {dup_name}")

        else:
            # Unique → add to DB
            ts = time.strftime("%Y%m%d-%H%M%S")
            new_name = f"{os.path.splitext(base_name)[0]}_{ts}.pdf"
            dest = os.path.join(unique_dir, new_name)
            shutil.move(new_pdf, dest)

            file_index.extend([[new_name, page] for page in range(len(new_embs))])
            all_embeddings = np.vstack([all_embeddings, new_embs])
            index.add(new_embs)
            save_database(all_embeddings, file_index, index)
            results_array.append(f"Added unique: {new_name}")
            print(f"✨ {base_name} was unique → added as {new_name}")

# ----------------------------
# Watchdog handler
# ----------------------------
class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".pdf"):
            process_pdf(event.src_path)

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = watch_dir

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/post", methods=["POST"])
def post():
    global results_array
    results_array.clear()  # Clear previous results
    pdfs = request.files.getlist('pdf')
    for pdf in pdfs:
        if pdf and pdf.filename.endswith('.pdf'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
            pdf.save(filepath)
            process_pdf(filepath)  # Process synchronously for instant results
    return render_template('upload.html', results=results_array)

@app.route("/uploaded")
def upload():
    global results_array
    return render_template('upload.html', results=results_array)

# ----------------------------
# Run Both Flask + Watchdog
# ----------------------------
def start_watcher():
    observer = Observer()
    handler = PDFHandler()
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()
    print(f"👀 Watching folder: {watch_dir} (drop PDFs here or upload via Flask)...")
    try:
        while True:
            time.sleep(3)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # Start Watchdog in background thread
    t = threading.Thread(target=start_watcher, daemon=True)
    t.start()

    # Start Flask
    app.run(debug=True, use_reloader=False)
