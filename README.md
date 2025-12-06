# DocGuard – AI-Powered PDF Similarity & Duplicate Detector

DocGuard is a mini project that **watches a folder for incoming PDF files**, compares them using **AI-based document embeddings**, and automatically separates **unique** and **near-duplicate** documents.

It is useful for:
- Cleaning datasets with many similar/duplicate PDFs  
- Organizing downloaded documents  
- Pre-processing PDFs before feeding them into RAG / LLM pipelines  

---

## ✨ Features

- 📂 **Real-time folder watching** – drop PDFs into a `watch_folder`, and the script picks them up automatically.
- 🧠 **AI-powered similarity** – uses embeddings to compare new PDFs with existing ones.
- 🧬 **Duplicate detection** – detects near-duplicate PDFs based on a similarity threshold.
- 🗂 **Unique document store** – unique PDFs are saved into a `unique` folder with timestamped names.
- ⚡ **GPU acceleration (CUDA)** – runs on GPU if available, falls back to CPU if not.
- 🧾 **Console logs** – clear log messages for each detected file and its status.

---

## 🏗 Project Structure

Example structure (your actual layout may vary):

```bash
Mini Project/
├─ version1.0.2.py         # Main script (DocGuard watcher)
├─ requirements.txt        # Python dependencies
├─ mini_env/               # Virtual environment (NOT uploaded to GitHub)
├─ watch_folder/           # Drop PDF files here
├─ unique/                 # Unique/accepted PDFs are saved here
└─ README.md               # This file
```
