from flask import Flask, render_template, request
import os

from core_logic import process_pdf

app = Flask(__name__)

# Separate folder for files uploaded via the web
WEB_UPLOAD_DIR = "web_uploads"
os.makedirs(WEB_UPLOAD_DIR, exist_ok=True)

app.config["UPLOAD_FOLDER"] = WEB_UPLOAD_DIR

results_array = []


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", results=results_array)


@app.route("/post", methods=["POST"])
def post():
    global results_array
    results_array.clear()

    pdfs = request.files.getlist("pdf")

    if not pdfs:
        results_array.append("No files were uploaded.")
    else:
        for pdf in pdfs:
            filename = pdf.filename
            if pdf and filename.lower().endswith(".pdf"):
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                pdf.save(filepath)

                # process the PDF and get list of messages
                msg_list = process_pdf(filepath)

                # extend results_array with all log lines
                if isinstance(msg_list, list):
                    results_array.extend(msg_list)
                else:
                    results_array.append(str(msg_list))
            else:
                if filename:
                    results_array.append(f"Skipped non-PDF file: {filename}")

    return render_template("index.html", results=results_array)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
<<<<<<< HEAD
#sriram__commit
=======
>>>>>>> ce249d939bbc16f0d14f211dc34e8350a095aaa3
