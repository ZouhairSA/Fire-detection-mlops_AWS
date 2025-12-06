from flask import Flask, send_file, request, render_template
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from ultralytics import YOLO
import os, base64

app = Flask(__name__)

# -------------------------------------------------------------------
# LOAD YOLO MODEL
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(BASE_DIR, "../model/best.pt"))

print("MODEL PATH =", model_path)
print("EXISTS =", os.path.exists(model_path))

model = YOLO(model_path)


# -------------------------------------------------------------------
# MAIN ROUTE
# -------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    result_data = None
    image_path = None
    annotated_path = None

    # dossier static
    STATIC_DIR = os.path.join(BASE_DIR, "static")
    os.makedirs(STATIC_DIR, exist_ok=True)

    input_path = os.path.join(STATIC_DIR, "input.jpg")
    annotated_path = os.path.join(STATIC_DIR, "annotated.jpg")

    if request.method == "POST":

        # ----------------------------------------------------------
        # 1️⃣ Vérifier si l'image vient de l'UPLOAD
        # ----------------------------------------------------------
        if "image" in request.files and request.files["image"].filename != "":
            file = request.files["image"]
            file.save(input_path)
            image_path = "static/input.jpg"

        # ----------------------------------------------------------
        # 2️⃣ Vérifier si l'image vient de la CAMÉRA (base64)
        # ----------------------------------------------------------
        else:
            camera_data = request.form.get("camera_image", "")

            if camera_data.startswith("data:image"):
                header, encoded = camera_data.split(",", 1)
                decoded = base64.b64decode(encoded)

                with open(input_path, "wb") as f:
                    f.write(decoded)

                image_path = "static/input.jpg"
            else:
                result_data = [{"error": "Aucune image reçue"}]
                return render_template(
                    "index.html",
                    results=result_data,
                    image_path=None,
                    annotated=None
                )

        # ----------------------------------------------------------
        # YOLO INFERENCE
        # ----------------------------------------------------------
        result = model(input_path, device="cpu")[0]

        # sauvegarder l'image annotée
        result.save(annotated_path)

        # extraire toutes les détections
        detections = []

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "class": result.names[cls],
                "confidence": round(conf, 3)
            })

        result_data = detections

    return render_template(
        "index.html",
        results=result_data,
        image_path="static/input.jpg" if os.path.exists(input_path) else None,
        annotated="static/annotated.jpg" if os.path.exists(annotated_path) else None
    )


# -------------------------------------------------------------------
# PDF GENERATION
# -------------------------------------------------------------------
@app.route("/download_pdf")
def download_pdf():

    STATIC_DIR = os.path.join(BASE_DIR, "static")
    os.makedirs(STATIC_DIR, exist_ok=True)

    pdf_path = os.path.join(STATIC_DIR, "result.pdf")

    results = request.args.get("results", "")
    image = request.args.get("image", "input.jpg")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c = canvas.Canvas(pdf_path, pagesize=A4)

    c.drawString(50, 800, "Rapport FireVision")
    c.drawString(50, 770, "Date : " + now)
    c.drawString(50, 750, "Image : " + image)

    c.drawString(50, 720, "Résultats :")
    c.drawString(50, 700, results)

    c.save()

    return send_file(pdf_path, as_attachment=True)


# -------------------------------------------------------------------
# RUN SERVER
# -------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)

# -------------------------------------------------------------------------------------------------------------------

# from flask import Flask, request, render_template
# from ultralytics import YOLO
# import os

# app = Flask(__name__)

# # -------------------------------------------------------------------
# # Chemin absolu du modèle (fonctionne partout : PC, Docker, EC2)
# # -------------------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.abspath(os.path.join(BASE_DIR, "../model/best.pt"))

# print("MODEL PATH =", model_path)
# print("EXISTS =", os.path.exists(model_path))

# # CORRIGÉ : plus de device= dans le constructeur ! (Ultralytics 8.2+)
# model = YOLO(model_path)


# # -------------------------------------------------------------------
# # Route principale
# # -------------------------------------------------------------------
# @app.route("/", methods=["GET", "POST"])
# def index():
#     result_data = None
#     image_path = None

#     if request.method == "POST":
#         if "image" not in request.files:
#             result_data = {"error": "No image uploaded"}
#         else:
#             file = request.files["image"]
#             # Chemin absolu dans le conteneur
#             image_path = "/app/static/input.jpg"
#             file.save(image_path)

#             # CORRIGÉ : device="cpu" passé ici, à l'inférence
#             result = model(image_path, device="cpu")[0]

#             detections = []
#             for box in result.boxes:
#                 cls = int(box.cls[0])
#                 conf = float(box.conf[0])
#                 detections.append({
#                     "class": result.names[cls],
#                     "confidence": round(conf, 3)
#                 })

#             result_data = detections

#     return render_template(
#         "index.html",
#         results=result_data,
#         image_path="static/input.jpg"  # Flask sert automatiquement /static/
#     )


# # -------------------------------------------------------------------
# # Lancer l'application
# # -------------------------------------------------------------------
# if __name__ == "__main__":
#     os.makedirs("/app/static", exist_ok=True)
#     app.run(host="0.0.0.0", port=5000, debug=False)
 
# -------------------------------------------------------------------------------------------------------------------
# from flask import Flask, request, render_template
# from ultralytics import YOLO
# import os

# app = Flask(__name__)

# # Charger le modèle YOLO
# model_path = os.path.join("..", "model", "best.pt")
# model = YOLO(model_path)

# # Page d'accueil avec upload
# @app.route("/", methods=["GET", "POST"])
# def index():
#     result_data = None
#     image_path = None

#     if request.method == "POST":
#         if 'image' not in request.files:
#             result_data = {"error": "No image uploaded"}
#         else:
#             file = request.files['image']
#             image_path = "static/input.jpg"
#             file.save(image_path)

#             result = model(image_path)[0]

#             detections = []
#             for box in result.boxes:
#                 cls = int(box.cls[0])
#                 conf = float(box.conf[0])
#                 detections.append({
#                     "class": result.names[cls],
#                     "confidence": round(conf, 3)
#                 })

#             result_data = detections

#     return render_template("index.html", results=result_data, image_path=image_path)

# if __name__ == "__main__":
#     # Crée le dossier static si non existant
#     os.makedirs("static", exist_ok=True)
#     app.run(host="0.0.0.0", port=5000, debug=True)
