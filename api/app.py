from flask import Flask, send_file, request, render_template
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from ultralytics import YOLO
import os

app = Flask(__name__)

# ----------------------------------------------------------
# Chargement du modèle
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(BASE_DIR, "../model/best.pt"))

print("MODEL PATH =", model_path)
print("EXISTS =", os.path.exists(model_path))

model = YOLO(model_path)

# ----------------------------------------------------------
# ROUTE PRINCIPALE
# ----------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    result_data = None
    image_path = None
    annotated = None

    if request.method == "POST":

        if "image" not in request.files:
            result_data = [{"error": "No image uploaded"}]
        else:

            file = request.files["image"]

            # chemins
            image_path = "/app/static/input.jpg"
            annotated = "/app/static/annotated.jpg"

            os.makedirs("/app/static", exist_ok=True)

            # sauvegarde image
            file.save(image_path)

            # YOLO inference
            result = model(image_path, device="cpu")[0]

            # sauvegarder image annotée
            result.save(annotated)

            # collecte résultats
            detections = []

            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                detections.append({
                    "class": result.names[cls],
                    "confidence": round(conf,3)
                })

            result_data = detections

    return render_template(
        "index.html",
        results=result_data,
        image_path="static/input.jpg",
        annotated="static/annotated.jpg"
    )


# ----------------------------------------------------------
# PDF
# ----------------------------------------------------------
@app.route("/download_pdf")
def download_pdf():

    pdf = "/app/static/result.pdf"

    results = request.args.get("results","")
    image = request.args.get("image","input.jpg")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c = canvas.Canvas(pdf, pagesize=A4)

    c.drawString(50,800,"Rapport FireVision")
    c.drawString(50,770,"Date : "+now)
    c.drawString(50,750,"Image : "+image)

    c.drawString(50,720,"Résultat : ")
    c.drawString(50,700,results)

    c.save()
    return send_file(pdf, as_attachment=True)


# ----------------------------------------------------------
# RUN
# ----------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("/app/static", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=False)

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
