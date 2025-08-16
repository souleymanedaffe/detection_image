from flask import Flask, render_template, request, send_from_directory, send_file
import os
from PIL import Image
from ultralytics import YOLO
import shutil
import streamlit as st

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
EXP_FOLDER = os.path.join(RESULT_FOLDER, 'exp')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# =========================================
# AUTO-TÉLÉCHARGEMENT DES IMAGES DE DÉMO
# =========================================

@app.before_first_request
def download_demo_images_if_needed():
    dataset_folder = "coco_20_images"
    script_path = "extract_coco_images.py"
    if not os.path.exists(dataset_folder):
        print("📦 Dossier 'coco_20_images' manquant. Téléchargement en cours...")
        os.system(f"python {script_path}")
    else:
        print("✅ Dossier 'coco_20_images' déjà présent. Aucun téléchargement nécessaire.")


model = YOLO('yolov5s.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    files = request.files.getlist('images')
    object_filter = request.form.get("filter")
    filter_label = object_filter if object_filter else None
    results_files = []
    summary = {}

    if os.path.exists(EXP_FOLDER):
        shutil.rmtree(EXP_FOLDER)
    os.makedirs(EXP_FOLDER, exist_ok=True)

    for file in files:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        results = model(filepath)
        result = results[0]

        names = result.names
        classes = result.boxes.cls.tolist() if result.boxes.cls is not None else []
        detected_labels = [names[int(cls)] for cls in classes]
        

                # Dictionnaire de traduction COCO → français
        coco_fr = {
            "person": "personne",
            "backpack": "sac à dos",
            "car": "voiture",
            "dog": "chien",
            "cat": "chat",
            "bottle": "bouteille",
            "sink": "évier",
            "tv": "télévision",
            "keyboard": "clavier",
            # ajoute d'autres objets ici si besoin
        }

        # Traduction des objets détectés
        translated_labels = []
        for label in detected_labels:
            translated = coco_fr.get(label, label)  # fallback si pas traduit
            translated_labels.append(translated)

        # Sauvegarde des objets traduits dans le résumé
        summary[filename] = translated_labels




        if object_filter:
            filtered = [label for label in detected_labels if label == object_filter]
            if filtered:
                summary[filename] = filtered
                result.save(filename=os.path.join(EXP_FOLDER, filename))
                results_files.append(filename)
        else:
            summary[filename] = detected_labels
            result.save(filename=os.path.join(EXP_FOLDER, filename))
            results_files.append(filename)

    return render_template('results.html', files=results_files, filter_label=filter_label, summary=summary)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(EXP_FOLDER, filename, as_attachment=True)

@app.route('/download-all')
def download_all():
    zip_path = os.path.join(RESULT_FOLDER, 'all_results.zip')
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', EXP_FOLDER)
    return send_file(zip_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)


# =========================================
# FOOTER
# =========================================
