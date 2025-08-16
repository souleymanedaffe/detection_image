import os
import shutil
import requests
from pycocotools.coco import COCO
from tqdm import tqdm

# Fichier d'annotations COCO
ANNOTATIONS_FILE = "annotations/instances_val2017.json"
OUTPUT_DIR = "coco_20_images"

# Classes demandées (COCO ID)
TARGET_CLASSES = {
    "person": 1,
    "backpack": 24,
    "car": 3,
    "dog": 18,
    "cat": 17,
    "elephant": 22,
    "giraffe": 25
}

# Charger les annotations
coco = COCO(ANNOTATIONS_FILE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_name, class_id in TARGET_CLASSES.items():
    print(f"\n📥 Téléchargement de la classe : {class_name}")
    img_ids = coco.getImgIds(catIds=[class_id])
    selected_ids = img_ids[:20]  # seulement 20 images
    
    class_folder = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_folder, exist_ok=True)

    for img_id in tqdm(selected_ids):
        img_info = coco.loadImgs(img_id)[0]
        url = img_info['coco_url']
        file_name = img_info['file_name']
        out_path = os.path.join(class_folder, file_name)

        if not os.path.exists(out_path):
            r = requests.get(url, stream=True)
            with open(out_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

print("\n✅ Téléchargement terminé ! Les images sont dans le dossier :", OUTPUT_DIR)

shutil.make_archive("coco_20_images", 'zip', OUTPUT_DIR)
print("📦 Archive créée : coco_20_images.zip")
