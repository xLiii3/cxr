"""Paths, label set, prompts and hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
IMAGES = RAW / "images"
PROC = DATA / "processed"
GALLERY = DATA / "gallery" / "thumbs"
OUT = ROOT / "outputs"
CLIP_DIR = ROOT / "models" / "BiomedCLIP"

LABELS_CSV = RAW / "Data_Entry_2017.csv"
SPLIT_CSV = PROC / "labels.csv"
IMAGE_NPY = PROC / "images_224.npy"
EMB_NPZ = PROC / "embeddings.npz"

CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
CLASS_ZH = {
    "Atelectasis": "肺不张", "Cardiomegaly": "心脏扩大", "Effusion": "胸腔积液",
    "Infiltration": "浸润", "Mass": "肿块", "Nodule": "结节", "Pneumonia": "肺炎",
    "Pneumothorax": "气胸", "Consolidation": "实变", "Edema": "水肿",
    "Emphysema": "肺气肿", "Fibrosis": "纤维化", "Pleural_Thickening": "胸膜增厚",
    "Hernia": "膈疝",
}

# Three prompt paraphrases per class; the mean embedding is used.
PROMPT_TEMPLATES = [
    "{}",
    "a chest x-ray showing {}",
    "{} is visible in this chest radiograph",
]
NORMAL_PROMPT = "a normal chest x-ray"

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
IMG_SIZE = 224
SEED = 42
SPLITS = (0.7, 0.1, 0.2)


def prompt_list():
    """14 x 3 prompts, flattened row-major by class."""
    return [[t.format(c.replace("_", " ")) for t in PROMPT_TEMPLATES] for c in CLASSES]
