"""Inference engine shared by the Gradio app and the HTTP API."""
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import cm
from PIL import Image

from . import config as C
from .data import to_tensor
from .model import embed_texts, freeze_all, inject_lora, load_clip, load_lora


class Engine:
    """Loads the fine-tuned encoder, the classifier head and the FAISS gallery once."""

    def __init__(self, device=None, use_lora=True, k=5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k
        self.model, _, self.tokenizer = load_clip(self.device)
        freeze_all(self.model)
        inject_lora(self.model)
        self.model.head = nn.Linear(self.model.embed_dim, len(C.CLASSES)).to(self.device)
        if use_lora and (C.OUT / "lora.pt").exists():
            load_lora(self.model, C.OUT / "lora.pt")
        self.model.eval()
        self.gallery = pd.read_csv(C.OUT / "gallery.csv")
        self.emb = np.load(C.OUT / "gallery_emb.npy").astype("float32")
        self.labels = np.load(C.OUT / "gallery_y.npy")
        self.index = faiss.read_index(str(C.OUT / "index_flat.faiss"))

    # -- encoders -------------------------------------------------------
    def embed_image(self, image):
        x = to_tensor(image).to(self.device)
        with torch.no_grad():
            f = self.model.encode_image(x).float().cpu().numpy()[0]
        return f / np.linalg.norm(f)

    def embed_text(self, text):
        # allow Chinese class names as queries, e.g. "心脏扩大"
        for cls, zh in C.CLASS_ZH.items():
            text = text.replace(zh, cls)
        f = embed_texts(self.model, self.tokenizer, [text], self.device)[0].numpy()
        return f / np.linalg.norm(f)

    # -- tasks ----------------------------------------------------------
    def predict(self, image):
        x = to_tensor(image).to(self.device)
        with torch.no_grad():
            p = torch.sigmoid(self.model.head(self.model.encode_image(x)))[0].cpu().numpy()
        order = np.argsort(-p)
        return [{"class": C.CLASSES[i], "zh": C.CLASS_ZH[C.CLASSES[i]], "prob": float(p[i])}
                for i in order]

    def search_image(self, image, k=None):
        return self._search(self.embed_image(image), k)

    def search_text(self, text, k=None):
        return self._search(self.embed_text(text), k)

    def _search(self, vec, k):
        k = int(k or self.k)
        scores, ids = self.index.search(vec[None].astype("float32"), k)
        out = []
        for s, i in zip(scores[0], ids[0]):
            names = [C.CLASS_ZH[c] for c, v in zip(C.CLASSES, self.labels[i]) if v]
            out.append({"image": self.open(self.gallery.image[i]), "score": float(s),
                        "labels": names or ["正常"], "file": self.gallery.image[i]})
        return out

    def open(self, name):
        return Image.open(C.IMAGES / name).convert("L")

    def explain(self, image, cls=None):
        """Grad-CAM heatmap blended on the input; cls defaults to the top class."""
        from .explain import gradcam  # local import keeps app start-up light

        x = to_tensor(image).to(self.device)
        with torch.no_grad():
            p = torch.sigmoid(self.model.head(self.model.encode_image(x)))[0]
        cls = int(p.argmax()) if cls is None else (C.CLASSES.index(cls) if isinstance(cls, str) else int(cls))
        cam = gradcam(self.model, x, cls)
        base = np.asarray(image.convert("L").resize((C.IMG_SIZE, C.IMG_SIZE)), dtype="float32") / 255.0
        a = (0.75 * np.clip(cam, 0, 1) ** 0.7)[..., None]
        heat = (base[..., None] * 255 * (1 - a) + cm.jet(cam)[..., :3] * 255 * a).clip(0, 255).astype("uint8")
        return Image.fromarray(heat), C.CLASSES[cls], float(p[cls])
