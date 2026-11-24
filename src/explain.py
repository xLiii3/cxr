"""Stage 6: Grad-CAM on the last ViT block -> assets/gradcam.png.

    python -m src.explain
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from . import config as C  # noqa: E402
from .data import read_split, to_tensor  # noqa: E402
from .model import blocks, freeze_all, inject_lora, load_clip, load_lora  # noqa: E402

CASES = ["Cardiomegaly", "Effusion", "Nodule", "Atelectasis"]


def _blur(cam, k=5):
    x = torch.linspace(-2, 2, k, device=cam.device)
    g = torch.exp(-x ** 2 / 2)
    g = g / g.sum()
    g = (g[:, None] * g[None, :])[None, None]
    return F.conv2d(cam[None, None], g, padding=k // 2)[0, 0]


def gradcam(model, x, cls):
    """Return a 224x224 heatmap for class index cls and one image."""
    store = {}

    def hook(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    x = x.detach().clone().requires_grad_(True)
    h = blocks(model)[-1].register_forward_hook(hook)
    logits = model.head(model.encode_image(x))
    model.zero_grad(set_to_none=True)
    logits[0, cls].backward()
    h.remove()

    act, grad = store["act"][0].detach(), store["grad"][0].detach()
    cam = F.relu((act * grad.mean(dim=0)).sum(-1)[1:]).reshape(14, 14)
    cam = _blur(cam / (cam.max() + 1e-8))
    return F.interpolate(cam[None, None], size=(C.IMG_SIZE, C.IMG_SIZE),
                         mode="bicubic", align_corners=False)[0, 0].clamp(0, 1).cpu().numpy()


def denorm(x):
    mean = torch.tensor(C.MEAN).view(3, 1, 1)
    std = torch.tensor(C.STD).view(3, 1, 1)
    return ((x * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()


def test_probs(model, df, device, batch=128):
    """Sigmoid scores for the whole test split."""
    from .data import loader
    out = []
    with torch.no_grad():
        for x, _ in loader(df, batch):
            out.append(torch.sigmoid(model.head(model.encode_image(x.to(device)))).float().cpu())
    return torch.cat(out).numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = load_clip(device)
    freeze_all(model)
    inject_lora(model)
    model.head = nn.Linear(model.embed_dim, len(C.CLASSES)).to(device)
    load_lora(model, C.OUT / "lora.pt")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    df = read_split()
    test = df[df.split == "test"].reset_index(drop=True)
    probs = test_probs(model, test, device)

    fig, axes = plt.subplots(2, len(CASES), figsize=(12, 5.8), dpi=160)
    for i, name in enumerate(CASES):
        cls = C.CLASSES.index(name)
        cand = np.where(test[name].to_numpy() == 1)[0]
        pick = cand[probs[cand, cls].argmax()]
        from PIL import Image
        img = Image.open(C.IMAGES / test["image"][pick]).convert("RGB")
        x = to_tensor(img).to(device).requires_grad_(True)
        with torch.enable_grad():
            cam = gradcam(model, x, cls)
        base = denorm(x[0].detach().cpu())
        axes[0][i].imshow(base)
        axes[0][i].set_title(f"{name}\np={probs[pick, cls]:.2f}", fontsize=8)
        axes[1][i].imshow(base)
        axes[1][i].imshow(cam, cmap="jet", alpha=0.5)
        axes[0][i].axis("off")
        axes[1][i].axis("off")
    axes[0][0].set_ylabel("input", fontsize=8)
    axes[1][0].set_ylabel("Grad-CAM", fontsize=8)
    fig.suptitle("Grad-CAM on the last ViT block: where does the model look?", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.ROOT / "assets" / "gradcam.png")
    print("saved assets/gradcam.png")


if __name__ == "__main__":
    main()
