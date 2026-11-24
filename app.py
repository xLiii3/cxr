"""Gradio demo: classify a chest X-ray, search by text, search by image.

    python app.py   ->  http://127.0.0.1:7860
"""
import gradio as gr

from src import config as C
from src.service import Engine

ENGINE = Engine()
EXAMPLES = [[str(p)] for p in sorted((C.ROOT / "examples").glob("*.png"))]


def classify(img):
    if img is None:
        raise gr.Error("请先上传一张胸片")
    probs = ENGINE.predict(img)
    labels = {f"{d['zh']} · {d['class']}": d["prob"] for d in probs}
    rows = [[d["zh"], d["class"], round(d["prob"], 4)] for d in probs]
    heat, name, p = ENGINE.explain(img)
    return labels, rows, heat, f"Grad-CAM: {C.CLASS_ZH[name]} ({p:.3f})"


def by_text(text, k):
    if not text.strip():
        raise gr.Error("请输入英文或中文病变名，例如 cardiomegaly / 心脏扩大")
    return [(r["image"], f"{r['score']:.3f}  {'、'.join(r['labels'])}")
            for r in ENGINE.search_text(text, int(k))]


def by_image(img, k):
    if img is None:
        raise gr.Error("请先上传一张胸片")
    return [(r["image"], f"{r['score']:.3f}  {'、'.join(r['labels'])}")
            for r in ENGINE.search_image(img, int(k))]


with gr.Blocks(title="胸片多模态诊断与检索") as demo:
    gr.Markdown(
        "# 胸片多模态诊断与检索\n"
        "BiomedCLIP（ViT-B/16 + PubMedBERT）在 NIH ChestX-ray14 上做 LoRA 微调；"
        "左侧为 14 类病变概率与 Grad-CAM，右侧支持文本/图像检索相似胸片。"
    )
    with gr.Tabs():
        with gr.Tab("病变识别"):
            with gr.Row():
                image_in = gr.Image(type="pil", label="胸片输入", height=320)
                heat_out = gr.Image(type="pil", label="Grad-CAM 热力图", height=320)
            run = gr.Button("分析", variant="primary")
            note = gr.Markdown()
            prob = gr.Label(num_top_classes=6, label="Top-6 概率")
            table = gr.Dataframe(
                headers=["病变", "英文", "概率"], label="14 类完整输出",
                interactive=False, wrap=True,
            )
            run.click(classify, image_in, [prob, table, heat_out, note])
            gr.Examples(EXAMPLES, inputs=image_in, label="示例胸片")

        with gr.Tab("文本检索"):
            with gr.Row():
                query = gr.Textbox(label="文本查询", placeholder="cardiomegaly / pleural effusion / 心脏扩大")
                k1 = gr.Slider(1, 10, value=5, step=1, label="返回数量")
            search_btn = gr.Button("检索相似胸片", variant="primary")
            gallery_text = gr.Gallery(label="Top-K 相似胸片", columns=5, height=380)
            search_btn.click(by_text, [query, k1], gallery_text)
            query.submit(by_text, [query, k1], gallery_text)

        with gr.Tab("图像检索"):
            with gr.Row():
                image_q = gr.Image(type="pil", label="查询胸片", height=280)
                k2 = gr.Slider(1, 10, value=5, step=1, label="返回数量")
            search_btn2 = gr.Button("检索相似胸片", variant="primary")
            gallery_image = gr.Gallery(label="Top-K 相似胸片", columns=5, height=380)
            search_btn2.click(by_image, [image_q, k2], gallery_image)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
