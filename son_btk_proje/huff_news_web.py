

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

print("📥 Model indiriliyor...")

model_name = "Raxus-99/huffpost-model"

# Model dosyalarını ilerleme çubuğu ile indir
model_path = snapshot_download(
    repo_id=model_name,
    local_dir="./huffpost_model_local",
    resume_download=True,
    tqdm_class=tqdm
)

print("✅ İndirme tamamlandı!")
print("📦 Model yükleniyor...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
print("✅ Tokenizer yüklendi")

model = AutoModelForSequenceClassification.from_pretrained(model_path)
print("✅ Model yüklendi")


def predict(headline, description):
    text = f"{headline} {description}".strip()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    top5 = torch.topk(probs, k=5)

    results = {}
    for idx, prob in zip(top5.indices, top5.values):
        results[model.config.id2label[idx.item()]] = float(prob.item())

    return results


print("🚀 Arayüz başlatılıyor...")

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Haber Başlığı", placeholder="Örn: Trump Announces New Policy"),
        gr.Textbox(label="Açıklama (opsiyonel)", placeholder="Kısa açıklama")
    ],
    outputs=gr.Label(label="Tahmin Sonuçları", num_top_classes=5),
    title="📰 Haber Kategori Sınıflandırıcı",
    description="BERT ile eğitilmiş HuffPost haber sınıflandırıcı",
    examples=[
        ["Trump Announces New Policy", ""],
        ["Best Travel Destinations", ""],
    ]
)

demo.launch(share=True)