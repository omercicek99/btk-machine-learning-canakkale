# 1. Kurulum
#!pip install transformers datasets accelerate matplotlib seaborn - q

import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import defaultdict

# 2. Veri Setini Yükleyin
print("📂 Veri yükleniyor...")

data = []
with open('/kaggle/input/data-set/News_Category_Dataset_v3.json', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

print(f"✅ Toplam {len(df)} haber yüklendi")

# 3. Veri İnceleme ve Hazırlık
df['text'] = df['headline'] + " " + df['short_description']
df = df[['text', 'category']]
df = df.rename(columns={'category': 'label'})
df = df.dropna()

print(f"\n✅ {len(df)} temiz örnek hazır")

# 4. Label Encoding
label2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
id2label = {idx: label for label, idx in label2id.items()}

print(f"\n🏷️  {len(label2id)} kategori")

df['label'] = df['label'].map(label2id)

# 5. Train/Test Split
train_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df['label']
)

print(f"\n📚 Train: {len(train_df)} | Test: {len(test_df)}")

# 6. Dataset Oluşturma
train_dataset = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df[['text', 'label']].reset_index(drop=True))

# 7. Tokenization
print("\n🔤 Tokenization başlıyor...")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )


train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

print("✅ Tokenization tamamlandı")

# 8. Model Yükleme
print("\n🤖 Model yükleniyor...")
num_labels = len(label2id)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

print(f"✅ Model hazır - {num_labels} kategorili sınıflandırma")


# 9. Eğitim Geçmişi için Callback
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.metrics = defaultdict(list)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                self.metrics[key].append(value)


metrics_callback = MetricsCallback()


# 10. Metrik Hesaplama
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1
    }


# 11. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    logging_steps=500,
    logging_dir='./logs',
    report_to='none',
    fp16=True,
)

# 12. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[metrics_callback]
)

# 13. Eğitim
print("\n🚀 Eğitim başlıyor...\n")
print("=" * 60)

trainer.train()

# 14. Model Kaydetme
model.save_pretrained('./huffpost_news_classifier')
tokenizer.save_pretrained('./huffpost_news_classifier')

with open('./huffpost_news_classifier/label_mapping.json', 'w') as f:
    json.dump({'label2id': label2id, 'id2label': {int(k): v for k, v in id2label.items()}}, f)

print("\n✅ Model kaydedildi")

# ============================================
# RAPOR OLUŞTURMA
# ============================================

print("\n" + "=" * 60)
print("📊 DETAYLI PERFORMANS RAPORU OLUŞTURULUYOR")
print("=" * 60)

# 1. Eğitim Geçmişi Grafiği
print("\n📈 1. Eğitim grafiği oluşturuluyor...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss grafiği
train_loss = [log for log in trainer.state.log_history if 'loss' in log]
eval_loss = [log for log in trainer.state.log_history if 'eval_loss' in log]

axes[0].plot([log['epoch'] for log in train_loss], [log['loss'] for log in train_loss], label='Train Loss', marker='o')
axes[0].plot([log['epoch'] for log in eval_loss], [log['eval_loss'] for log in eval_loss], label='Validation Loss',
             marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy grafiği
eval_acc = [log for log in trainer.state.log_history if 'eval_accuracy' in log]
axes[1].plot([log['epoch'] for log in eval_acc], [log['eval_accuracy'] for log in eval_acc],
             label='Validation Accuracy', marker='s', color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Validation Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✅ Grafik kaydedildi: training_history.png")

# 2. Test Seti Değerlendirme
print("\n📊 2. Test seti değerlendirmesi...")

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
true_labels = test_dataset['label']

# Confusion Matrix
print("\n🔍 3. Confusion matrix oluşturuluyor...")

cm = confusion_matrix(true_labels, preds)

plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
            xticklabels=[id2label[i] for i in sorted(id2label.keys())],
            yticklabels=[id2label[i] for i in sorted(id2label.keys())])
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Gerçek Kategori', fontsize=12)
plt.xlabel('Tahmin Edilen Kategori', fontsize=12)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix kaydedildi: confusion_matrix.png")

# 3. Kategori Başına Performans
print("\n📈 4. Kategori bazlı analiz...")

report = classification_report(
    true_labels,
    preds,
    target_names=[id2label[i] for i in sorted(id2label.keys())],
    digits=4,
    output_dict=True
)

# DataFrame oluştur
categories_df = pd.DataFrame({
    'Category': [cat for cat in report.keys() if cat not in ['accuracy', 'macro avg', 'weighted avg']],
    'Precision': [report[cat]['precision'] for cat in report.keys() if
                  cat not in ['accuracy', 'macro avg', 'weighted avg']],
    'Recall': [report[cat]['recall'] for cat in report.keys() if cat not in ['accuracy', 'macro avg', 'weighted avg']],
    'F1-Score': [report[cat]['f1-score'] for cat in report.keys() if
                 cat not in ['accuracy', 'macro avg', 'weighted avg']],
    'Support': [report[cat]['support'] for cat in report.keys() if cat not in ['accuracy', 'macro avg', 'weighted avg']]
})

categories_df = categories_df.sort_values('F1-Score', ascending=False)

# En iyi ve en kötü 10 kategori
print("\n✅ En İyi 10 Kategori (F1-Score):")
print(categories_df.head(10).to_string(index=False))

print("\n❌ En Kötü 10 Kategori (F1-Score):")
print(categories_df.tail(10).to_string(index=False))

# Grafik
fig, ax = plt.subplots(figsize=(12, 10))
top_bottom = pd.concat([categories_df.head(10), categories_df.tail(10)])
colors = ['green'] * 10 + ['red'] * 10
ax.barh(range(len(top_bottom)), top_bottom['F1-Score'], color=colors, alpha=0.7)
ax.set_yticks(range(len(top_bottom)))
ax.set_yticklabels(top_bottom['Category'])
ax.set_xlabel('F1-Score')
ax.set_title('En İyi ve En Kötü 10 Kategori')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('category_performance.png', dpi=300, bbox_inches='tight')
print("\n✅ Kategori performans grafiği kaydedildi: category_performance.png")

# 4. Veri Dağılımı Analizi
print("\n📊 5. Veri dağılımı analizi...")

category_counts = train_df['label'].value_counts()
category_names = [id2label[i] for i in category_counts.index]

plt.figure(figsize=(15, 8))
plt.bar(range(len(category_counts)), category_counts.values, alpha=0.7)
plt.xticks(range(len(category_counts)), category_names, rotation=90)
plt.xlabel('Kategori')
plt.ylabel('Örnek Sayısı')
plt.title('Eğitim Setinde Kategori Dağılımı')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Veri dağılımı grafiği kaydedildi: data_distribution.png")

# 5. En Çok Karıştırılan Kategoriler
print("\n🔄 6. En çok karıştırılan kategoriler...")

# Her kategori için en çok hangi kategoriyle karıştırıldığını bul
confusion_pairs = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i][j] > 0:
            confusion_pairs.append({
                'True': id2label[i],
                'Predicted': id2label[j],
                'Count': cm[i][j]
            })

confusion_df = pd.DataFrame(confusion_pairs).sort_values('Count', ascending=False)

print("\nEn Çok Karıştırılan 15 Kategori Çifti:")
print(confusion_df.head(15).to_string(index=False))

# 6. Yanlış Tahminler Analizi
print("\n❌ 7. Yanlış tahmin örnekleri...")

wrong_predictions = []
test_texts = test_df.reset_index(drop=True)['text']

for idx, (pred, true) in enumerate(zip(preds, true_labels)):
    if pred != true:
        wrong_predictions.append({
            'Text': test_texts[idx][:100] + '...',
            'True': id2label[true],
            'Predicted': id2label[pred]
        })

wrong_df = pd.DataFrame(wrong_predictions[:20])  # İlk 20 yanlış
print("\nİlk 20 Yanlış Tahmin:")
print(wrong_df.to_string(index=False))

# 7. Özet Rapor
print("\n" + "=" * 60)
print("📋 ÖZET RAPOR")
print("=" * 60)

final_results = trainer.evaluate()

print(f"\n✅ Test Accuracy: {final_results['eval_accuracy']:.4f}")
print(f"✅ Test F1 Score (Weighted): {final_results['eval_f1']:.4f}")
print(f"✅ Toplam Kategori Sayısı: {num_labels}")
print(f"✅ Train Örnekleri: {len(train_df)}")
print(f"✅ Test Örnekleri: {len(test_df)}")
print(f"✅ Toplam Epoch: {training_args.num_train_epochs}")

# Makro ortalamalar
print(f"\n📊 Makro Ortalamalar:")
print(f"  Precision: {report['macro avg']['precision']:.4f}")
print(f"  Recall: {report['macro avg']['recall']:.4f}")
print(f"  F1-Score: {report['macro avg']['f1-score']:.4f}")

# Veri dengesizliği
imbalance_ratio = category_counts.max() / category_counts.min()
print(f"\n⚖️  Veri Dengesizlik Oranı: {imbalance_ratio:.2f}x")
print(f"  En fazla örnek: {category_counts.max()} ({id2label[category_counts.idxmax()]})")
print(f"  En az örnek: {category_counts.min()} ({id2label[category_counts.idxmin()]})")

# 8. Tüm Sonuçları CSV'ye Kaydet
categories_df.to_csv('category_results.csv', index=False)
confusion_df.to_csv('confusion_pairs.csv', index=False)
wrong_df.to_csv('wrong_predictions.csv', index=False)

print("\n✅ Tüm analizler tamamlandı ve kaydedildi!")
print("\nOluşturulan dosyalar:")
print("  - training_history.png")
print("  - confusion_matrix.png")
print("  - category_performance.png")
print("  - data_distribution.png")
print("  - category_results.csv")
print("  - confusion_pairs.csv")
print("  - wrong_predictions.csv")

print("\n🎉 Rapor hazır!")

# Model klasörünü zip'le
#!zip - r
#huffpost_model.zip. / huffpost_news_classifier
#!zip - r
#analysis_results.zip *.png *.csv

print("\n📦 Model ve analizler zip'lendi:")
print("  - huffpost_model.zip (modeli indirin)")
print("  - analysis_results.zip (grafik ve CSV'leri indirin)")