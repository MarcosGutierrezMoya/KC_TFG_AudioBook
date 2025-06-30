import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Fuerza uso exclusivo de CPU (comentado para usar GPU)

import pandas as pd
import random
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextDataset,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from rouge_score import rouge_scorer

torch.cuda.empty_cache()

# ========== 1. CARGA Y FILTRADO ==========
streamed_dataset = load_dataset(
    "josecannete/large_spanish_corpus",
    split="train",
    trust_remote_code=True,
    streaming=True
)

sample_list = []
for i, sample in enumerate(streamed_dataset):
    sample_list.append(sample)
    if len(sample_list) == 4000:
        break

print(f"✅ Cargadas {len(sample_list)} muestras")

random.seed(42)
random.shuffle(sample_list)
sample_list = sample_list[:2000]
print(f"🎯 Seleccionadas {len(sample_list)} muestras aleatorias")

df = pd.DataFrame(sample_list)

# ========== 2. LIMPIEZA ==========
df["text"] = (
    df["text"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.replace(r"<[^>]+>", "", regex=True)
    .str.replace(r"[^\w\s.,;¡!¿?\-\"']", "", regex=True)
    .str.strip()
)
df = df[df["text"].str.len() > 10].reset_index(drop=True)

# ========== 3. GUARDAR .TXT ==========
with open("train.txt", "w", encoding="utf-8") as f:
    for row in df["text"]:
        f.write(row + "\n")
print("📄 Guardado en train.txt (formato entrenable)")

# ========== 4. FINE-TUNING ==========
print("🧠 Cargando tokenizer y modelo base...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", cache_dir="gemma-3-4b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", cache_dir="gemma-3-4b-it")

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="train.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./gemma-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    save_steps=500,
    gradient_checkpointing=True,  # ✅ Habilita gradient checkpointing para ahorrar VRAM
    save_total_limit=1,
    logging_steps=100,
    fp16=True,  # ✅ Usa float16 para ahorrar VRAM
    # no_cuda=True  # ❌ Quitar para usar GPU
)

if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # Refuerza tras habilitar gradient checkpointing

print("🚀 Iniciando entrenamiento en GPU (si está disponible)...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)
trainer.train()

# ========== 5. GUARDAR ==========
print("💾 Guardando modelo y tokenizer...")
trainer.save_model("./gemma-finetuned")
tokenizer.save_pretrained("./gemma-finetuned")
print("✅ Fine-tuning completo.")

# ========== 6. EVALUACIÓN ROUGE ==========
print("📏 Evaluando métricas ROUGE...")

# Re-cargar modelo entrenado en GPU o CPU según disponibilidad
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("./gemma-finetuned").to(device)
tokenizer = AutoTokenizer.from_pretrained("./gemma-finetuned")

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
evaluation_subset = df.sample(50, random_state=42).reset_index(drop=True)

rouge_scores = []

for i, row in evaluation_subset.iterrows():
    input_text = row["text"][:100]
    original = row["text"]

    try:
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        score = scorer.score(original, generated)
        rouge_scores.append({
            "input": input_text,
            "original": original,
            "generated": generated,
            "rouge1": score["rouge1"].fmeasure,
            "rougeL": score["rougeL"].fmeasure
        })
    except Exception as e:
        rouge_scores.append({
            "input": input_text,
            "original": original,
            "generated": "",
            "rouge1": 0,
            "rougeL": 0,
            "error": str(e)
        })

# Guardar métricas
avg_rouge1 = sum(s["rouge1"] for s in rouge_scores) / len(rouge_scores)
avg_rougeL = sum(s["rougeL"] for s in rouge_scores) / len(rouge_scores)

metrics = {
    "average_rouge1": avg_rouge1,
    "average_rougeL": avg_rougeL,
    "num_samples": len(rouge_scores)
}

with open("rouge_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

pd.DataFrame(rouge_scores).to_csv("rouge_detailed_scores.csv", index=False, encoding="utf-8")
print("📊 Métricas ROUGE guardadas en:")
print(" - rouge_metrics.json")
print(" - rouge_detailed_scores.csv")
