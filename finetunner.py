import pandas as pd
import random
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils import prepare_model_for_kbit_training
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

# ========== 4. FINE-TUNING CON PEFT (LoRA) ==========
print("🧠 Cargando tokenizer y modelo base...")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", cache_dir="gemma-3-4b-it")


# Cambiar a quantización 4-bit (más estable con LoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)


model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    cache_dir="gemma-3-4b-it",
    device_map="auto",
    quantization_config=bnb_config
)

# Prepare model for k-bit training (LoRA + 8bit/4bit)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)


model = get_peft_model(model, lora_config)

# Debug: Print trainable parameters (if needed to check setup)
# trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
# frozen_params = [n for n, p in model.named_parameters() if not p.requires_grad]
# print(f"Trainable parameters: {len(trainable_params)}")
# print(f"Frozen parameters: {len(frozen_params)}")
# if len(trainable_params) == 0:
#     print("❌ No parameters require gradients! Training will fail.")
# else:
#     print("✅ Parameters set up for training.")

# ========== 5. TOKENIZAR DATOS ==========
from datasets import load_dataset as load_text_dataset

dataset = load_text_dataset("text", data_files={"train": "train.txt"})
def tokenize_fn(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model.config.use_cache = False

# ========== 6. ENTRENAMIENTO ==========
training_args = TrainingArguments(
    output_dir="./gemma-peft-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    save_steps=500,
    gradient_checkpointing=False,  # Disabled due to bitsandbytes bug
    save_total_limit=1,
    logging_steps=100,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

print("🚀 Iniciando entrenamiento...")
trainer.train()

# ========== 7. GUARDAR ADAPTADORES ==========
print("💾 Guardando adaptadores LoRA...")
model.save_pretrained("./gemma-peft-adapters")
tokenizer.save_pretrained("./gemma-peft-adapters")
print("✅ Fine-tuning completo con LoRA.")

# ========== 8. EVALUACIÓN ROUGE ==========
print("📏 Evaluando métricas ROUGE...")

from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", device_map="auto")
model = PeftModel.from_pretrained(base_model, "./gemma-peft-adapters").to(device)

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