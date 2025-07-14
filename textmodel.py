from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
import os
from dotenv import load_dotenv
import shared_variable
from playwright.sync_api import sync_playwright

def init():
    # 📁 Establecer carpeta actual como destino para guardar el modelo
    local_model_dir = os.path.join(os.getcwd(), "gemma-3-4b-it")

    # Cargar variables del archivo .env
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    # Login con Hugging Face
    if token:
        login(token=token)
        print("✅ Login exitoso.")
    else:
        print("❌ No se encontró HF_TOKEN en el archivo .env.")

    # 🚫 Desactivar TorchDynamo (evita errores con Triton)
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    # 🔁 Cargar modelo/tokenizador desde Hugging Face, pero guardado en la carpeta local
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", cache_dir=local_model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-it",
        cache_dir=local_model_dir,
        # torch_dtype= torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        torch_dtype= torch.float32
    )

    # 🧠 Prompt
    prompt = shared_variable.get_variable()
    inputs = tokenizer(prompt["prompt"], return_tensors="pt", padding=True)

    # GPU si está disponible
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     inputs = {k: v.cuda() for k, v in inputs.items()}

    # ✍️ Generar texto
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=100,
        attention_mask=inputs["attention_mask"],
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # 📤 Mostrar respuesta
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Texto generado:",generated_text)
    # Quitar el texto del prompt si está al inicio del texto generado
    prompt_text = prompt["prompt"].strip()
    if generated_text.strip().startswith(prompt_text):
        cleaned_text = generated_text.strip()[len(prompt_text):].lstrip()
    else:
        cleaned_text = generated_text.strip()
    shared_variable.set_variable(cleaned_text, prompt["language"])
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # headless=True para ocultar el navegador
        page = browser.new_page()
        page.goto("http://localhost:5000")  # Cambia la URL si es necesario

        # Ejecuta funciones JS en el contexto de la página
        page.evaluate("executeNotebook()")
        page.evaluate(f"updateVariable('{cleaned_text}')")

        # Espera para ver resultados o interactuar más
        page.wait_for_timeout(5000)
        browser.close()
    print("\n📗 Respuesta (modelo manual):")
    print(cleaned_text)
