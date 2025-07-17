# Audiolibro
### by [Marc Van Cutsem](https://github.com/MarcVCE) y [Marcos Gutierrez](https://github.com/MarcosGutierrezMoya)

La idea es tener un modelo de lenguaje finetunueado con libros e historias y otro modelo que sea capaz de convertir a audio el texto.  
Para ello hemos cogido un modelo de código abierto y de uso local (precio gratuito) de Hugging Face, el **gemma-3-4b-it**, que es decente en la generación de texto, junto con la librería 
de **josecannete/large_spanish_corpus**, que nos ofrecía un dataset en español para poder finetunnear nuestra generación de texto a una mejor generación que la ofrecida por el propio modelo.
Al usar el modelo de manera local, hemos limitado los tokens de salida.

Si no quieres instalar las librerías en local, puedes crearte un entorno virtual:
```
python -m venv <nombre_del_entorno> #para crearlo
./env/Scripts/activate #Para acceder a él
source ./env/Scripts/activate #Para acceder si estás en windows
```
Se necesitaran las librerías de `requirements.txt`:
```
pip install -r requirements.txt
```
Para iniciar el front de la aplicación:
(abrir el puerto en local, seguramente el 5000)
```
python app.py
```
También requiere una cuenta en Hugging Face y una api key que tendréis que tener en un .env:
```py
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx
```

El fine-tuning no ha llegado a funcionar por que uno no tiene el espacio en disco suficiente y el otro tiene una gráfica demasiado actual (serie 50XX) y la librería de Pytorch aún no funciona con esa gráfica.  
Igualmente el entrenamiento en local es costoso y, para intentar mejorar el rendimiento del modelo y optimizar los recursos, hemos utilizado la librería **PEFT** y cambiamos a Qlora 4bits ya que nos dió un bug al usar el de 8bits.

Hemos usado un modelo de voz también de código abierto, [Kokoro](https://github.com/hexgrad/kokoro?tab=readme-ov-file#advanced-usage).  
Aún así hemos distosionado un poco la voz, cambiándole los ajustes como el pitch o la velocidad, con el fin de evitar problemas de derechos de autor por voces no propias/no autorizadas.
