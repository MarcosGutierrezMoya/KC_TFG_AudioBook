# shared_variable.py
# Este archivo actúa como puente entre el notebook y Flask

import json
import os
from threading import Lock

# Archivo donde se guardará la variable
VARIABLE_FILE = 'variable_data.json'
lock = Lock()

# Variable por defecto
DEFAULT_VALUE = 'Valor Inicial'
LANG_VALUE = 'Lenguaje Inicial'

def get_variable():
    """Obtiene la variable desde el archivo JSON"""
    try:
        if os.path.exists(VARIABLE_FILE):
            with lock:
                with open(VARIABLE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {"prompt": data.get('variable_a_modificar', DEFAULT_VALUE),"language":data.get('language', LANG_VALUE)}
        return {"prompt":DEFAULT_VALUE, "language":LANG_VALUE}
    except Exception as e:
        print(f"Error al leer la variable: {e}")
        return {"prompt": DEFAULT_VALUE, "language": LANG_VALUE}

def set_variable(new_value, language_value):
    """Guarda la variable en el archivo JSON"""
    try:
        with lock:
            data = {'variable_a_modificar': new_value, 'language': language_value}
            with open(VARIABLE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Variable actualizada a: {new_value}")
        return new_value
    except Exception as e:
        print(f"Error al guardar la variable: {e}")
        return None

def initialize_variable(initial_value=None, initial_language=None):
    """Inicializa la variable si no existe el archivo"""
    if not os.path.exists(VARIABLE_FILE):
        value = initial_value if initial_value is not None else DEFAULT_VALUE
        language = initial_language if initial_language is not None else LANG_VALUE
        set_variable(value, language)
        return {"prompt": value, "language": language}
    return get_variable()

# Inicializar al importar
initialize_variable()