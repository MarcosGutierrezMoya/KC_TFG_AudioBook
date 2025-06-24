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

def get_variable():
    """Obtiene la variable desde el archivo JSON"""
    try:
        if os.path.exists(VARIABLE_FILE):
            with lock:
                with open(VARIABLE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('variable_a_modificar', DEFAULT_VALUE)
        return DEFAULT_VALUE
    except Exception as e:
        print(f"Error al leer la variable: {e}")
        return DEFAULT_VALUE

def set_variable(new_value):
    """Guarda la variable en el archivo JSON"""
    try:
        with lock:
            data = {'variable_a_modificar': new_value}
            with open(VARIABLE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Variable actualizada a: {new_value}")
        return new_value
    except Exception as e:
        print(f"Error al guardar la variable: {e}")
        return None

def initialize_variable(initial_value=None):
    """Inicializa la variable si no existe el archivo"""
    if not os.path.exists(VARIABLE_FILE):
        value = initial_value if initial_value is not None else DEFAULT_VALUE
        set_variable(value)
        return value
    return get_variable()

# Inicializar al importar
initialize_variable()