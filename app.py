from flask import Flask, request,render_template, jsonify, render_template_string
from flask_cors import CORS
import shared_variable
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import subprocess
import sys
from threading import Thread
import textmodel

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True

@app.route('/')
def index():
    """Página principal - sirve el archivo HTML"""
    return render_template('index.html')

@app.route('/api/variable', methods=['GET'])
def api_get_variable():
    try:
        value = shared_variable.get_variable()
        return jsonify({
            'value': value,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/variable', methods=['POST'])
def api_set_variable():
    try:
        data = request.get_json()
        new_value = data.get('value', '')
        
        if new_value is not None:
            updated_value = shared_variable.set_variable(new_value)
            textmodel.init()
            if updated_value is not None:
                return jsonify({
                    'value': updated_value,
                    'status': 'success',
                    'message': f'Variable actualizada a: {updated_value}'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Error al actualizar la variable'
                }), 500
        else:
            return jsonify({
                'status': 'error',
                'message': 'No se proporcionó un valor'
            }), 400
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/execute-notebook', methods=['POST'])
def execute_notebook():
    try:
        data = request.get_json()
        notebook_path = data.get('notebook_path', 'tu_notebook.ipynb')
        
        # Verificar que el archivo existe
        if not os.path.exists(notebook_path):
            return jsonify({
                'status': 'error',
                'message': f'Notebook no encontrado: {notebook_path}'
            }), 404
        
        def run_notebook():
            try:
                # Leer el notebook
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                
                # Configurar el ejecutor
                ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
                
                # Ejecutar el notebook
                ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path) or '.'}})
                
                # Guardar el notebook ejecutado
                with open(notebook_path, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
                    
                print(f"✅ Notebook {notebook_path} ejecutado correctamente")
                
            except Exception as e:
                print(f"❌ Error ejecutando notebook: {e}")
        
        # Ejecutar en hilo separado para no bloquear la respuesta
        thread = Thread(target=run_notebook)
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': f'Ejecutando notebook: {notebook_path}'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error al ejecutar notebook: {str(e)}'
        }), 500

@app.route('/api/execute-cell', methods=['POST'])
def execute_cell():
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        if not code.strip():
            return jsonify({
                'status': 'error',
                'message': 'No se proporcionó código para ejecutar'
            }), 400
        
        def run_code():
            try:
                # Ejecutar el código en el contexto actual
                exec(code, globals())
                print(f"✅ Código ejecutado correctamente")
            except Exception as e:
                print(f"❌ Error ejecutando código: {e}")
        
        # Ejecutar en hilo separado
        thread = Thread(target=run_code)
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Código ejecutado correctamente'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error al ejecutar código: {str(e)}'
        }), 500

@app.route('/api/run-python-file', methods=['POST'])
def run_python_file():
    try:
        data = request.get_json()
        file_path = data.get('file_path', '')
        
        if not file_path:
            return jsonify({
                'status': 'error',
                'message': 'No se proporcionó ruta del archivo'
            }), 400
            
        if not os.path.exists(file_path):
            return jsonify({
                'status': 'error',
                'message': f'Archivo no encontrado: {file_path}'
            }), 404
        
        def run_file():
            try:
                # Ejecutar el archivo Python
                result = subprocess.run([sys.executable, file_path], 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print(f"✅ Archivo {file_path} ejecutado correctamente")
                    if result.stdout:
                        print("📤 Output:", result.stdout)
                else:
                    print(f"❌ Error ejecutando {file_path}:", result.stderr)
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ Timeout ejecutando {file_path}")
            except Exception as e:
                print(f"❌ Error ejecutando archivo: {e}")
        
        # Ejecutar en hilo separado
        thread = Thread(target=run_file)
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': f'Ejecutando archivo: {file_path}'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error al ejecutar archivo: {str(e)}'
        }), 500

# Template HTML (el mismo que antes)
HTML_TEMPLATE = '''
'''

if __name__ == '__main__':
    print("🚀 Iniciando servidor Flask para Notebook...")
    print("📱 Frontend disponible en: http://localhost:5000")
    print("📓 Conectado a la variable del notebook via archivo compartido")
    app.run(debug=True, host='0.0.0.0', port=5000)