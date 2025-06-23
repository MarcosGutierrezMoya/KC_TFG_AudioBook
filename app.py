from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permite requests desde el frontend

# Tu variable global
variable_a_modificar = 'Valor Inicial'

def get_variable():
    return variable_a_modificar

def set_variable(new_value):
    global variable_a_modificar
    variable_a_modificar = new_value
    print(f"La variable ha sido actualizada a: {variable_a_modificar}")
    return variable_a_modificar

# Ruta para servir el frontend
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# API endpoint para obtener la variable
@app.route('/api/variable', methods=['GET'])
def api_get_variable():
    return jsonify({
        'value': get_variable(),
        'status': 'success'
    })

# API endpoint para actualizar la variable
@app.route('/api/variable', methods=['POST'])
def api_set_variable():
    try:
        data = request.get_json()
        new_value = data.get('value', '')
        
        if new_value is not None:
            updated_value = set_variable(new_value)
            return jsonify({
                'value': updated_value,
                'status': 'success',
                'message': f'Variable actualizada a: {updated_value}'
            })
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

# Template HTML embebido
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Modificador de Variable</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.18);
            max-width: 500px;
            width: 100%;
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
            font-size: 1.1em;
        }
        
        input[type="text"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 1.1em;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.8);
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            transform: translateY(-2px);
        }
        
        .buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 25px;
        }
        
        button {
            padding: 15px 25px;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn-update {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .btn-update:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-get {
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: white;
        }
        
        .btn-get:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(245, 87, 108, 0.4);
        }
        
        .current-value {
            background: rgba(102, 126, 234, 0.1);
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
        }
        
        .current-value h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        
        .value-display {
            font-size: 1.1em;
            color: #333;
            font-weight: 500;
            word-wrap: break-word;
            min-height: 25px;
        }
        
        .message {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: 500;
            text-align: center;
            opacity: 0;
            transform: translateY(-10px);
            transition: all 0.3s ease;
        }
        
        .message.show {
            opacity: 1;
            transform: translateY(0);
        }
        
        .message.success {
            background: rgba(40, 167, 69, 0.1);
            color: #28a745;
            border: 1px solid rgba(40, 167, 69, 0.2);
        }
        
        .message.error {
            background: rgba(220, 53, 69, 0.1);
            color: #dc3545;
            border: 1px solid rgba(220, 53, 69, 0.2);
        }
        
        @media (max-width: 600px) {
            .container {
                padding: 25px;
                margin: 10px;
            }
            
            .buttons {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Modificador de Variable</h1>
        
        <div id="message" class="message"></div>
        
        <div class="current-value">
            <h3>Valor Actual:</h3>
            <div id="currentValue" class="value-display">Cargando...</div>
        </div>
        
        <div class="form-group">
            <label for="newValue">Nuevo Valor:</label>
            <input type="text" id="newValue" placeholder="Introduce el nuevo valor...">
        </div>
        
        <div class="buttons">
            <button class="btn-update" onclick="updateVariable()">
                Actualizar Variable
            </button>
            <button class="btn-get" onclick="getCurrentValue()">
                Obtener Valor Actual
            </button>
        </div>
    </div>

    <script>
        // Función para mostrar mensajes
        function showMessage(text, type = 'success') {
            const messageDiv = document.getElementById('message');
            messageDiv.textContent = text;
            messageDiv.className = `message ${type}`;
            messageDiv.classList.add('show');
            
            setTimeout(() => {
                messageDiv.classList.remove('show');
            }, 3000);
        }
        
        // Función para obtener el valor actual
        async function getCurrentValue() {
            try {
                const response = await fetch('/api/variable');
                const data = await response.json();
                
                if (data.status === 'success') {
                    document.getElementById('currentValue').textContent = data.value || 'Sin valor';
                    showMessage('Valor obtenido correctamente', 'success');
                } else {
                    showMessage('Error al obtener el valor', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showMessage('Error de conexión', 'error');
            }
        }
        
        // Función para actualizar la variable
        async function updateVariable() {
            const newValue = document.getElementById('newValue').value;
            
            if (!newValue.trim()) {
                showMessage('Por favor introduce un valor', 'error');
                return;
            }
            
            try {
                const response = await fetch('/api/variable', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        value: newValue
                    })
                });
                
                const data = await response.json();
                
                if (data.status === 'success') {
                    document.getElementById('currentValue').textContent = data.value;
                    document.getElementById('newValue').value = '';
                    showMessage(data.message, 'success');
                } else {
                    showMessage(data.message || 'Error al actualizar', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showMessage('Error de conexión', 'error');
            }
        }
        
        // Permitir actualizar con Enter
        document.getElementById('newValue').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                updateVariable();
            }
        });
        
        // Cargar valor inicial al cargar la página
        document.addEventListener('DOMContentLoaded', function() {
            getCurrentValue();
        });
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("🚀 Iniciando servidor Flask...")
    print("📱 Frontend disponible en: http://localhost:5000")
    print("🔧 API endpoints:")
    print("   GET  /api/variable - Obtener valor actual")
    print("   POST /api/variable - Actualizar variable")
    app.run(debug=True, host='0.0.0.0', port=5000)