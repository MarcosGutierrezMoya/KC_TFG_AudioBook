        // setInterval(getCurrentValue, 5000);

        function showMessage(text, type = 'success') {
            const messageDiv = document.getElementById('message');
            messageDiv.textContent = text;
            messageDiv.className = `message ${type}`;
            messageDiv.classList.add('show');

            setTimeout(() => {
                messageDiv.classList.remove('show');
            }, 3000);
        }

        async function getCurrentValue() {
            try {
                const response = await fetch('/api/variable');
                const data = await response.json();

                if (data.status === 'success') {
                    document.getElementById('currentValue').textContent = data.value || 'Sin valor';
                } else {
                    console.error('Error al obtener el valor');
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }

        async function updateVariable(text = "") {
            const newValue = document.getElementById('newValue').value;
            const language = document.getElementById('language').value;

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
                        value: text === "" ? newValue : text,
                        language: language
                    })
                });

                const data = await response.json();

                if (data.status === 'success') {
                    document.getElementById('currentValue').textContent = data.value;
                    document.getElementById('newValue').value = '';
                    showMessage('✅ Variable actualizada en el notebook!\nAhora comienza la conversión a audio', 'success');
                } else {
                    showMessage(data.message || 'Error al actualizar', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showMessage('Error de conexión', 'error');
            }
        }
        
        // Nuevas funciones para ejecutar código
        async function executeNotebook() {
            // const notebookPath = document.getElementById('notebookPath').value || 'ttsCollab.ipynb';

            try {
                showMessage('🔄 Ejecutando notebook...', 'success');

                const response = await fetch('/api/execute-notebook', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        notebook_path: 'ttsCollab.ipynb'
                    })
                });

                const data = await response.json();

                if (data.status === 'success') {
                    showMessage('✅ Notebook ejecutándose en segundo plano', 'success');
                    // Actualizar la variable después de un momento
                    setTimeout(() => {
                        getCurrentValue();
                    }, 2000);
                } else {
                    showMessage(data.message || 'Error al ejecutar notebook', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showMessage('Error de conexión al ejecutar notebook', 'error');
            }
        }

        async function runPythonFile() {
            const filePath = document.getElementById('pythonFilePath').value;

            if (!filePath.trim()) {
                showMessage('Por favor introduce la ruta del archivo Python', 'error');
                return;
            }

            try {
                showMessage('🔄 Ejecutando archivo Python...', 'success');

                const response = await fetch('/api/run-python-file', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        file_path: filePath
                    })
                });

                const data = await response.json();

                if (data.status === 'success') {
                    showMessage('✅ Archivo Python ejecutándose', 'success');
                    setTimeout(() => {
                        getCurrentValue();
                    }, 2000);
                } else {
                    showMessage(data.message || 'Error al ejecutar archivo', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showMessage('Error de conexión al ejecutar archivo', 'error');
            }
        }

        async function executeCode() {
            const code = document.getElementById('codeToExecute').value;

            if (!code.trim()) {
                showMessage('Por favor introduce código para ejecutar', 'error');
                return;
            }

            try {
                showMessage('⚡ Ejecutando código...', 'success');

                const response = await fetch('/api/execute-cell', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        code: code
                    })
                });

                const data = await response.json();

                if (data.status === 'success') {
                    showMessage('✅ Código ejecutado correctamente', 'success');
                    setTimeout(() => {
                        getCurrentValue();
                    }, 1000);
                } else {
                    showMessage(data.message || 'Error al ejecutar código', 'error');
                }
            } catch (error) {
                console.error('Error:', error);
                showMessage('Error de conexión al ejecutar código', 'error');
            }
        }