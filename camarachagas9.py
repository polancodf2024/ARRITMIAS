import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tempfile
import os
import time
from datetime import datetime
import logging
from scipy import ndimage
from skimage import filters, measure, morphology
import pytesseract
from difflib import SequenceMatcher
import re
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import shutil
import paramiko
from io import StringIO, BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import matplotlib.ticker as ticker

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de matplotlib para mejor visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)

# Configuración de pytesseract (OCR)
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    try:
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    except:
        st.warning("Tesseract OCR no encontrado. La detección de texto estará limitada.")

# Configuración remota desde secrets.toml
REMOTE_HOST = st.secrets.get("remote_host", "187.217.52.137")
REMOTE_USER = st.secrets.get("remote_user", "POLANCO6")
REMOTE_PASSWORD = st.secrets.get("remote_password", "tt6plco6")
REMOTE_PORT = st.secrets.get("remote_port", 3792)
REMOTE_DIR = st.secrets.get("remote_dir", "/home/POLANCO6/CHAGAS")
REMOTE_CHAGAS = st.secrets.get("remote_chagas", "registro_chagas.csv")

# Nombres de archivos remotos
REMOTE_DATASET = "registro_chagas.csv"
REMOTE_MODEL = "chagas_model.pkl"
REMOTE_IMAGES_DIR = "training_images"
REMOTE_HISTORY = "accuracy_history.json"

# Categorías de resultado válidas
RESULT_CATEGORIES = ["POSITIVE", "NEGATIVE", "WEAK POSITIVE", "INVALID"]

# Número fijo de características para garantizar consistencia
NUM_FEATURES = 20

def check_camera_access():
    """Verifica y guía sobre el acceso a la cámara - MEJORADO"""
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 15px; border-radius: 10px; border-left: 5px solid #ffc107; margin: 10px 0;'>
    <h4>🔧 SOLUCIÓN PARA PROBLEMAS DE CÁMARA</h4>
    
    <strong>📱 PARA MÓVILES:</strong>
    • Asegúrate de usar HTTPS (Streamlit Cloud lo proporciona automáticamente)
    • Permite el acceso a la cámara cuando el navegador lo solicite
    • Usa Chrome o Safari para mejor compatibilidad
    
    <strong>💻 PARA LAPTOP:</strong>
    • <strong>Ejecuta localmente:</strong> <code>streamlit run camarachagas8.py</code>
    • <strong>Abre en Chrome:</strong> http://localhost:8501
    • <strong>Permite cámara:</strong> Cuando el navegador lo solicite
    
    <strong>🔄 ALTERNATIVA SI NO FUNCIONA:</strong>
    • Usa la opción <strong>SUBIR ARCHIVO</strong> 📁 - funciona igual de bien
    • Verifica que la cámara funcione en otras aplicaciones
    • Reinicia el navegador y prueba de nuevo
    </div>
    """, unsafe_allow_html=True)

def check_https_status():
    """Verifica si está usando HTTPS (requerido para cámara en producción) - MEJORADO"""
    try:
        st.markdown("""
        <div style='background-color: #d4edda; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; margin: 10px 0;'>
        <h4>✅ CONEXIÓN SEGURA HTTPS ACTIVADA</h4>
        <p>Esta app está usando conexión segura HTTPS. La cámara debería funcionar en dispositivos móviles.</p>
        <p><strong>💡 CONSEJO:</strong> Para mejor funcionamiento en laptop, ejecuta localmente: <code>streamlit run camarachagas8.py</code></p>
        </div>
        """, unsafe_allow_html=True)
    except:
        pass

def enhance_camera_capture():
    """Configuración MEJORADA para la cámara - Funciona en laptop y móvil"""
    st.markdown("""
    <style>
    .camera-container {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        background: #f9f9f9;
        margin: 10px 0;
    }
    .camera-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='camera-warning'>
    <strong>📸 CONSEJOS PARA MEJOR CAPTURA:</strong><br>
    • <strong>Enfoca bien</strong> la tira reactiva<br>
    • <strong>Buena iluminación</strong> para imagen clara<br>
    • Asegúrate que las letras <strong>C</strong> y <strong>T</strong> sean visibles<br>
    • Mantén la cámara <strong>estable</strong> al tomar la foto
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)
    
    try:
        picture = st.camera_input(
            "📸 Toma una foto CLARA de la tira reactiva de Chagas",
            help="Asegúrate de permitir el acceso a la cámara. Si no funciona, usa la opción de Subir Archivo."
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        return picture
        
    except Exception as e:
        st.markdown('</div>', unsafe_allow_html=True)
        st.error(f"❌ Error con la cámara: {e}")
        st.info("💡 Usa la opción de **Subir Archivo** como alternativa")
        return None

def initialize_learning_system():
    """Inicializa el sistema de aprendizaje desde servidor remoto"""
    if 'learning_data' not in st.session_state:
        st.session_state.learning_data = load_remote_learning_data()
    
    if 'model' not in st.session_state:
        st.session_state.model = load_or_create_remote_model()
    
    if 'accuracy_history' not in st.session_state:
        st.session_state.accuracy_history = load_remote_accuracy_history()
    
    if 'training_count' not in st.session_state:
        st.session_state.training_count = len(st.session_state.accuracy_history)
    
    if 'show_correction' not in st.session_state:
        st.session_state.show_correction = False
    
    if 'auto_corrections' not in st.session_state:
        st.session_state.auto_corrections = 0
    
    if 'camera_working' not in st.session_state:
        st.session_state.camera_working = True

def connect_remote():
    """Conecta al servidor remoto"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASSWORD)
        return ssh
    except Exception as e:
        logger.error(f"Error conectando al servidor remoto: {e}")
        return None

def execute_remote_command(command):
    """Ejecuta comando en servidor remoto"""
    try:
        ssh = connect_remote()
        if ssh is None:
            return False
        
        stdin, stdout, stderr = ssh.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        ssh.close()
        return exit_status == 0
    except Exception as e:
        logger.error(f"Error ejecutando comando remoto: {e}")
        return False

def ensure_remote_directories():
    """Asegura que los directorios remotos existan"""
    try:
        commands = [
            f"mkdir -p {REMOTE_DIR}",
            f"mkdir -p {REMOTE_DIR}/{REMOTE_IMAGES_DIR}"
        ]
        
        for cmd in commands:
            if not execute_remote_command(cmd):
                logger.warning(f"No se pudo crear directorio con comando: {cmd}")
                return False
        return True
    except Exception as e:
        logger.error(f"Error creando directorios remotos: {e}")
        return False

def create_remote_headers():
    """Crea el archivo remoto con encabezados si no existe"""
    try:
        ssh = connect_remote()
        if ssh is None:
            st.error("No se pudo conectar al servidor remoto")
            return False
        
        sftp = ssh.open_sftp()
        remote_path = f"{REMOTE_DIR}/{REMOTE_DATASET}"
        
        # Verificar si el archivo existe y tiene contenido
        try:
            with sftp.open(remote_path, 'r') as f:
                content = f.read(100)  # Leer primeros 100 bytes
                if len(content) > 0:
                    st.info("El archivo remoto ya existe y tiene contenido")
                    sftp.close()
                    ssh.close()
                    return True
        except:
            pass  # El archivo no existe o está vacío
        
        # Crear archivo con encabezados
        headers = "timestamp,features,predicted_result,correct_result,confidence,quality_score,evaluation_type,source,analysis_id\n"
        
        with sftp.open(remote_path, 'w') as f:
            f.write(headers)
        
        sftp.close()
        ssh.close()
        st.success("✅ Encabezados creados en archivo remoto")
        return True
        
    except Exception as e:
        st.error(f"Error creando encabezados: {e}")
        return False

def load_remote_learning_data():
    """Carga los datos de aprendizaje del archivo remoto - VERSIÓN CORREGIDA"""
    try:
        ssh = connect_remote()
        if ssh is None:
            st.warning("No se pudo conectar al servidor remoto. Usando datos vacíos.")
            return []
        
        ensure_remote_directories()
        
        sftp = ssh.open_sftp()
        remote_path = f"{REMOTE_DIR}/{REMOTE_DATASET}"
        
        try:
            # Verificar si el archivo existe y tiene contenido
            file_stats = sftp.stat(remote_path)
            if file_stats.st_size == 0:
                st.info("🆕 Archivo remoto existe pero está vacío")
                sftp.close()
                ssh.close()
                return []
            
            with sftp.open(remote_path, 'r') as remote_file:
                # Leer primeras líneas para verificar encabezados
                first_lines = []
                for i in range(2):  # Leer primeras 2 líneas
                    line = remote_file.readline()
                    if line:
                        first_lines.append(line.strip())
                
                # Si no hay suficientes líneas o no tiene encabezados válidos
                if len(first_lines) < 1 or 'timestamp' not in first_lines[0]:
                    st.warning("❌ Archivo remoto no tiene formato válido. Se creará uno nuevo.")
                    sftp.close()
                    ssh.close()
                    create_remote_headers()  # Crear encabezados
                    return []
                
                # Volver al inicio y cargar datos normalmente
                remote_file.seek(0)
                df = pd.read_csv(remote_file)
                
                if len(df) == 0:
                    st.info("📭 Archivo remoto tiene encabezados pero no datos")
                    sftp.close()
                    ssh.close()
                    return []
                
                if 'features' in df.columns:
                    df['features'] = df['features'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                
                st.success(f"✅ Datos remotos cargados: {len(df)} ejemplos")
                sftp.close()
                ssh.close()
                return df.to_dict('records')
                
        except FileNotFoundError:
            st.info("🆕 No hay archivo de datos remotos")
            sftp.close()
            ssh.close()
            return []
        except pd.errors.EmptyDataError:
            st.warning("📭 Archivo remoto está vacío")
            sftp.close()
            ssh.close()
            return []
        except Exception as e:
            st.warning(f"⚠️ Error leyendo archivo remoto: {e}")
            sftp.close()
            ssh.close()
            return []
            
    except Exception as e:
        st.error(f"Error cargando datos remotos: {e}")
        return []

def save_remote_learning_data():
    """Guarda los datos de aprendizaje en el archivo remoto"""
    try:
        if not st.session_state.learning_data:
            st.info("No hay datos para guardar")
            return
            
        ssh = connect_remote()
        if ssh is None:
            st.error("❌ No se pudo conectar al servidor remoto para guardar datos")
            return
        
        ensure_remote_directories()
        
        # Asegurar que todas las features tengan la misma longitud
        for item in st.session_state.learning_data:
            if 'features' in item and len(item['features']) != NUM_FEATURES:
                item['features'] = ensure_feature_length(item['features'])
        
        df = pd.DataFrame(st.session_state.learning_data)
        
        sftp = ssh.open_sftp()
        remote_path = f"{REMOTE_DIR}/{REMOTE_DATASET}"
        
        # Guardar en archivo temporal primero
        temp_csv = "temp_chagas_data.csv"
        df.to_csv(temp_csv, index=False)
        
        # Subir al servidor remoto
        sftp.put(temp_csv, remote_path)
        
        # Limpiar archivo temporal
        os.remove(temp_csv)
        
        sftp.close()
        ssh.close()
        
        st.success(f"💾 Datos guardados remotamente: {len(df)} ejemplos en {REMOTE_DATASET}")
        
    except Exception as e:
        logger.error(f"Error guardando datos remotos: {e}")
        st.error("❌ Error guardando datos en servidor remoto")

def load_or_create_remote_model():
    """Carga un modelo existente o crea uno nuevo desde servidor remoto"""
    try:
        ssh = connect_remote()
        if ssh is None:
            st.warning("No se pudo conectar al servidor remoto. Creando modelo local temporal.")
            return RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        
        sftp = ssh.open_sftp()
        remote_path = f"{REMOTE_DIR}/{REMOTE_MODEL}"
        
        try:
            # Verificar si el archivo existe y tiene tamaño
            file_stats = sftp.stat(remote_path)
            if file_stats.st_size == 0:
                st.info("🆕 Modelo remoto existe pero está vacío. Creando nuevo.")
                model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
                sftp.close()
                ssh.close()
                return model
            
            # Descargar modelo remoto
            temp_model = "temp_model.pkl"
            sftp.get(remote_path, temp_model)
            
            model = joblib.load(temp_model)
            os.remove(temp_model)
            
            st.success("✅ Modelo de aprendizaje cargado desde servidor remoto")
            sftp.close()
            ssh.close()
            return model
            
        except FileNotFoundError:
            st.info("🆕 No hay modelo remoto previo. Creando nuevo modelo.")
            model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            sftp.close()
            ssh.close()
            return model
            
    except Exception as e:
        st.warning(f"Creando nuevo modelo: {e}")
        return RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)

def save_remote_model():
    """Guarda el modelo entrenado en el servidor remoto"""
    try:
        ssh = connect_remote()
        if ssh is None:
            st.error("❌ No se pudo conectar al servidor remoto para guardar modelo")
            return
        
        ensure_remote_directories()
        
        sftp = ssh.open_sftp()
        remote_path = f"{REMOTE_DIR}/{REMOTE_MODEL}"
        
        # Guardar en archivo temporal primero
        temp_model = "temp_model.pkl"
        joblib.dump(st.session_state.model, temp_model)
        
        # Subir al servidor remoto
        sftp.put(temp_model, remote_path)
        
        # Limpiar archivo temporal
        os.remove(temp_model)
        
        sftp.close()
        ssh.close()
        
        st.success(f"💾 Modelo guardado en servidor remoto: {REMOTE_MODEL}")
        
    except Exception as e:
        logger.error(f"Error guardando modelo remoto: {e}")
        st.error("❌ Error guardando modelo en servidor remoto")

def load_remote_accuracy_history():
    """Carga el historial de precisión desde servidor remoto"""
    try:
        ssh = connect_remote()
        if ssh is None:
            return []
        
        sftp = ssh.open_sftp()
        remote_path = f"{REMOTE_DIR}/{REMOTE_HISTORY}"
        
        try:
            # Verificar si el archivo existe y tiene contenido
            file_stats = sftp.stat(remote_path)
            if file_stats.st_size == 0:
                sftp.close()
                ssh.close()
                return []
            
            with sftp.open(remote_path, 'r') as remote_file:
                history = json.load(remote_file)
                sftp.close()
                ssh.close()
                return history
        except FileNotFoundError:
            sftp.close()
            ssh.close()
            return []
        except json.JSONDecodeError:
            st.warning("Historial remoto tiene formato inválido")
            sftp.close()
            ssh.close()
            return []
            
    except Exception as e:
        logger.error(f"Error cargando historial remoto: {e}")
        return []

def save_remote_accuracy_history():
    """Guarda el historial de precisión en el servidor remoto"""
    try:
        if not st.session_state.accuracy_history:
            return
            
        ssh = connect_remote()
        if ssh is None:
            return
        
        ensure_remote_directories()
        
        sftp = ssh.open_sftp()
        remote_path = f"{REMOTE_DIR}/{REMOTE_HISTORY}"
        
        # Guardar en archivo temporal primero
        temp_history = "temp_history.json"
        with open(temp_history, 'w') as f:
            json.dump(st.session_state.accuracy_history, f)
        
        # Subir al servidor remoto
        sftp.put(temp_history, remote_path)
        
        # Limpiar archivo temporal
        os.remove(temp_history)
        
        sftp.close()
        ssh.close()
        
    except Exception as e:
        logger.error(f"Error guardando historial remoto: {e}")

def save_remote_training_image(image_array, analysis_id, correct_result):
    """Guarda imagen para entrenamiento en servidor remoto"""
    try:
        ssh = connect_remote()
        if ssh is None:
            logger.error("No se pudo conectar para guardar imagen")
            return None
        
        ensure_remote_directories()
        
        # Convertir imagen a bytes
        img_pil = Image.fromarray(image_array)
        img_bytes = BytesIO()
        img_pil.save(img_bytes, format='JPEG', quality=90)
        img_bytes.seek(0)
        
        # Nombre del archivo remoto
        filename = f"{analysis_id}_{correct_result}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        remote_path = f"{REMOTE_DIR}/{REMOTE_IMAGES_DIR}/{filename}"
        
        sftp = ssh.open_sftp()
        
        # Subir imagen al servidor remoto
        with sftp.file(remote_path, 'wb') as remote_file:
            remote_file.write(img_bytes.getvalue())
        
        sftp.close()
        ssh.close()
        
        logger.info(f"✅ Imagen guardada remotamente: {filename}")
        return remote_path
        
    except Exception as e:
        logger.error(f"Error guardando imagen remota: {e}")
        return None

def load_remote_images_list():
    """Carga la lista de imágenes disponibles en el servidor remoto"""
    try:
        ssh = connect_remote()
        if ssh is None:
            return []
        
        sftp = ssh.open_sftp()
        remote_dir = f"{REMOTE_DIR}/{REMOTE_IMAGES_DIR}"
        
        try:
            files = sftp.listdir(remote_dir)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            sftp.close()
            ssh.close()
            return image_files
        except FileNotFoundError:
            sftp.close()
            ssh.close()
            return []
            
    except Exception as e:
        logger.error(f"Error listando imágenes remotas: {e}")
        return []

def load_remote_image(filename):
    """Carga una imagen específica desde el servidor remoto"""
    try:
        ssh = connect_remote()
        if ssh is None:
            return None
        
        sftp = ssh.open_sftp()
        remote_path = f"{REMOTE_DIR}/{REMOTE_IMAGES_DIR}/{filename}"
        
        try:
            # Descargar imagen a memoria
            with sftp.open(remote_path, 'rb') as remote_file:
                img_bytes = remote_file.read()
            
            # Convertir bytes a imagen numpy
            img_pil = Image.open(BytesIO(img_bytes))
            img_array = np.array(img_pil)
            
            sftp.close()
            ssh.close()
            return img_array
            
        except FileNotFoundError:
            sftp.close()
            ssh.close()
            return None
            
    except Exception as e:
        logger.error(f"Error cargando imagen remota {filename}: {e}")
        return None

def ensure_feature_length(features, target_length=NUM_FEATURES):
    """Asegura que las características tengan la longitud correcta"""
    if len(features) == target_length:
        return features
    elif len(features) > target_length:
        return features[:target_length]
    else:
        return features + [0.0] * (target_length - len(features))

def extract_features_from_analysis(analysis):
    """Extrae características para el modelo de ML"""
    try:
        features = []
        
        # 1. Características de bandas (6 características)
        features.extend([
            float(analysis.get('control_present', 0)),
            float(analysis.get('test_present', 0)),
            float(analysis.get('control_intensity', 0)),
            float(analysis.get('test_intensity', 0)),
            float(analysis.get('intensity_ratio', 0)),
            float(analysis.get('confidence', 0))
        ])
        
        # 2. Características de calidad (4 características)
        quality = analysis.get('quality_analysis', {})
        features.extend([
            float(quality.get('brightness', 0)),
            float(quality.get('contrast', 0)),
            float(quality.get('sharpness', 0)),
            float(quality.get('quality_score', 0))
        ])
        
        # 3. Características de texto (4 características)
        text_data = analysis.get('text_detection', {})
        features.extend([
            float(len(text_data.get('keywords', []))),
            float(1 if text_data.get('has_chagas_text') else 0),
            float(1 if text_data.get('has_control_text') else 0),
            float(1 if text_data.get('has_test_text') else 0)
        ])
        
        # 4. Características de letras (4 características)
        letters_data = analysis.get('letters_detection', {})
        features.extend([
            float(1 if letters_data and letters_data.get('C_detected') else 0),
            float(1 if letters_data and letters_data.get('T_detected') else 0),
            float(letters_data.get('C_confidence', 0) if letters_data else 0),
            float(letters_data.get('T_confidence', 0) if letters_data else 0)
        ])
        
        # 5. Características adicionales para completar (2 características)
        features.extend([
            float(analysis.get('quality_score', 0) / 100.0),
            float(min(analysis.get('validated_confidence', 0) / 100.0, 1.0))
        ])
        
        features = ensure_feature_length(features, NUM_FEATURES)
        return features
        
    except Exception as e:
        logger.error(f"Error extrayendo características: {e}")
        return [0.0] * NUM_FEATURES

def train_model():
    """Entrena el modelo con los datos acumulados y guarda en remoto"""
    if len(st.session_state.learning_data) < 2:  # Reducido a 2 para más agresividad
        st.warning(f"Se necesitan al menos 2 ejemplos para entrenar. Actual: {len(st.session_state.learning_data)}")
        return None
    
    try:
        features = []
        labels = []
        valid_items = 0
        
        for item in st.session_state.learning_data:
            if 'features' in item and 'correct_result' in item:
                if (isinstance(item['features'], (list, np.ndarray)) and 
                    len(item['features']) == NUM_FEATURES and
                    all(isinstance(x, (int, float, np.number)) for x in item['features'])):
                    
                    feature_array = np.array(item['features'], dtype=np.float64)
                    features.append(feature_array)
                    labels.append(item['correct_result'])
                    valid_items += 1
        
        if valid_items < 2:
            st.warning(f"Solo {valid_items} ejemplos válidos para entrenar")
            return None
        
        X = np.array(features)
        y = np.array(labels)
        
        logger.info(f"Forma de X: {X.shape}, Forma de y: {y.shape}")
        
        # Configuración más agresiva para pocos datos
        if len(X) < 10:
            st.session_state.model = RandomForestClassifier(
                n_estimators=30, 
                random_state=42, 
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1
            )
        
        # Usar todos los datos para entrenamiento si hay pocos
        if len(X) <= 5:
            X_train, X_test, y_train, y_test = X, X, y, y
            test_size = 0.0
        else:
            test_size = min(0.2, 1.0 / len(X))  # Test size más pequeño
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        
        st.session_state.model.fit(X_train, y_train)
        
        # Evaluar
        if len(X_test) > 0:
            y_pred = st.session_state.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
        else:
            # Si no hay test data, usar training accuracy
            y_pred_train = st.session_state.model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred_train)
        
        # Guardar TODO en remoto
        save_remote_model()
        save_remote_learning_data()
        
        st.session_state.accuracy_history.append({
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'training_samples': valid_items
        })
        
        save_remote_accuracy_history()
        
        st.session_state.training_count += 1
        return accuracy
        
    except Exception as e:
        logger.error(f"Error entrenando modelo: {e}")
        st.error(f"Error en entrenamiento: {str(e)}")
        return None

def predict_with_model(features):
    """Predice usando el modelo entrenado"""
    try:
        if (len(st.session_state.learning_data) >= 2 and  # Reducido a 2
            hasattr(st.session_state.model, 'classes_') and
            hasattr(st.session_state.model, 'predict')):
            
            features_array = np.array(features, dtype=np.float64).reshape(1, -1)
            prediction = st.session_state.model.predict(features_array)[0]
            probabilities = st.session_state.model.predict_proba(features_array)[0]
            confidence = max(probabilities)
            return prediction, confidence
        else:
            return None, 0
    except Exception as e:
        logger.error(f"Error en predicción ML: {e}")
        return None, 0

def calculate_image_similarity(img1, img2):
    """Calcula la similitud entre dos imágenes usando múltiples métodos"""
    try:
        if img1.shape != img2.shape:
            # Redimensionar img2 para que coincida con img1
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Método 1: SSIM (Structural Similarity)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
        
        from skimage.metrics import structural_similarity as ssim
        ssim_score = ssim(gray1, gray2, data_range=gray2.max() - gray2.min())
        
        # Método 2: Histogram comparison
        hist1 = cv2.calcHist([gray1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [64], [0, 256])
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Método 3: MSE (Mean Squared Error)
        mse = np.mean((gray1 - gray2) ** 2)
        mse_score = 1 / (1 + mse)  # Convertir a score entre 0-1
        
        # Combinar scores
        final_score = (ssim_score * 0.5 + hist_score * 0.3 + mse_score * 0.2)
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error calculando similitud: {e}")
        return 0.0

def find_similar_historical_image(current_image, threshold=0.75):  # Umbral más bajo
    """Busca imágenes similares en el registro histórico REMOTO"""
    try:
        similar_images = []
        
        # Cargar lista de imágenes remotas
        remote_images = load_remote_images_list()
        
        if not remote_images:
            return []
        
        for filename in remote_images:
            try:
                # Cargar imagen desde servidor remoto
                historical_img = load_remote_image(filename)
                if historical_img is not None:
                    similarity = calculate_image_similarity(current_image, historical_img)
                    
                    if similarity >= threshold:
                        # Extraer información del nombre del archivo
                        parts = filename.split('_')
                        correct_result = "POSITIVE"  # Por defecto
                        if len(parts) >= 2:
                            # Buscar el resultado en el nombre del archivo
                            for part in parts:
                                if part in RESULT_CATEGORIES:
                                    correct_result = part
                                    break
                        
                        similar_images.append({
                            'filename': filename,
                            'similarity': similarity,
                            'correct_result': correct_result,
                            'analysis_id': parts[0] if len(parts) > 0 else "unknown"
                        })
            except Exception as e:
                logger.warning(f"Error procesando imagen remota {filename}: {e}")
                continue
        
        # Ordenar por similitud (mayor primero)
        similar_images.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_images[:3]  # Retornar las 3 más similares
        
    except Exception as e:
        logger.error(f"Error buscando imágenes similares remotas: {e}")
        return []

def auto_correct_with_historical_data(current_analysis, current_image):
    """Corrige automáticamente basándose en imágenes históricas remotas"""
    try:
        similar_images = find_similar_historical_image(current_image, threshold=0.70)  # Umbral más bajo
        
        if not similar_images:
            return current_analysis['validated_result'], current_analysis['validated_confidence'], "No se encontraron imágenes similares en el servidor remoto"
        
        best_match = similar_images[0]
        
        if best_match['similarity'] >= 0.80:  # Alta similitud (reducido de 0.90)
            st.session_state.auto_corrections += 1
            corrected_result = best_match['correct_result']
            
            # Aumentar confianza basada en la similitud
            new_confidence = min(95, current_analysis['validated_confidence'] + 20)  # Boost mayor
            
            st.info(f"🔄 **Auto-corrección aplicada**: Imagen muy similar ({best_match['similarity']:.1%}) fue clasificada como **{corrected_result}** en el servidor remoto")
            
            return corrected_result, new_confidence, f"Auto-corregido basado en imagen remota similar: {best_match['filename']}"
        
        elif best_match['similarity'] >= 0.65:  # Similitud media (reducido de 0.80)
            if best_match['correct_result'] != current_analysis['validated_result']:
                st.warning(f"⚠️ **Posible discrepancia**: Imagen similar ({best_match['similarity']:.1%}) en servidor remoto fue clasificada como **{best_match['correct_result']}**")
            
            return current_analysis['validated_result'], current_analysis['validated_confidence'], f"Imagen similar encontrada en remoto: {best_match['filename']} ({best_match['similarity']:.1%})"
        
        else:
            return current_analysis['validated_result'], current_analysis['validated_confidence'], f"Imágenes similares encontradas en remoto pero con baja similitud ({best_match['similarity']:.1%})"
            
    except Exception as e:
        logger.error(f"Error en auto-corrección remota: {e}")
        return current_analysis['validated_result'], current_analysis['validated_confidence'], f"Error en auto-corrección remota: {str(e)}"

def upload_images_to_remote():
    """Sube imágenes manualmente al servidor remoto para entrenamiento"""
    st.header("📤 Cargar Imágenes al Histórico Remoto")
    
    st.warning("""
    **⚠️ IMPORTANTE:** Sube imágenes que ya hayas evaluado correctamente.
    El nombre del archivo debe indicar el resultado correcto:
    - `POSITIVE_imagen.jpg`
    - `NEGATIVE_imagen.jpg` 
    - `WEAK_POSITIVE_imagen.jpg`
    - `INVALID_imagen.jpg`
    """)
    
    uploaded_files = st.file_uploader(
        "Selecciona imágenes para agregar al entrenamiento histórico",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🚀 Subir Imágenes al Servidor Remoto"):
        success_count = 0
        for uploaded_file in uploaded_files:
            try:
                # Leer imagen
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                # Determinar resultado del nombre del archivo
                filename = uploaded_file.name.upper()
                if 'POSITIVE' in filename and 'WEAK' not in filename:
                    correct_result = "POSITIVE"
                elif 'NEGATIVE' in filename:
                    correct_result = "NEGATIVE" 
                elif 'WEAK' in filename and 'POSITIVE' in filename:
                    correct_result = "WEAK POSITIVE"
                elif 'INVALID' in filename:
                    correct_result = "INVALID"
                else:
                    st.error(f"❌ No se puede determinar resultado para: {uploaded_file.name}")
                    continue
                
                # Guardar en remoto
                analysis_id = f"MANUAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{success_count}"
                remote_path = save_remote_training_image(img_array, analysis_id, correct_result)
                
                if remote_path:
                    success_count += 1
                    st.success(f"✅ {uploaded_file.name} → {correct_result}")
                
            except Exception as e:
                st.error(f"❌ Error procesando {uploaded_file.name}: {e}")
        
        if success_count > 0:
            st.success(f"🎉 {success_count} imágenes agregadas al histórico remoto")
            st.info("**Ahora el sistema podrá hacer auto-correcciones basadas en estas imágenes**")
        else:
            st.error("❌ No se pudo subir ninguna imagen")

def force_learning_from_current_data():
    """Fuerza el aprendizaje con los datos actuales del CSV"""
    st.header("🎯 Entrenamiento Forzado")
    
    if len(st.session_state.learning_data) < 2:
        st.error("Se necesitan al menos 2 ejemplos para entrenar")
        return
    
    st.warning("""
    **ESTA ACCIÓN:** 
    - Entrenará el modelo con TODOS los datos actuales
    - Sobrescribirá el modelo anterior
    - Forzará el aprendizaje incluso con pocos datos
    """)
    
    if st.button("🚀 EJECUTAR ENTRENAMIENTO FORZADO", type="primary"):
        with st.spinner("Entrenamiento forzado en progreso..."):
            # Usar configuración más agresiva para pocos datos
            if len(st.session_state.learning_data) < 10:
                st.session_state.model = RandomForestClassifier(
                    n_estimators=30, 
                    random_state=42, 
                    max_depth=5,  # Menor profundidad para evitar overfitting
                    min_samples_split=2,  # Más flexible con pocos datos
                    min_samples_leaf=1
                )
            
            accuracy = train_model()
            
            if accuracy:
                st.success(f"✅ Modelo forzado - Precisión: {accuracy:.1%}")
                st.info(f"📊 Se usaron {len(st.session_state.learning_data)} ejemplos")
                
                # Mostrar distribución de datos
                df = pd.DataFrame(st.session_state.learning_data)
                st.write("**Distribución actual:**")
                st.write(df['correct_result'].value_counts())
            else:
                st.error("❌ Falló el entrenamiento forzado")

def clean_corrupted_data():
    """Limpia datos corruptos del dataset remoto"""
    try:
        original_count = len(st.session_state.learning_data)
        cleaned_data = []
        
        for item in st.session_state.learning_data:
            if ('features' in item and 'correct_result' in item and
                isinstance(item['features'], (list, np.ndarray)) and
                len(item['features']) == NUM_FEATURES and
                all(isinstance(x, (int, float, np.number)) for x in item['features'])):
                cleaned_data.append(item)
        
        st.session_state.learning_data = cleaned_data
        save_remote_learning_data()
        st.success(f"🧹 Datos limpiados: {original_count} → {len(cleaned_data)} ejemplos válidos")
        
    except Exception as e:
        st.error(f"Error limpiando datos: {e}")

def apply_smart_enhancement(img_array):
    """Aplica mejoras inteligentes a la imagen - MEJORADO"""
    try:
        height, width = img_array.shape[:2]
        
        # Redimensionar si es necesario (especialmente para móviles)
        if height < 400 or width < 400:
            scale_factor = max(600/width, 450/height, 1.5)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            st.success(f"🔄 Imagen mejorada: {width}x{height} → {new_width}x{new_height}")
        
        # Mejorar contraste para análisis
        if len(img_array.shape) == 3:
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE para mejorar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return img_array
    except Exception as e:
        logger.error(f"Error en mejora de imagen: {e}")
        return img_array

def analyze_image_quality_improved(img_array):
    """Análisis de calidad MEJORADO - menos estricto"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        height, width = gray.shape
        
        brightness = np.mean(gray)
        contrast = np.std(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        brightness_score = max(0, 100 - abs(brightness - 120) / 255 * 100)
        contrast_score = min(100, contrast / 2.0 * 100)
        sharpness_score = min(100, sharpness / 200 * 100)
        
        quality_score = (brightness_score * 0.3 + contrast_score * 0.4 + sharpness_score * 0.3)
        
        return {
            'brightness': brightness, 
            'contrast': contrast, 
            'sharpness': sharpness,
            'resolution': f"{width}x{height}", 
            'quality_score': quality_score,
            'quality_category': 'BUENA' if quality_score > 50 else 'ACEPTABLE' if quality_score > 30 else 'BAJA'
        }
    except:
        return {'quality_score': 60, 'quality_category': 'ACEPTABLE'}

def detect_chagas_bands_improved(img_array):
    """Detección de bandas MEJORADA - menos estricta"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        height, width = gray.shape
        
        control_region = gray[int(height*0.25):int(height*0.75), int(width*0.55):int(width*0.80)]
        test_region = gray[int(height*0.25):int(height*0.75), int(width*0.20):int(width*0.45)]
        background_region = gray[int(height*0.1):int(height*0.2), int(width*0.1):int(width*0.2)]
        
        control_mean = np.mean(control_region) if control_region.size > 0 else 255
        test_mean = np.mean(test_region) if test_region.size > 0 else 255
        background_mean = np.mean(background_region) if background_region.size > 0 else 255
        
        control_present = (background_mean - control_mean) > 15
        test_present = (background_mean - test_mean) > 10
        
        control_diff = background_mean - control_mean
        test_diff = background_mean - test_mean
        
        base_confidence = 60
        
        if not control_present:
            result = "INVALID"
            confidence = max(30, base_confidence - 30)
        elif control_present and not test_present:
            result = "NEGATIVE"
            confidence = min(90, base_confidence + 20 + min(control_diff/2, 20))
        elif control_present and test_present:
            intensity_ratio = test_mean / control_mean if control_mean > 0 else 1
            if intensity_ratio < 0.8:
                result = "POSITIVE"
                confidence = min(85, base_confidence + 15 + min(test_diff/2, 15))
            else:
                result = "WEAK POSITIVE"
                confidence = min(75, base_confidence + 10 + min(test_diff/2, 10))
        else:
            result = "INDETERMINADO"
            confidence = 50
            
        return {
            'result': result, 
            'confidence': confidence,
            'control_present': control_present, 
            'test_present': test_present,
            'control_intensity': control_mean, 
            'test_intensity': test_mean,
            'intensity_ratio': test_mean/control_mean if control_mean > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error en detección de bandas: {e}")
        return {'result': 'INDETERMINADO', 'confidence': 40, 'control_present': False, 'test_present': False}

def detect_text_on_strip(img_array):
    """Detección general de texto"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        custom_config = r'--oem 3 --psm 6'
        detected_text = pytesseract.image_to_string(gray, config=custom_config)
        cleaned = ' '.join(detected_text.split())
        keywords = ['CHAGAS', 'TEST', 'CONTROL', 'POSITIVE', 'NEGATIVE', 'INVALID', 'C', 'T']
        found_keywords = [k for k in keywords if k in cleaned.upper()]
        return {
            'raw_text': cleaned, 
            'keywords': found_keywords,
            'has_chagas_text': 'CHAGAS' in found_keywords,
            'has_control_text': 'CONTROL' in found_keywords or 'C' in found_keywords,
            'has_test_text': 'TEST' in found_keywords or 'T' in found_keywords
        }
    except:
        return {"raw_text": "", "keywords": [], "has_chagas_text": False}

def detect_letters_c_t_improved(img_array):
    """Detección MEJORADA de letras C y T con múltiples métodos"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        height, width = gray.shape
        
        results = {
            'C_detected': False,
            'T_detected': False,
            'C_confidence': 0,
            'T_confidence': 0,
            'letters_found': []
        }
        
        ocr_results = detect_letters_ocr_optimized(gray, height, width)
        contour_results = detect_letters_contours(gray, height, width)
        
        if ocr_results['C_detected'] or contour_results['C_detected']:
            results['C_detected'] = True
            results['C_confidence'] = max(ocr_results['C_confidence'], contour_results['C_confidence'])
            
        if ocr_results['T_detected'] or contour_results['T_detected']:
            results['T_detected'] = True
            results['T_confidence'] = max(ocr_results['T_confidence'], contour_results['T_confidence'])
        
        if results['C_detected']:
            results['letters_found'].append('C')
        if results['T_detected']:
            results['letters_found'].append('T')
            
        return results
        
    except Exception as e:
        logger.error(f"Error en detección de letras: {e}")
        return {
            'C_detected': False, 'T_detected': False,
            'C_confidence': 0, 'T_confidence': 0,
            'letters_found': []
        }

def detect_letters_ocr_optimized(gray, height, width):
    """OCR optimizado para letras C y T"""
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=CTct'
        
        regions_to_check = [
            (int(width*0.15), int(height*0.4), int(width*0.30), int(height*0.6)),
            (int(width*0.65), int(height*0.4), int(width*0.80), int(height*0.6)),
            (int(width*0.10), int(height*0.3), int(width*0.25), int(height*0.5)),
            (int(width*0.70), int(height*0.3), int(width*0.85), int(height*0.5))
        ]
        
        C_detected = False
        T_detected = False
        C_confidence = 0
        T_confidence = 0
        
        for i, (x1, y1, x2, y2) in enumerate(regions_to_check):
            region = binary[y1:y2, x1:x2]
            if region.size == 0:
                continue
                
            detected_text = pytesseract.image_to_string(region, config=custom_config)
            cleaned_text = re.sub(r'[^CTct]', '', detected_text.upper())
            
            region_confidence = calculate_text_confidence(region)
            
            if 'C' in cleaned_text and region_confidence > C_confidence:
                C_detected = True
                C_confidence = region_confidence
                
            if 'T' in cleaned_text and region_confidence > T_confidence:
                T_detected = True
                T_confidence = region_confidence
        
        return {
            'C_detected': C_detected,
            'T_detected': T_detected,
            'C_confidence': C_confidence,
            'T_confidence': T_confidence
        }
        
    except Exception as e:
        logger.warning(f"OCR para letras falló: {e}")
        return {'C_detected': False, 'T_detected': False, 'C_confidence': 0, 'T_confidence': 0}

def detect_letters_contours(gray, height, width):
    """Detección de letras usando procesamiento de contornos"""
    try:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        C_candidates = []
        T_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 30 or area > 2000:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            if 0.3 < aspect_ratio < 3.0:
                if x > width * 0.6:
                    C_candidates.append((x, y, w, h, area))
                elif x < width * 0.4:
                    T_candidates.append((x, y, w, h, area))
        
        C_detected = len(C_candidates) > 0
        T_detected = len(T_candidates) > 0
        
        C_confidence = min(80, len(C_candidates) * 20) if C_detected else 0
        T_confidence = min(80, len(T_candidates) * 20) if T_detected else 0
        
        return {
            'C_detected': C_detected,
            'T_detected': T_detected,
            'C_confidence': C_confidence,
            'T_confidence': T_confidence
        }
        
    except Exception as e:
        logger.error(f"Error en detección por contornos: {e}")
        return {'C_detected': False, 'T_detected': False, 'C_confidence': 0, 'T_confidence': 0}

def calculate_text_confidence(region):
    """Calcula confianza basada en la claridad del texto"""
    try:
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        contrast = np.std(region)
        
        confidence = (edge_density * 50 + min(contrast/5, 30))
        return min(90, confidence)
        
    except:
        return 50

def validate_with_letters(chagas_analysis, quality_analysis, text_detection, letters_detection):
    """Validación MEJORADA - mucho menos estricta"""
    confidence = chagas_analysis['confidence']
    quality_score = quality_analysis['quality_score']
    
    quality_factor = 0.5 + (quality_score / 100.0) * 0.5
    adjusted_confidence = confidence * quality_factor
    
    if letters_detection:
        if letters_detection['C_detected']:
            adjusted_confidence += 10
        if letters_detection['T_detected']:
            adjusted_confidence += 10
    
    final_confidence = min(95, adjusted_confidence)
    
    if quality_score < 20:
        result = "INDETERMINADO"
        notes = "Calidad de imagen muy baja"
    elif final_confidence < 40:
        result = "INDETERMINADO" 
        notes = "Confianza insuficiente"
    else:
        result = chagas_analysis['result']
        notes = "Análisis completado"
        
        if letters_detection and letters_detection['letters_found']:
            notes += f" - Letras: {', '.join(letters_detection['letters_found'])}"
    
    return {
        'validated_result': result,
        'validated_confidence': final_confidence,
        'validation_notes': notes,
        'quality_score': quality_score
    }

def render_capture_tab():
    """Pestaña de captura y análisis principal - COMPLETAMENTE MEJORADA"""
    st.header("🪳 Análisis Principal con ML Remoto")
    
    if len(st.session_state.learning_data) >= 2 and st.session_state.training_count > 0:
        st.success("✅ Sistema de ML remoto activo - Usando aprendizaje acumulado en servidor")
    else:
        st.info("🔄 Sistema en fase de aprendizaje - Necesita más evaluaciones")
    
    # Sección de cámara MEJORADA
    st.subheader("📷 Captura con Cámara")
    
    # Información expandible sobre problemas de cámara
    with st.expander("🔧 ¿Problemas con la cámara? Haz clic aquí para solución completa"):
        check_camera_access()
    
    # Usar la función MEJORADA de captura de cámara
    picture = enhance_camera_capture()
    
    if picture is not None:
        process_image_for_analysis(picture, "Cámara")
    
    # Alternativa de subida de archivo
    st.subheader("📁 Subir Archivo (Alternativa)")
    uploaded_file = st.file_uploader(
        "O sube una imagen desde tu dispositivo", 
        type=['jpg', 'jpeg', 'png'],
        help="Formatos aceptados: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        process_image_for_analysis(uploaded_file, "Archivo")
        st.success("✅ Archivo cargado correctamente")

def process_image_for_analysis(picture, source):
    """Procesa imagen para análisis con ML remoto"""
    try:
        image = Image.open(picture)
        img_array = np.array(image)
        
        with st.spinner("🔍 Analizando imagen y consultando servidor remoto..."):
            enhanced_img = apply_smart_enhancement(img_array)
            quality_analysis = analyze_image_quality_improved(enhanced_img)
            chagas_analysis = detect_chagas_bands_improved(enhanced_img)
            text_detection = detect_text_on_strip(enhanced_img)
            
            letters_detection = None
            if st.session_state.detect_letters:
                letters_detection = detect_letters_c_t_improved(enhanced_img)
            
            validated_analysis = validate_with_letters(
                chagas_analysis, quality_analysis, text_detection, letters_detection
            )
            
            features = extract_features_from_analysis({
                **chagas_analysis,
                'quality_analysis': quality_analysis,
                'text_detection': text_detection,
                'letters_detection': letters_detection,
                'validated_confidence': validated_analysis['validated_confidence'],
                'quality_score': validated_analysis['quality_score']
            })
            
            ml_prediction, ml_confidence = predict_with_model(features)
            
            final_result = validated_analysis['validated_result']
            final_confidence = validated_analysis['validated_confidence']
            correction_notes = "Sin auto-corrección aplicada"
            
            if st.session_state.auto_correction and len(st.session_state.learning_data) > 0:
                corrected_result, corrected_confidence, notes = auto_correct_with_historical_data(
                    validated_analysis, enhanced_img
                )
                final_result = corrected_result
                final_confidence = corrected_confidence
                correction_notes = notes
            
            if ml_prediction and ml_confidence > 0.5:  # Umbral más bajo
                final_result = ml_prediction
                final_confidence = ml_confidence * 100
            
            final_analysis = {
                **validated_analysis,
                'validated_result': final_result,
                'validated_confidence': final_confidence,
                'features': features,
                'ml_prediction': ml_prediction,
                'ml_confidence': ml_confidence,
                'analysis_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'image_array': enhanced_img,
                'letters_detection': letters_detection,
                'text_detection': text_detection,
                'correction_notes': correction_notes,
                'source': source
            }
        
        display_analysis_results(final_analysis, enhanced_img)
        
        st.session_state.last_analysis = final_analysis
        st.session_state.last_image = enhanced_img
        
    except Exception as e:
        st.error(f"❌ Error en análisis: {e}")

def display_analysis_results(analysis, image):
    """Muestra resultados del análisis"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Imagen Analizada")
        st.image(image, use_container_width=True)
        
        if analysis.get('correction_notes') and "auto-corrección" in analysis['correction_notes'].lower():
            st.info(f"**🔄 Auto-corrección:** {analysis['correction_notes']}")
    
    with col2:
        st.subheader("🎯 Resultados")
        
        if analysis['ml_prediction'] and analysis['ml_confidence'] > 0:
            st.info(f"🤖 ML: {analysis['ml_prediction']} ({analysis['ml_confidence']:.1%})")
        
        result_config = {
            "POSITIVE": {"icon": "🔴", "color": "#dc3545", "bg_color": "#f8d7da"},
            "NEGATIVE": {"icon": "🟢", "color": "#28a745", "bg_color": "#d4edda"},
            "WEAK POSITIVE": {"icon": "🟡", "color": "#ffc107", "bg_color": "#fff3cd"},
            "INVALID": {"icon": "🔵", "color": "#17a2b8", "bg_color": "#d1ecf1"},
            "INDETERMINADO": {"icon": "⚫", "color": "#6c757d", "bg_color": "#f8f9fa"}
        }
        
        config = result_config.get(analysis['validated_result'], result_config["INDETERMINADO"])
        
        st.markdown(f"""
        <div style='background-color: {config["bg_color"]}; padding: 20px; border-radius: 10px; border-left: 5px solid {config["color"]};'>
            <h2 style='color: {config["color"]}; margin: 0;'>{config["icon"]} {analysis['validated_result']}</h2>
            <p style='margin: 10px 0; font-size: 1.2em;'><strong>Confianza:</strong> {analysis['validated_confidence']:.1f}%</p>
            <p style='margin: 0;'><strong>Calidad:</strong> {analysis['quality_score']:.1f}/100</p>
        </div>
        """, unsafe_allow_html=True)
        
        if analysis.get('letters_detection'):
            display_letters_info(analysis['letters_detection'])
        
        if analysis.get('text_detection') and analysis['text_detection'].get('keywords'):
            st.write("**🔤 Texto detectado:**")
            st.write(f"Palabras clave: {', '.join(analysis['text_detection']['keywords'])}")
    
    st.markdown("---")
    st.subheader("📝 Evaluación Rápida")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Correcto", use_container_width=True, type="primary"):
            save_evaluation("correct", analysis, image, analysis['validated_result'])
    
    with col2:
        if st.button("❌ Incorrecto", use_container_width=True):
            st.session_state.show_correction = True
            st.rerun()
    
    if st.session_state.get('show_correction', False):
        st.markdown("---")
        st.subheader("✏️ Corrección del Resultado")
        
        correct_result = st.selectbox(
            "Selecciona el resultado correcto:",
            RESULT_CATEGORIES,
            index=RESULT_CATEGORIES.index(analysis['validated_result']) if analysis['validated_result'] in RESULT_CATEGORIES else 0
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Guardar Corrección", use_container_width=True, type="primary"):
                save_evaluation("incorrect", analysis, image, correct_result)
                st.session_state.show_correction = False
                st.rerun()
        
        with col2:
            if st.button("↩️ Cancelar", use_container_width=True):
                st.session_state.show_correction = False
                st.rerun()

def display_letters_info(letters_detection):
    """Muestra información de letras detectadas"""
    st.write("**🔤 Letras detectadas:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if letters_detection['C_detected']:
            st.success(f"✅ **Letra C detectada** (Confianza: {letters_detection['C_confidence']:.0f}%)")
        else:
            st.error("❌ **Letra C no detectada**")
    
    with col2:
        if letters_detection['T_detected']:
            st.success(f"✅ **Letra T detectada** (Confianza: {letters_detection['T_confidence']:.0f}%)")
        else:
            st.error("❌ **Letra T no detectada**")
    
    if letters_detection['letters_found']:
        st.info(f"📝 **Letras identificadas:** {', '.join(letters_detection['letters_found'])}")

def save_evaluation(evaluation_type, analysis, image, correct_result):
    """Guarda evaluación del usuario en servidor remoto - VERSIÓN MEJORADA"""
    try:
        learning_item = {
            'timestamp': datetime.now().isoformat(),
            'features': analysis['features'],
            'predicted_result': analysis['validated_result'],
            'correct_result': correct_result,
            'confidence': analysis['validated_confidence'],
            'quality_score': analysis['quality_score'],
            'evaluation_type': evaluation_type,
            'source': analysis.get('source', 'desconocido'),
            'analysis_id': analysis['analysis_id']
        }
        
        st.session_state.learning_data.append(learning_item)
        save_remote_learning_data()
        
        # ✅ FORZAR guardado de imagen incluso si hay errores
        max_retries = 3
        image_saved = False
        for attempt in range(max_retries):
            try:
                image_path = save_remote_training_image(image, analysis['analysis_id'], correct_result)
                if image_path:
                    image_saved = True
                    break
                else:
                    logger.warning(f"Reintentando guardar imagen... ({attempt + 1}/{max_retries})")
                    time.sleep(1)
            except Exception as e:
                logger.warning(f"Error guardando imagen, reintentando... ({attempt + 1}/{max_retries}): {e}")
                time.sleep(1)
        
        if not image_saved:
            st.warning("⚠️ No se pudo guardar la imagen, pero los datos se guardaron correctamente")
        
        if evaluation_type == "correct":
            st.success("✅ Evaluación guardada en servidor remoto - El sistema aprenderá de este acierto")
        else:
            st.success("✅ Corrección guardada en servidor remoto - El sistema se ajustará para mejorar")
            
        # ✅ Entrenar inmediatamente después de guardar (más agresivo)
        if len(st.session_state.learning_data) >= 2:  # Bajó el mínimo a 2
            with st.spinner("🔄 Entrenando modelo inmediatamente..."):
                accuracy = train_model()
                if accuracy:
                    st.success(f"🎯 Modelo actualizado - Precisión: {accuracy:.1%}")
        
    except Exception as e:
        st.error(f"Error guardando evaluación en remoto: {e}")

def render_evaluation_tab():
    """Pestaña de evaluación detallada"""
    st.header("📊 Evaluación y Corrección")
    
    if 'last_analysis' not in st.session_state:
        st.info("Realiza un análisis primero en la pestaña 'Análisis'")
        return
    
    analysis = st.session_state.last_analysis
    image = st.session_state.last_image
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Imagen a Evaluar")
        st.image(image, use_container_width=True)
        
        st.write("**Resultado del sistema:**")
        st.write(f"- **Predicción:** {analysis['validated_result']}")
        st.write(f"- **Confianza:** {analysis['validated_confidence']:.1f}%")
        st.write(f"- **Calidad:** {analysis['quality_score']:.1f}/100")
        
        if analysis['ml_prediction']:
            st.write(f"- **ML:** {analysis['ml_prediction']} ({analysis['ml_confidence']:.1%})")
        
        if analysis.get('correction_notes'):
            st.write(f"- **Notas:** {analysis['correction_notes']}")
    
    with col2:
        st.subheader("✏️ Evaluación Detallada")
        
        st.write("**Evaluación del resultado:**")
        
        col_correct, col_incorrect = st.columns(2)
        
        with col_correct:
            if st.button("✅ CORRECTO", use_container_width=True, type="primary"):
                save_evaluation("correct", analysis, image, analysis['validated_result'])
                st.rerun()
        
        with col_incorrect:
            if st.button("❌ INCORRECTO", use_container_width=True):
                save_evaluation("incorrect", analysis, image, analysis['validated_result'])
                st.rerun()
        
        st.markdown("---")
        st.subheader("📝 Corrección Manual")
        
        st.write("**Selecciona el resultado correcto:**")
        
        cols = st.columns(2)
        
        with cols[0]:
            if st.button("🔴 POSITIVE", use_container_width=True):
                save_training_example(analysis, image, "POSITIVE")
                st.rerun()
            
            if st.button("🟢 NEGATIVE", use_container_width=True):
                save_training_example(analysis, image, "NEGATIVE")
                st.rerun()
        
        with cols[1]:
            if st.button("🟡 WEAK POSITIVE", use_container_width=True):
                save_training_example(analysis, image, "WEAK POSITIVE")
                st.rerun()
            
            if st.button("🔵 INVALID", use_container_width=True):
                save_training_example(analysis, image, "INVALID")
                st.rerun()

def save_training_example(analysis, image, correct_result):
    """Guarda ejemplo para entrenamiento en servidor remoto"""
    try:
        learning_item = {
            'timestamp': datetime.now().isoformat(),
            'features': analysis['features'],
            'predicted_result': analysis['validated_result'],
            'correct_result': correct_result,
            'confidence': analysis['validated_confidence'],
            'quality_score': analysis['quality_score'],
            'evaluation_type': 'manual',
            'source': analysis.get('source', 'manual'),
            'analysis_id': analysis['analysis_id']
        }
        
        st.session_state.learning_data.append(learning_item)
        save_remote_learning_data()
        
        # Guardar imagen con múltiples intentos
        max_retries = 3
        for attempt in range(max_retries):
            try:
                save_remote_training_image(image, analysis['analysis_id'], correct_result)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning("No se pudo guardar la imagen después de varios intentos")
        
        st.success(f"✅ Ejemplo {correct_result} guardado para entrenamiento en servidor remoto")
        
        # Entrenar inmediatamente
        if len(st.session_state.learning_data) >= 2:
            with st.spinner("🔄 Re-entrenando modelo en servidor remoto..."):
                accuracy = train_model()
                if accuracy:
                    st.success(f"🎯 Modelo actualizado en servidor remoto - Precisión: {accuracy:.1%}")
        
    except Exception as e:
        st.error(f"Error guardando ejemplo en remoto: {e}")

def create_improved_accuracy_chart():
    """Crea un gráfico mejorado de la evolución de la precisión"""
    if not st.session_state.accuracy_history:
        st.info("No hay datos de precisión para mostrar")
        return
    
    try:
        # Crear figura con mejor estilo
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Preparar datos
        history_df = pd.DataFrame(st.session_state.accuracy_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp')
        
        # Crear gráfico con mejor estilo
        ax.plot(history_df['timestamp'], history_df['accuracy'], 
                marker='o', linewidth=2.5, markersize=8, 
                color='#2E86AB', markerfacecolor='#A23B72', 
                markeredgecolor='white', markeredgewidth=1.5,
                label='Precisión del Modelo')
        
        # Configurar límites para mejor visualización
        min_accuracy = max(0.0, history_df['accuracy'].min() - 0.1)
        max_accuracy = min(1.0, history_df['accuracy'].max() + 0.1)
        ax.set_ylim(min_accuracy, max_accuracy)
        
        # Configurar eje X para mostrar tiempos claramente
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Configurar eje Y para mostrar porcentajes
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        
        # Añadir cuadrícula para mejor lectura
        ax.grid(True, alpha=0.3)
        ax.grid(True, which='minor', alpha=0.2)
        
        # Añadir título y etiquetas
        ax.set_title('Evolución de la Precisión del Modelo', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Tiempo', fontsize=12, labelpad=10)
        ax.set_ylabel('Precisión', fontsize=12, labelpad=10)
        
        # Añadir leyenda
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        
        # Añadir área sombreada para resaltar la mejora
        ax.fill_between(history_df['timestamp'], history_df['accuracy'], 
                       alpha=0.2, color='#2E86AB')
        
        # Añadir anotaciones para puntos importantes
        if len(history_df) > 1:
            # Punto de mayor precisión
            max_idx = history_df['accuracy'].idxmax()
            max_point = history_df.loc[max_idx]
            ax.annotate(f'Máximo: {max_point["accuracy"]:.1%}', 
                       xy=(max_point['timestamp'], max_point['accuracy']),
                       xytext=(10, 30), textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', color='#A23B72'),
                       fontsize=9, color='#A23B72')
        
        # Ajustar el diseño
        plt.tight_layout()
        
        # Mostrar el gráfico en Streamlit
        st.pyplot(fig)
        
        # Mostrar estadísticas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precisión Actual", 
                     f"{history_df['accuracy'].iloc[-1]:.1%}")
        with col2:
            st.metric("Mejor Precisión", 
                     f"{history_df['accuracy'].max():.1%}")
        with col3:
            st.metric("Mejora Total", 
                     f"{(history_df['accuracy'].iloc[-1] - history_df['accuracy'].iloc[0]):.1%}")
            
    except Exception as e:
        st.error(f"Error generando gráfico: {e}")
        # Fallback: gráfico simple
        if st.session_state.accuracy_history:
            history_df = pd.DataFrame(st.session_state.accuracy_history)
            st.line_chart(history_df.set_index('timestamp')['accuracy'])

def render_learning_tab():
    """Pestaña de gestión del aprendizaje remoto - VERSIÓN MEJORADA"""
    st.header("🧠 Gestión del Aprendizaje Automático Remoto")
    
    # NUEVO: Sección de herramientas avanzadas
    st.subheader("🛠️ Herramientas Avanzadas de Entrenamiento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Entrenamiento Forzado", use_container_width=True, type="primary"):
            force_learning_from_current_data()
            
    with col2:
        if st.button("📤 Cargar Imágenes Históricas", use_container_width=True):
            upload_images_to_remote()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Estado del Modelo Remoto")
        
        st.metric("Ejemplos de Entrenamiento", len(st.session_state.learning_data))
        st.metric("Sesiones de Entrenamiento", st.session_state.training_count)
        st.metric("Auto-correcciones", st.session_state.auto_corrections)
        
        if st.session_state.accuracy_history:
            latest = st.session_state.accuracy_history[-1]
            st.metric("Precisión Actual", f"{latest['accuracy']:.1%}")
        
        remote_images = load_remote_images_list()
        st.metric("Imágenes en Servidor", len(remote_images))
        
        if st.session_state.learning_data:
            df = pd.DataFrame(st.session_state.learning_data)
            result_counts = df['correct_result'].value_counts()
            st.write("**Distribución de resultados:**")
            for result in RESULT_CATEGORIES:
                count = result_counts.get(result, 0)
                st.write(f"- {result}: {count}")
    
    with col2:
        st.subheader("🛠️ Acciones de Entrenamiento Remoto")
        
        if st.button("🎯 Entrenar Modelo Ahora", use_container_width=True):
            with st.spinner("Entrenando modelo en servidor remoto..."):
                accuracy = train_model()
                if accuracy:
                    st.success(f"✅ Modelo entrenado en servidor remoto - Precisión: {accuracy:.1%}")
                else:
                    st.error("❌ No hay suficientes datos para entrenar")
        
        if st.button("📊 Ver Todos los Datos", use_container_width=True):
            display_learning_data()
        
        if st.button("🔄 Sincronizar con Remoto", use_container_width=True):
            st.session_state.learning_data = load_remote_learning_data()
            st.session_state.accuracy_history = load_remote_accuracy_history()
            st.success("✅ Datos sincronizados con servidor remoto")
            st.rerun()
        
        if st.button("📋 Listar Imágenes Remotas", use_container_width=True):
            remote_images = load_remote_images_list()
            if remote_images:
                st.write(f"**Imágenes en servidor remoto ({len(remote_images)}):**")
                for img in remote_images[:10]:  # Mostrar primeras 10
                    st.write(f"- {img}")
                if len(remote_images) > 10:
                    st.write(f"... y {len(remote_images) - 10} más")
            else:
                st.info("No hay imágenes en el servidor remoto")
                
        if st.button("📝 Crear Encabezados Remotos", use_container_width=True):
            if create_remote_headers():
                st.rerun()
    
    # NUEVO: Gráfico mejorado de evolución de precisión
    if st.session_state.accuracy_history:
        st.subheader("📈 Evolución de la Precisión - Gráfico Mejorado")
        create_improved_accuracy_chart()

def display_learning_data():
    """Muestra los datos de aprendizaje remotos"""
    if not st.session_state.learning_data:
        st.info("No hay datos de aprendizaje en el servidor remoto")
        return
    
    df = pd.DataFrame(st.session_state.learning_data)
    
    st.subheader("📋 Datos de Aprendizaje Remotos")
    st.dataframe(df)
    
    st.subheader("📊 Estadísticas de Datos Remotos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Ejemplos", len(df))
        correct_count = len(df[df['evaluation_type'] == 'correct'])
        st.metric("Evaluaciones Correctas", correct_count)
    
    with col2:
        st.write("**Distribución de Resultados Correctos:**")
        result_counts = df['correct_result'].value_counts()
        for result in RESULT_CATEGORIES:
            count = result_counts.get(result, 0)
            st.write(f"- {result}: {count}")
    
    with col3:
        st.write("**Tipos de Evaluación:**")
        eval_counts = df['evaluation_type'].value_counts()
        st.write(eval_counts)

def render_guide_tab():
    """Pestaña de guía del sistema de aprendizaje remoto"""
    st.header("📚 Guía del Sistema de Aprendizaje Remoto")
    
    st.markdown("""
    ### 🪳 SISTEMA 100% REMOTO - TODOS LOS ARCHIVOS EN SERVIDOR
    
    **📁 ARCHIVOS GUARDADOS EN SERVIDOR:**
    
    - `registro_chagas.csv` - Datos de aprendizaje (características + resultados)
    - `chagas_model.pkl` - Modelo de Machine Learning entrenado
    - `accuracy_history.json` - Historial de precisión del modelo
    - `training_images/` - Directorio con todas las imágenes de entrenamiento
    
    **🔄 FLUJO DE APRENDIZAJE REMOTO:**
    
    1. **📸 Captura**: Toma foto con cámara o sube archivo
    2. **🔍 Análisis**: Sistema analiza imagen localmente
    3. **🌐 Consulta remota**: Busca imágenes similares en servidor
    4. **🤖 ML + Auto-corrección**: Combina ML local con histórico remoto
    5. **✅ Evaluación**: Usuario confirma o corrige resultado
    6. **💾 Guardado remoto**: Todo se guarda en servidor inmediatamente
    
    **🎯 BENEFICIOS DEL SISTEMA REMOTO:**
    
    - **Persistencia total**: No se pierden datos al reiniciar
    - **Colaborativo**: Múltiples usuarios contribuyen al aprendizaje
    - **Escalable**: El servidor maneja todo el almacenamiento
    - **Seguro**: Datos centralizados y respaldados
    - **Consistente**: Todos usan el mismo modelo actualizado
    
    **🔧 ARCHIVOS EN SERVIDOR REMOTO:**
    """)
    
    # Mostrar estructura de archivos remotos
    st.code("""
    /home/POLANCO6/CHAGAS/
    ├── registro_chagas.csv          # Datos de aprendizaje
    ├── chagas_model.pkl             # Modelo ML entrenado  
    ├── accuracy_history.json        # Historial de precisión
    └── training_images/             # Imágenes de entrenamiento
        ├── 20240115103000_POSITIVE_20240115103015.jpg
        ├── 20240115103245_NEGATIVE_20240115103250.jpg
        ├── 20240115103520_WEAK_POSITIVE_20240115103525.jpg
        └── ...
    """, language="bash")
    
    st.success("""
    **🚀 CONSEJO RÁPIDO:** 
    El sistema funciona completamente desde el servidor remoto. 
    No se guarda nada localmente - todo está centralizado para máximo aprendizaje colaborativo.
    """)

# Pestañas principales
def main():
    st.set_page_config(
        page_title="Analizador Chagas con Aprendizaje Automático Remoto",
        page_icon="🪳",  # CAMBIADO: Chinche besucona en lugar de mosquito
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    initialize_learning_system()
    check_https_status()
    
    st.title("🪳 Analizador Chagas con APRENDIZAJE AUTOMÁTICO REMOTO")  # CAMBIADO: Icono de chinche
    st.markdown("### **Sistema que mejora con cada evaluación - Almacenamiento 100% Remoto**")
    
    with st.sidebar:
        st.header("🧠 Sistema de Aprendizaje Remoto")
        
        st.metric("Ejemplos de Entrenamiento", len(st.session_state.learning_data))
        st.metric("Sesiones de Entrenamiento", st.session_state.training_count)
        st.metric("Auto-correcciones", st.session_state.auto_corrections)
        
        if st.session_state.accuracy_history:
            latest_accuracy = st.session_state.accuracy_history[-1]['accuracy']
            st.metric("Precisión Actual", f"{latest_accuracy:.1%}")
        
        progress = min(len(st.session_state.learning_data) / 20, 1.0)
        st.progress(progress)
        st.caption(f"Progreso: {len(st.session_state.learning_data)}/20 ejemplos")
        
        st.markdown("---")
        st.header("⚙️ Configuración")
        st.session_state.detect_letters = st.checkbox("Detección de Letras C/T", value=True)
        st.session_state.auto_correction = st.checkbox("Auto-corrección con histórico remoto", value=True)
        st.session_state.min_confidence = st.slider("Confianza Mínima (%)", 50, 90, 60)
        
        if st.button("🔄 Re-entrenar Modelo", use_container_width=True):
            accuracy = train_model()
            if accuracy:
                st.success(f"✅ Modelo re-entrenado - Precisión: {accuracy:.1%}")
            else:
                st.error("❌ No hay suficientes datos para entrenar")
        
        if st.button("🧹 Limpiar Datos Corruptos", use_container_width=True):
            clean_corrupted_data()
            st.rerun()

        st.markdown("---")
        st.header("🌐 Almacenamiento Remoto")
        st.info(f"**Servidor:** {REMOTE_HOST}:{REMOTE_PORT}")
        st.info(f"**Usuario:** {REMOTE_USER}")
        st.info(f"**Directorio:** {REMOTE_DIR}")
        
        remote_images = load_remote_images_list()
        st.info(f"**Imágenes en remoto:** {len(remote_images)}")
        
        if st.button("🔄 Sincronizar con Remoto", use_container_width=True):
            st.session_state.learning_data = load_remote_learning_data()
            st.session_state.accuracy_history = load_remote_accuracy_history()
            st.success("✅ Datos sincronizados con servidor remoto")
            st.rerun()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🪳 Análisis", "📊 Evaluación", "🧠 Aprendizaje", "📚 Guía"  # CAMBIADO: Icono de chinche
    ])
    
    with tab1:
        render_capture_tab()
    with tab2:
        render_evaluation_tab()
    with tab3:
        render_learning_tab()
    with tab4:
        render_guide_tab()

if __name__ == "__main__":
    main()
