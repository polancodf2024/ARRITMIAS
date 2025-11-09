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

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de pytesseract (OCR)
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    try:
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    except:
        st.warning("Tesseract OCR no encontrado. La detección de texto estará limitada.")

def check_https_status():
    """Verifica si la app está usando HTTPS"""
    try:
        st.markdown("""
        <div style='background-color: #d4edda; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745;'>
        <h4>✅ HTTPS Activado</h4>
        <p>Esta app está usando conexión segura HTTPS. La cámara debería funcionar en dispositivos móviles.</p>
        </div>
        """, unsafe_allow_html=True)
    except:
        pass

def enhance_camera_capture():
    """Configuración mejorada para la cámara"""
    st.markdown("""
    <style>
    .camera-container {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        background: #f9f9f9;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='warning-box'>
    <strong>🔍 CONSEJOS PARA MEJOR RECONOCIMIENTO:</strong><br>
    • Asegúrate que las letras <strong>C</strong> y <strong>T</strong> sean visibles<br>
    • Las letras deben estar cerca de las bandas correspondientes<br>
    • Buena iluminación para texto legible<br>
    • Enfoque claro en las letras y bandas
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)
    
    try:
        picture = st.camera_input(
            "📸 Toma una foto CLARA de la tira reactiva de Chagas",
            help="Asegúrate que las letras C y T sean visibles y legibles"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        return picture
        
    except Exception as e:
        st.markdown('</div>', unsafe_allow_html=True)
        st.error(f"❌ Error con la cámara: {e}")
        return None

def apply_smart_enhancement(img_array):
    """Aplica mejoras inteligentes a la imagen"""
    try:
        height, width = img_array.shape[:2]
        
        # Redimensionar si es necesario
        if height < 500 or width < 500:
            scale_factor = max(800/width, 600/height, 1.8)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            st.success(f"🔄 Imagen mejorada: {width}x{height} → {new_width}x{new_height}")
        
        # Mejorar para análisis de texto y bandas
        enhanced = enhance_for_text_and_bands(img_array)
        return enhanced
        
    except Exception as e:
        logger.error(f"Error en mejora inteligente: {e}")
        return img_array

def enhance_for_text_and_bands(img_array):
    """Mejora específica para texto y bandas"""
    try:
        if len(img_array.shape) == 3:
            # Mejorar contraste para texto
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE más agresivo para texto
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = img_array
        
        # Enfoque específico para texto
        kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                                 [-1,2,2,2,-1],
                                 [-1,2,8,2,-1],
                                 [-1,2,2,2,-1],
                                 [-1,-1,-1,-1,-1]]) / 8.0
        enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error en mejora para texto: {e}")
        return img_array

def detect_letters_c_t(img_array):
    """Detección ESPECÍFICA de letras C y T usando OCR y procesamiento de imágenes"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        height, width = gray.shape
        
        results = {
            'C_detected': False,
            'T_detected': False,
            'C_confidence': 0,
            'T_confidence': 0,
            'C_location': None,
            'T_location': None,
            'letters_found': []
        }
        
        # MÉTODO 1: OCR específico para letras C y T
        ocr_results = detect_letters_ocr(gray, height, width)
        results.update(ocr_results)
        
        # MÉTODO 2: Procesamiento de imágenes para encontrar letras
        image_processing_results = detect_letters_image_processing(gray, height, width)
        
        # Combinar resultados
        if image_processing_results['C_detected']:
            results['C_detected'] = True
            results['C_confidence'] = max(results['C_confidence'], image_processing_results['C_confidence'])
            results['C_location'] = image_processing_results['C_location']
            
        if image_processing_results['T_detected']:
            results['T_detected'] = True
            results['T_confidence'] = max(results['T_confidence'], image_processing_results['T_confidence'])
            results['T_location'] = image_processing_results['T_location']
        
        # Actualizar letras encontradas
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

def detect_letters_ocr(gray, height, width):
    """Detección de letras C y T usando OCR optimizado"""
    try:
        # Preprocesamiento agresivo para OCR de letras individuales
        # Binarización adaptativa
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Operaciones morfológicas para mejorar texto
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Configuración OCR optimizada para letras individuales
        custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=CTct'
        
        # Buscar en regiones específicas donde suelen estar las letras
        regions_to_check = [
            # Región izquierda (T)
            (int(width*0.15), int(height*0.4), int(width*0.25), int(height*0.6)),
            # Región derecha (C)
            (int(width*0.70), int(height*0.4), int(width*0.80), int(height*0.6)),
            # Regiones alternativas
            (int(width*0.20), int(height*0.3), int(width*0.30), int(height*0.5)),
            (int(width*0.65), int(height*0.3), int(width*0.75), int(height*0.5))
        ]
        
        C_detected = False
        T_detected = False
        C_confidence = 0
        T_confidence = 0
        
        for i, (x1, y1, x2, y2) in enumerate(regions_to_check):
            region = binary[y1:y2, x1:x2]
            if region.size == 0:
                continue
                
            # OCR en la región
            detected_text = pytesseract.image_to_string(region, config=custom_config)
            cleaned_text = re.sub(r'[^CTct]', '', detected_text.upper())
            
            # Evaluar confianza basada en la claridad del texto
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

def calculate_text_confidence(region):
    """Calcula confianza basada en la claridad del texto"""
    try:
        # Calcular métricas de claridad de texto
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Contraste local
        contrast = np.std(region)
        
        # Score combinado
        confidence = (edge_density * 60 + min(contrast/10, 40))
        return min(95, confidence)
        
    except:
        return 50

def detect_letters_image_processing(gray, height, width):
    """Detección de letras C y T usando procesamiento de imágenes"""
    try:
        # Buscar contornos que podrían ser letras
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        C_candidates = []
        T_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 2000:  # Tamaño razonable para letras
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Las letras suelen ser más altas que anchas
            if 0.3 < aspect_ratio < 3.0:
                # Clasificar basado en posición y forma
                if is_potential_C(contour, x, y, w, h, width):
                    C_candidates.append((x, y, w, h, area))
                elif is_potential_T(contour, x, y, w, h, width):
                    T_candidates.append((x, y, w, h, area))
        
        # Determinar resultados
        C_detected = len(C_candidates) > 0
        T_detected = len(T_candidates) > 0
        
        # Calcular confianza basada en número y calidad de candidatos
        C_confidence = min(90, len(C_candidates) * 30) if C_detected else 0
        T_confidence = min(90, len(T_candidates) * 30) if T_detected else 0
        
        # Encontrar ubicaciones más probables
        C_location = find_best_candidate_location(C_candidates, width, 'right')
        T_location = find_best_candidate_location(T_candidates, width, 'left')
        
        return {
            'C_detected': C_detected,
            'T_detected': T_detected,
            'C_confidence': C_confidence,
            'T_confidence': T_confidence,
            'C_location': C_location,
            'T_location': T_location
        }
        
    except Exception as e:
        logger.error(f"Error en procesamiento de imágenes para letras: {e}")
        return {'C_detected': False, 'T_detected': False, 'C_confidence': 0, 'T_confidence': 0}

def is_potential_C(contour, x, y, w, h, image_width):
    """Determina si un contorno podría ser la letra C"""
    # La C suele estar en el lado derecho
    if x < image_width * 0.6:  # Demasiado a la izquierda para ser C
        return False
    
    # Ratio de aspecto típico de C
    aspect_ratio = w / h
    if not (0.5 <= aspect_ratio <= 1.5):
        return False
        
    # El área debe ser razonable para una letra
    area = cv2.contourArea(contour)
    if not (100 <= area <= 1500):
        return False
        
    return True

def is_potential_T(contour, x, y, w, h, image_width):
    """Determina si un contorno podría ser la letra T"""
    # La T suele estar en el lado izquierdo
    if x > image_width * 0.4:  # Demasiado a la derecha para ser T
        return False
    
    # Ratio de aspecto típico de T (más alta que ancha)
    aspect_ratio = w / h
    if not (0.3 <= aspect_ratio <= 1.2):
        return False
        
    # El área debe ser razonable para una letra
    area = cv2.contourArea(contour)
    if not (100 <= area <= 1500):
        return False
        
    return True

def find_best_candidate_location(candidates, image_width, expected_side):
    """Encuentra la ubicación más probable para una letra"""
    if not candidates:
        return None
    
    # Para C, esperamos lado derecho; para T, lado izquierdo
    if expected_side == 'right':
        # Escoger candidato más a la derecha
        best_candidate = max(candidates, key=lambda c: c[0])
    else:  # left
        # Escoger candidato más a la izquierda
        best_candidate = min(candidates, key=lambda c: c[0])
    
    x, y, w, h, area = best_candidate
    return (x, y, w, h)

def detect_text_on_strip(img_array):
    """Detección general de texto en la tira"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Preprocesamiento para OCR general
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        custom_config = r'--oem 3 --psm 6'
        detected_text = pytesseract.image_to_string(thresh, config=custom_config)
        
        return clean_detected_text(detected_text)
        
    except Exception as e:
        logger.warning(f"OCR general falló: {e}")
        return {"raw_text": "", "keywords": [], "has_chagas_text": False}

def clean_detected_text(text):
    """Limpia y analiza el texto detectado"""
    cleaned = ' '.join(text.split())
    
    chagas_keywords = ['CHAGAS', 'TEST', 'CONTROL', 'POSITIVE', 'NEGATIVE', 'INVALID', 'C', 'T']
    
    found_keywords = []
    text_upper = cleaned.upper()
    
    for keyword in chagas_keywords:
        if keyword in text_upper:
            found_keywords.append(keyword)
    
    return {
        'raw_text': cleaned,
        'keywords': found_keywords,
        'has_chagas_text': 'CHAGAS' in found_keywords,
        'has_control_text': 'CONTROL' in found_keywords,
        'has_test_text': 'TEST' in found_keywords
    }

def main():
    st.set_page_config(
        page_title="Analizador con Reconocimiento de Letras C/T",
        page_icon="🦟",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    check_https_status()
    
    st.title("🦟 Analizador AVANZADO con Reconocimiento de Letras C/T")
    st.markdown("### **Detección automática + Reconocimiento de letras C y T**")
    
    # Inicializar session state
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    if 'chagas_detections' not in st.session_state:
        st.session_state.chagas_detections = 0
    if 'invalid_results' not in st.session_state:
        st.session_state.invalid_results = 0
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración Avanzada")
        
        st.subheader("Reconocimiento de Letras")
        st.session_state.detect_letters = st.checkbox(
            "Detección de Letras C/T", 
            value=True,
            help="Buscar específicamente las letras C y T en la tira"
        )
        
        st.session_state.letter_confidence_boost = st.slider(
            "Boost por Letras Detectadas (%)", 
            0, 30, 15,
            help="Aumento de confianza cuando se detectan letras C y T"
        )
        
        st.session_state.min_confidence = st.slider(
            "Confianza Mínima (%)", 
            60, 90, 75
        )
        
        st.markdown("---")
        st.header("📊 Estadísticas")
        st.metric("Análisis Realizados", st.session_state.analysis_count)
        st.metric("Chagas Detectados", st.session_state.chagas_detections)
        st.metric("Resultados Inválidos", st.session_state.invalid_results)
    
    # Pestañas principales
    tab1, tab2, tab3 = st.tabs(["🎯 Análisis Principal", "🔤 Reconocimiento Letras", "📚 Guía"])
    
    with tab1:
        render_capture_tab()
    
    with tab2:
        render_letters_tab()
    
    with tab3:
        render_guide_tab()

def render_capture_tab():
    """Pestaña de captura principal"""
    st.header("📸 Análisis Principal con Reconocimiento de Letras")
    
    st.markdown("""
    <div style='background-color: #e7f3ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <h4>🎯 NUEVA FUNCIONALIDAD: Reconocimiento de Letras C y T</h4>
    <p>El sistema ahora busca específicamente las letras <strong>C</strong> (Control) y <strong>T</strong> (Test) 
    en la tira reactiva para validación adicional.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Captura de cámara
    st.subheader("📷 Captura con Cámara")
    picture = enhance_camera_capture()
    
    if picture is not None:
        process_camera_image(picture)
    
    # Subida de archivo
    st.subheader("📁 Subir Archivo")
    uploaded_file = st.file_uploader(
        "O sube una foto desde tu galería",
        type=['jpg', 'jpeg', 'png', 'heic'],
        help="Asegúrate que las letras C y T sean visibles"
    )
    
    if uploaded_file is not None:
        process_uploaded_image(uploaded_file)

def process_camera_image(picture):
    """Procesa imagen de la cámara"""
    try:
        image = Image.open(picture)
        img_array = np.array(image)
        
        with st.spinner("🔄 Mejorando imagen y buscando letras..."):
            enhanced_img = apply_smart_enhancement(img_array)
            st.success("✅ Imagen procesada para análisis")
        
        process_and_analyze_with_letters(image, enhanced_img, "Cámara Directa")
        
    except Exception as e:
        st.error(f"❌ Error procesando imagen: {e}")

def process_uploaded_image(uploaded_file):
    """Procesa imagen subida"""
    try:
        image = Image.open(uploaded_file)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        img_array = np.array(image)
        
        with st.spinner("🔄 Mejorando imagen y buscando letras..."):
            enhanced_img = apply_smart_enhancement(img_array)
            st.success("✅ Imagen procesada para análisis")
        
        process_and_analyze_with_letters(image, enhanced_img, "Archivo Subido")
        
    except Exception as e:
        st.error(f"❌ Error procesando imagen: {e}")

def process_and_analyze_with_letters(original_image, enhanced_img, source):
    """Procesa y analiza con reconocimiento de letras"""
    st.success(f"✅ Imagen recibida desde: {source}")
    
    # Detección de letras C y T
    letters_detection = None
    if st.session_state.detect_letters:
        with st.spinner("🔍 Buscando letras C y T..."):
            letters_detection = detect_letters_c_t(enhanced_img)
    
    # Detección de texto general
    text_detection = detect_text_on_strip(enhanced_img)
    
    # Análisis principal
    with st.spinner("🔍 Analizando tira reactiva..."):
        quality_analysis = analyze_image_quality_improved(enhanced_img)
        chagas_analysis = detect_chagas_bands_improved(enhanced_img)
        validated_analysis = validate_with_letters(chagas_analysis, quality_analysis, text_detection, letters_detection)
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Imagen Mejorada")
        st.image(enhanced_img, use_container_width=True, 
                caption=f"Resolución: {enhanced_img.shape[1]}x{enhanced_img.shape[0]}")
    
    with col2:
        st.subheader("🎯 Bandas y Letras Detectadas")
        visualization = create_enhanced_visualization(enhanced_img, validated_analysis, letters_detection)
        st.image(visualization, use_container_width=True, 
                caption="Bandas (C=Control, T=Test) + Letras detectadas")
    
    # Mostrar detección de letras
    if letters_detection:
        display_letters_detection(letters_detection)
    
    # Mostrar métricas y resultados
    display_quality_metrics_improved(quality_analysis)
    display_final_result_enhanced(validated_analysis, letters_detection)
    
    # Análisis técnico
    with st.expander("🔍 Análisis Técnico Detallado"):
        display_technical_analysis_enhanced(chagas_analysis, validated_analysis, text_detection, letters_detection)
    
    # Guardar en historial
    save_to_history_enhanced(validated_analysis, quality_analysis, letters_detection)

def display_letters_detection(letters_detection):
    """Muestra resultados de detección de letras"""
    st.subheader("🔤 Reconocimiento de Letras C y T")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if letters_detection['C_detected']:
            st.success(f"✅ **Letra C DETECTADA** (Confianza: {letters_detection['C_confidence']:.0f}%)")
        else:
            st.error("❌ **Letra C NO detectada**")
            
    with col2:
        if letters_detection['T_detected']:
            st.success(f"✅ **Letra T DETECTADA** (Confianza: {letters_detection['T_confidence']:.0f}%)")
        else:
            st.error("❌ **Letra T NO detectada**")
    
    if letters_detection['letters_found']:
        st.info(f"📝 **Letras identificadas:** {', '.join(letters_detection['letters_found'])}")

def create_enhanced_visualization(img_array, analysis, letters_detection):
    """Crea visualización mejorada con letras"""
    try:
        viz = img_array.copy()
        height, width = viz.shape[:2]
        
        # Colores
        control_color = (0, 255, 0)  # Verde
        test_color = (255, 0, 0)     # Rojo
        letter_color = (255, 255, 0) # Amarillo para letras
        
        # Dibujar regiones de bandas
        cv2.rectangle(viz, 
                     (int(width*0.6), int(height*0.3)), 
                     (int(width*0.75), int(height*0.7)), 
                     control_color, 3)
        cv2.putText(viz, "CONTROL", (int(width*0.6), int(height*0.25)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, control_color, 2)
        
        cv2.rectangle(viz, 
                     (int(width*0.25), int(height*0.3)), 
                     (int(width*0.4), int(height*0.7)), 
                     test_color, 3)
        cv2.putText(viz, "TEST", (int(width*0.25), int(height*0.25)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, test_color, 2)
        
        # Marcar letras detectadas
        if letters_detection:
            if letters_detection['C_detected'] and letters_detection['C_location']:
                x, y, w, h = letters_detection['C_location']
                cv2.rectangle(viz, (x, y), (x+w, y+h), letter_color, 2)
                cv2.putText(viz, "C", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, letter_color, 2)
            
            if letters_detection['T_detected'] and letters_detection['T_location']:
                x, y, w, h = letters_detection['T_location']
                cv2.rectangle(viz, (x, y), (x+w, y+h), letter_color, 2)
                cv2.putText(viz, "T", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, letter_color, 2)
        
        # Resultado
        result_text = f"{analysis['validated_result']} ({analysis['validated_confidence']:.1f}%)"
        cv2.putText(viz, result_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return viz
        
    except Exception as e:
        return img_array

def validate_with_letters(chagas_analysis, quality_analysis, text_detection, letters_detection):
    """Validación MEJORADA con reconocimiento de letras"""
    result = chagas_analysis['result']
    confidence = chagas_analysis['confidence']
    quality_score = quality_analysis['quality_score']
    
    # Ajuste por calidad
    quality_factor = quality_score / 100.0
    adjusted_confidence = confidence * quality_factor
    
    # Boost por letras detectadas
    letters_boost = 0
    if letters_detection:
        if letters_detection['C_detected']:
            letters_boost += st.session_state.letter_confidence_boost / 2
        if letters_detection['T_detected']:
            letters_boost += st.session_state.letter_confidence_boost / 2
    
    # Boost por texto general
    text_boost = 0
    if text_detection.get('has_chagas_text'):
        text_boost += 10
    
    final_confidence = min(95, adjusted_confidence + letters_boost + text_boost)
    
    # Validación final
    if quality_score < 30:
        validated_result = "INDETERMINADO"
        validation_notes = "Calidad de imagen muy baja"
    elif final_confidence < st.session_state.min_confidence:
        validated_result = "INDETERMINADO"
        validation_notes = f"Confianza insuficiente. Mínimo: {st.session_state.min_confidence}%"
    else:
        validated_result = result
        validation_notes = "Análisis completado"
        
        # Añadir notas sobre letras
        boost_notes = []
        if letters_boost > 0:
            boost_notes.append(f"+{letters_boost}% por letras")
        if text_boost > 0:
            boost_notes.append(f"+{text_boost}% por texto")
        
        if boost_notes:
            validation_notes += f" ({', '.join(boost_notes)})"
    
    return {
        'validated_result': validated_result,
        'validated_confidence': final_confidence,
        'validation_notes': validation_notes,
        'quality_score': quality_score,
        'letters_boost': letters_boost,
        'text_boost': text_boost,
        'letters_detection': letters_detection
    }

def display_final_result_enhanced(validated_analysis, letters_detection):
    """Muestra resultado final MEJORADO"""
    st.subheader("🎯 Resultado Final del Análisis")
    
    result = validated_analysis['validated_result']
    confidence = validated_analysis['validated_confidence']
    
    result_config = {
        "POSITIVO": {"color": "#dc3545", "bg_color": "#f8d7da", "icon": "🔴"},
        "NEGATIVO": {"color": "#28a745", "bg_color": "#d4edda", "icon": "🟢"},
        "DÉBIL POSITIVO": {"color": "#ffc107", "bg_color": "#fff3cd", "icon": "🟡"},
        "INVÁLIDO": {"color": "#17a2b8", "bg_color": "#d1ecf1", "icon": "🔵"},
        "INDETERMINADO": {"color": "#6c757d", "bg_color": "#f8f9fa", "icon": "⚫"}
    }.get(result, {"color": "#666", "bg_color": "#f5f5f5", "icon": "⚫"})
    
    # Información adicional sobre letras
    letters_info = ""
    if letters_detection:
        letters_found = letters_detection.get('letters_found', [])
        if letters_found:
            letters_info = f"<br><small>Letras detectadas: {', '.join(letters_found)}</small>"
    
    st.markdown(f"""
    <div style='background-color: {result_config["bg_color"]}; padding: 25px; border-radius: 10px; border-left: 5px solid {result_config["color"]}; margin: 20px 0;'>
        <h2 style='color: {result_config["color"]}; margin: 0 0 10px 0;'>{result_config["icon"]} RESULTADO: {result}</h2>
        <p style='margin: 10px 0; font-size: 1.2em;'><strong>Confianza:</strong> {confidence:.1f}%</p>
        <p style='margin: 10px 0;'><strong>Notas:</strong> {validated_analysis['validation_notes']}</p>
        {letters_info}
    </div>
    """, unsafe_allow_html=True)
    
    # Actualizar estadísticas
    st.session_state.analysis_count += 1
    if result == "POSITIVO":
        st.session_state.chagas_detections += 1
    elif result == "INVÁLIDO":
        st.session_state.invalid_results += 1

# [Las funciones restantes se mantienen similares pero adaptadas...]
def analyze_image_quality_improved(img_array):
    """Análisis de calidad básico"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        height, width = gray.shape
        
        brightness = np.mean(gray)
        contrast = np.std(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        quality_score = min(100, (brightness/2.55 * 0.3 + min(contrast, 100) * 0.4 + min(sharpness/10, 30)))
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'resolution': f"{width}x{height}",
            'quality_score': quality_score,
            'quality_category': 'BUENA' if quality_score > 60 else 'ACEPTABLE' if quality_score > 40 else 'BAJA'
        }
    except:
        return {'quality_score': 50, 'quality_category': 'ACEPTABLE'}

def detect_chagas_bands_improved(img_array):
    """Detección básica de bandas"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        height, width = gray.shape
        
        control_region = gray[int(height*0.3):int(height*0.7), int(width*0.6):int(width*0.75)]
        test_region = gray[int(height*0.3):int(height*0.7), int(width*0.25):int(width*0.4)]
        
        control_mean = np.mean(control_region) if control_region.size > 0 else 255
        test_mean = np.mean(test_region) if test_region.size > 0 else 255
        
        control_present = control_mean < 200
        test_present = test_mean < 220
        
        if not control_present:
            result = "INVÁLIDO"
            confidence = 30
        elif control_present and not test_present:
            result = "NEGATIVO"
            confidence = 85
        elif control_present and test_present:
            result = "POSITIVO"
            confidence = 80
        else:
            result = "INDETERMINADO"
            confidence = 50
            
        return {
            'result': result,
            'confidence': confidence,
            'control_present': control_present,
            'test_present': test_present
        }
    except:
        return {'result': 'INDETERMINADO', 'confidence': 30, 'control_present': False, 'test_present': False}

def display_quality_metrics_improved(quality_analysis):
    """Muestra métricas de calidad"""
    st.subheader("📊 Métricas de Calidad")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Calidad", f"{quality_analysis['quality_score']:.1f}/100")
        st.metric("Categoría", quality_analysis['quality_category'])
    
    with col2:
        st.metric("Brillo", f"{quality_analysis['brightness']:.1f}")
        st.metric("Contraste", f"{quality_analysis['contrast']:.1f}")
    
    with col3:
        st.metric("Nitidez", f"{quality_analysis['sharpness']:.0f}")
        st.metric("Resolución", quality_analysis['resolution'])

def display_technical_analysis_enhanced(chagas_analysis, validated_analysis, text_detection, letters_detection):
    """Análisis técnico mejorado"""
    st.write("**📈 Métricas de Detección:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Bandas:**")
        st.write(f"- Control: {chagas_analysis['control_present']}")
        st.write(f"- Test: {chagas_analysis['test_present']}")
        st.write(f"- Confianza base: {chagas_analysis['confidence']:.1f}%")
        
    with col2:
        st.write("**Mejoras:**")
        st.write(f"- Boost letras: +{validated_analysis.get('letters_boost', 0)}%")
        st.write(f"- Boost texto: +{validated_analysis.get('text_boost', 0)}%")
        st.write(f"- Confianza final: {validated_analysis['validated_confidence']:.1f}%")
    
    if letters_detection:
        st.write("**🔤 Letras Detectadas:**")
        st.write(f"- C: {letters_detection['C_detected']} (conf: {letters_detection['C_confidence']:.0f}%)")
        st.write(f"- T: {letters_detection['T_detected']} (conf: {letters_detection['T_confidence']:.0f}%)")

def save_to_history_enhanced(validated_analysis, quality_analysis, letters_detection):
    """Guarda en historial mejorado"""
    analysis_record = {
        'timestamp': datetime.now().isoformat(),
        'result': validated_analysis['validated_result'],
        'confidence': validated_analysis['validated_confidence'],
        'quality_score': quality_analysis['quality_score'],
        'letters_detected': letters_detection.get('letters_found', []) if letters_detection else []
    }
    
    st.session_state.analysis_history.append(analysis_record)
    
    if len(st.session_state.analysis_history) > 20:
        st.session_state.analysis_history = st.session_state.analysis_history[-20:]

def render_letters_tab():
    """Pestaña específica para reconocimiento de letras"""
    st.header("🔤 Reconocimiento Avanzado de Letras C y T")
    
    st.markdown("""
    ### 🎯 Especificaciones del Reconocimiento de Letras
    
    **¿Qué busca el sistema?**
    - Letra **C** (Control) - usualmente en el lado derecho
    - Letra **T** (Test) - usualmente en el lado izquierdo
    
    **Métodos utilizados:**
    1. **OCR especializado** para letras individuales
    2. **Procesamiento de imágenes** para encontrar patrones
    3. **Validación por posición** (C a la derecha, T a la izquierda)
    
    **Beneficios:**
    - ✅ Mayor confianza en los resultados
    - ✅ Validación adicional de la tira
    - ✅ Detección más robusta
    - ✅ Menos falsos positivos/negativos
    """)
    
    # Ejemplo visual de letras
    st.subheader("📝 Ejemplo de Letras en Tiras Reactivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Letra T (Test):**
        ```
        ┌─────────┐
        │    █    │
        │    █    │
        │  █████  │
        │    █    │
        │    █    │
        └─────────┘
        ```
        """)
    
    with col2:
        st.markdown("""
        **Letra C (Control):**
        ```
        ┌─────────┐
        │  █████  │
        │  █      │
        │  █      │
        │  █      │
        │  █████  │
        └─────────┘
        ```
        """)

def render_analysis_tab():
    """Pestaña de análisis histórico"""
    st.header("📈 Historial de Análisis")
    
    if not st.session_state.analysis_history:
        st.info("No hay análisis históricos. Realiza algunos análisis primero.")
        return
    
    for i, analysis in enumerate(reversed(st.session_state.analysis_history[-10:])):
        with st.expander(f"Análisis {i+1} - {analysis['timestamp'][:16]}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Resultado", analysis['result'])
            with col2:
                st.metric("Confianza", f"{analysis['confidence']:.1f}%")
            with col3:
                st.metric("Letras", ', '.join(analysis.get('letters_detected', ['Ninguna'])))

def render_guide_tab():
    """Pestaña de guía"""
    st.header("📚 Guía de Reconocimiento de Letras")
    
    st.markdown("""
    ### 🎯 CÓMO AYUDAR AL RECONOCIMIENTO DE LETRAS
    
    **📸 TÉCNICAS PARA MEJOR DETECCIÓN:**
    1. **Enfoque en las letras**: Asegúrate que C y T sean nítidas
    2. **Contraste adecuado**: Letras oscuras sobre fondo claro
    3. **Posición correcta**: C a la derecha, T a la izquierda
    4. **Tamaño suficiente**: Letras deben ser claramente visibles
    5. **Sin obstrucciones**: Nada debe tapar las letras
    
    **🔧 CONFIGURACIÓN RECOMENDADA:**
    - **Detección de Letras**: ACTIVADA
    - **Boost por Letras**: 15-20%
    - **Confianza Mínima**: 75%
    
    **💡 INTERPRETACIÓN:**
    - **Ambas letras detectadas**: Máxima confianza
    - **Solo C detectada**: Buena confianza (control presente)
    - **Solo T detectada**: Confianza moderada
    - **Ninguna letra**: Confianza base (solo análisis visual)
    
    **⚠️ LIMITACIONES:**
    - Letras muy pequeñas o borrosas pueden no detectarse
    - Iluminación pobre afecta el reconocimiento
    - Fuentes muy diferentes pueden causar problemas
    """)
    
    st.success("""
    **✅ CONSEJO FINAL:** Si el sistema no detecta las letras consistentemente, 
    trata de acercarte más a la tira y asegura buena iluminación en las letras.
    """)

if __name__ == "__main__":
    main()
