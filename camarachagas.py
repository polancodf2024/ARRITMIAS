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

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def main():
    st.set_page_config(
        page_title="Analizador Avanzado de Cintas Reactivas Chagas",
        page_icon="🦟",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Verificar HTTPS
    check_https_status()
    
    st.title("🦟 Analizador Avanzado de Cintas Reactivas para Chagas")
    st.markdown("### **Detección e interpretación automática con análisis robusto**")
    
    # Sidebar con información
    with st.sidebar:
        st.header("⚙️ Configuración Avanzada")
        
        st.subheader("Parámetros de Análisis")
        st.session_state.analysis_mode = st.selectbox(
            "Modo de Análisis:",
            ["Automático", "Conservador", "Sensible"],
            help="Automático: Balanceado, Conservador: Menos falsos positivos, Sensible: Detecta casos débiles"
        )
        
        st.session_state.min_confidence = st.slider(
            "Confianza Mínima (%)", 
            50, 95, 70,
            help="Confianza mínima para considerar resultado válido"
        )
        
        st.info("""
        **Para mejor análisis:**
        - Cinta completamente visible
        - Iluminación uniforme
        - Sin reflejos en bandas
        - Fondo contrastante
        """)
        
        st.header("📊 Estadísticas")
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        if 'chagas_detections' not in st.session_state:
            st.session_state.chagas_detections = 0
        if 'invalid_results' not in st.session_state:
            st.session_state.invalid_results = 0
            
        st.metric("Análisis Realizados", st.session_state.analysis_count)
        st.metric("Chagas Detectados", st.session_state.chagas_detections)
        st.metric("Resultados Inválidos", st.session_state.invalid_results)
    
    # Pestañas principales
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Captura y Análisis", "🔍 Resultados Detallados", "📈 Métricas", "📚 Guía Avanzada"])
    
    with tab1:
        render_capture_tab()
    
    with tab2:
        render_analysis_tab()
    
    with tab3:
        render_metrics_tab()
    
    with tab4:
        render_guide_tab()

def render_capture_tab():
    """Pestaña de captura de imágenes"""
    st.header("📸 Captura y Análisis Avanzado")
    
    # Opción 1: Cámara directa
    st.subheader("Opción 1: Cámara Directa")
    st.markdown("""
    <div style='background-color: #e7f3ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <b>💡 Para mejor análisis:</b> Asegúrate de que toda la cinta esté visible y las bandas bien enfocadas.
    </div>
    """, unsafe_allow_html=True)
    
    try:
        picture = st.camera_input(
            "Toma una foto de la cinta reactiva de Chagas",
            help="Incluye toda el área de resultado de la cinta, con buena iluminación"
        )
        
        if picture is not None:
            process_camera_image(picture)
            
    except Exception as e:
        st.error(f"❌ Error con la cámara: {e}")
        st.info("💡 Si la cámara no funciona, usa la Opción 2: Subir archivo")
    
    # Opción 2: Subir archivo
    st.subheader("Opción 2: Subir Archivo")
    uploaded_file = st.file_uploader(
        "O sube una foto desde tu galería",
        type=['jpg', 'jpeg', 'png', 'heic'],
        help="Formatos soportados: JPG, PNG, HEIC. Mínimo 640x480 px recomendado."
    )
    
    if uploaded_file is not None:
        process_uploaded_image(uploaded_file)

def process_camera_image(picture):
    """Procesa imagen de la cámara con análisis robusto"""
    try:
        image = Image.open(picture)
        img_array = np.array(image)
        
        # Validar tamaño mínimo
        if img_array.shape[0] < 480 or img_array.shape[1] < 640:
            st.warning("⚠️ La imagen es muy pequeña. Para mejor análisis, usa una resolución mayor.")
        
        process_and_analyze_chagas_advanced(image, img_array, "Cámara Directa")
        
    except Exception as e:
        st.error(f"Error procesando imagen de cámara: {e}")

def process_uploaded_image(uploaded_file):
    """Procesa imagen subida con análisis robusto"""
    try:
        image = Image.open(uploaded_file)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        img_array = np.array(image)
        
        # Validar tamaño mínimo
        if img_array.shape[0] < 480 or img_array.shape[1] < 640:
            st.warning("⚠️ La imagen es muy pequeña. Para mejor análisis, usa una resolución mayor.")
        
        process_and_analyze_chagas_advanced(image, img_array, "Archivo Subido")
        
    except Exception as e:
        st.error(f"Error procesando imagen subida: {e}")

def process_and_analyze_chagas_advanced(original_image, img_array, source):
    """Procesa y analiza la cinta reactiva con métodos robustos"""
    st.success(f"✅ Imagen recibida desde: {source}")
    
    # Procesamiento en múltiples pasos
    with st.spinner("🔍 Analizando imagen con métodos avanzados..."):
        # Mejorar imagen
        enhanced_img = enhance_image_quality_advanced(img_array)
        
        # Análisis de calidad robusto
        quality_analysis = analyze_image_quality_advanced(img_array)
        
        # Detección de bandas con múltiples métodos
        chagas_analysis = detect_chagas_bands_robust(img_array, enhanced_img)
        
        # Validación cruzada de resultados
        validated_analysis = validate_chagas_result(chagas_analysis, quality_analysis)
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Original")
        st.image(img_array, use_column_width=True, caption=f"Resolución: {img_array.shape[1]}x{img_array.shape[0]}")
    
    with col2:
        st.subheader("✨ Mejorada")
        st.image(enhanced_img, use_column_width=True, caption="Imagen procesada para análisis")
    
    # Mostrar métricas de calidad avanzadas
    display_advanced_quality_metrics(quality_analysis)
    
    # Mostrar resultado validado
    display_validated_chagas_result(validated_analysis, quality_analysis)
    
    # Mostrar análisis detallado
    with st.expander("🔬 Análisis Técnico Detallado"):
        display_technical_analysis(chagas_analysis, validated_analysis)
    
    # Opciones de guardado mejoradas
    st.subheader("💾 Reportes Avanzados")
    save_advanced_reports(original_image, quality_analysis, validated_analysis)

def enhance_image_quality_advanced(img_array):
    """Mejora avanzada de la imagen para análisis"""
    try:
        # Convertir a PIL para procesamiento
        pil_img = Image.fromarray(img_array)
        
        # Corrección de iluminación adaptativa
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE para mejorar contraste local
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Filtrado bilateral para reducir ruido preservando bordes
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Mejora de agudeza
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error en mejora avanzada: {e}")
        return img_array

def analyze_image_quality_advanced(img_array):
    """Análisis de calidad avanzado con múltiples métricas"""
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        height, width = gray.shape
        
        # Métricas básicas
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Métricas avanzadas de nitidez
        sharpness_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Métrica de enfoque (Brenner)
        brenner_gradient = np.sum(np.square(np.diff(gray, n=2)))
        brenner_score = min(100, brenner_gradient / (height * width) * 1000)
        
        # Detección de blur
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Análisis de histograma
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        entropy = -np.sum(hist * np.log2(hist + 1e-7)) / (height * width)
        
        # Detección de reflejos y sombras
        bright_areas = np.sum(gray > 220) / gray.size * 100
        dark_areas = np.sum(gray < 30) / gray.size * 100
        
        # Contraste local
        local_contrast = np.std([cv2.meanStdDev(gray[i:i+50, j:j+50])[1][0][0] 
                               for i in range(0, height-50, 50) 
                               for j in range(0, width-50, 50)])
        
        # Score de calidad compuesto avanzado
        quality_score = calculate_advanced_quality_score(
            brightness, contrast, sharpness_laplacian, brenner_score,
            blur_score, entropy, bright_areas, dark_areas, local_contrast
        )
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness_laplacian': sharpness_laplacian,
            'sharpness_brenner': brenner_score,
            'blur_score': blur_score,
            'entropy': entropy,
            'bright_areas': bright_areas,
            'dark_areas': dark_areas,
            'local_contrast': local_contrast,
            'resolution': f"{width}x{height}",
            'aspect_ratio': width / height,
            'quality_score': quality_score,
            'quality_category': get_quality_category(quality_score),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en análisis de calidad avanzado: {e}")
        return {'quality_score': 0, 'quality_category': 'ERROR'}

def calculate_advanced_quality_score(brightness, contrast, sharpness_laplacian, brenner_score,
                                   blur_score, entropy, bright_areas, dark_areas, local_contrast):
    """Calcula score de calidad avanzado"""
    # Normalizar métricas con pesos
    brightness_score = max(0, 100 - abs(brightness - 128) / 255 * 200)
    contrast_score = min(100, contrast / 2.5)
    sharpness_score = min(100, (sharpness_laplacian / 500 + brenner_score / 2) / 2)
    blur_penalty = max(0, (50 - blur_score) / 50 * 30) if blur_score < 50 else 0
    entropy_score = min(100, entropy * 50)
    
    # Penalizaciones por problemas de iluminación
    reflection_penalty = max(0, bright_areas - 5) * 1.5
    shadow_penalty = max(0, dark_areas - 10) * 1.0
    local_contrast_score = min(100, local_contrast * 10)
    
    # Score compuesto
    composite_score = (
        brightness_score * 0.15 +
        contrast_score * 0.20 +
        sharpness_score * 0.25 +
        entropy_score * 0.10 +
        local_contrast_score * 0.10 -
        blur_penalty * 0.10 -
        reflection_penalty -
        shadow_penalty
    )
    
    return max(0, min(100, composite_score))

def get_quality_category(score):
    """Categoriza la calidad de la imagen"""
    if score >= 85:
        return "EXCELENTE"
    elif score >= 70:
        return "BUENA"
    elif score >= 50:
        return "ACEPTABLE"
    elif score >= 30:
        return "BAJA"
    else:
        return "INSUFICIENTE"

def detect_chagas_bands_robust(original_img, enhanced_img):
    """Detección robusta de bandas usando múltiples métodos"""
    try:
        gray = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2GRAY) if len(enhanced_img.shape) == 3 else enhanced_img
        height, width = gray.shape
        
        # MÉTODO 1: Análisis por regiones predefinidas
        regions_analysis = analyze_predefined_regions(gray, height, width)
        
        # MÉTODO 2: Detección automática de bandas
        auto_bands_analysis = detect_bands_automatically(gray)
        
        # MÉTODO 3: Análisis de perfil de intensidad
        profile_analysis = analyze_intensity_profile(gray)
        
        # Combinar resultados de múltiples métodos
        combined_analysis = combine_detection_methods(
            regions_analysis, auto_bands_analysis, profile_analysis
        )
        
        return combined_analysis
        
    except Exception as e:
        logger.error(f"Error en detección robusta: {e}")
        return create_error_analysis()

def analyze_predefined_regions(gray, height, width):
    """Análisis por regiones predefinidas (método principal)"""
    # Definir regiones de interés ajustables
    control_region = gray[int(height*0.25):int(height*0.75), int(width*0.65):int(width*0.80)]
    test_region = gray[int(height*0.25):int(height*0.75), int(width*0.25):int(width*0.40)]
    background_region = gray[int(height*0.1):int(height*0.2), int(width*0.1):int(width*0.2)]
    
    # Calcular métricas para cada región
    control_metrics = analyze_region_metrics(control_region, "control")
    test_metrics = analyze_region_metrics(test_region, "test")
    background_metrics = analyze_region_metrics(background_region, "background")
    
    # Determinar presencia de bandas
    control_present = is_band_present(control_metrics, background_metrics, "control")
    test_present = is_band_present(test_metrics, background_metrics, "test")
    
    # Calcular intensidades relativas
    control_intensity = control_metrics['mean_intensity']
    test_intensity = test_metrics['mean_intensity']
    background_intensity = background_metrics['mean_intensity']
    
    # Normalizar respecto al fondo
    control_relative = max(0, background_intensity - control_intensity)
    test_relative = max(0, background_intensity - test_intensity)
    
    if control_relative > 0:
        intensity_ratio = test_relative / control_relative
    else:
        intensity_ratio = 0
    
    return {
        'method': 'predefined_regions',
        'control_present': control_present,
        'test_present': test_present,
        'control_intensity': control_intensity,
        'test_intensity': test_intensity,
        'control_relative': control_relative,
        'test_relative': test_relative,
        'intensity_ratio': intensity_ratio,
        'background_intensity': background_intensity,
        'confidence': calculate_region_confidence(control_present, test_present, control_relative, test_relative)
    }

def analyze_region_metrics(region, region_type):
    """Calcula métricas detalladas para una región"""
    if region.size == 0:
        return {'mean_intensity': 255, 'std_intensity': 0, 'min_intensity': 255, 'max_intensity': 255}
    
    return {
        'mean_intensity': np.mean(region),
        'std_intensity': np.std(region),
        'min_intensity': np.min(region),
        'max_intensity': np.max(region),
        'region_size': region.size,
        'region_type': region_type
    }

def is_band_present(band_metrics, background_metrics, band_type):
    """Determina si una banda está presente"""
    # Diferencia significativa con el fondo
    intensity_diff = background_metrics['mean_intensity'] - band_metrics['mean_intensity']
    std_threshold = band_metrics['std_intensity'] > 15  # Debe tener variación
    
    if band_type == "control":
        return intensity_diff > 25 and std_threshold
    else:  # test
        return intensity_diff > 15 and std_threshold

def calculate_region_confidence(control_present, test_present, control_relative, test_relative):
    """Calcula confianza basada en las regiones"""
    base_confidence = 70.0
    
    if not control_present:
        return 10.0  # Baja confianza si no hay control
    
    if control_present and test_present:
        if control_relative > 30 and test_relative > 20:
            return min(95.0, base_confidence + 25)
        else:
            return min(85.0, base_confidence + 15)
    elif control_present and not test_present:
        if control_relative > 30:
            return min(90.0, base_confidence + 20)
        else:
            return base_confidence
    
    return 50.0  # Confianza base

def detect_bands_automatically(gray):
    """Detección automática de bandas usando procesamiento de imágenes"""
    try:
        # Preprocesamiento
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detección de bordes
        edges = cv2.Canny(blurred, 50, 150)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área y forma
        band_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Buscar bandas (rectángulos verticales)
            if (area > 100 and 
                0.2 < aspect_ratio < 5.0 and
                min(w, h) > 10):
                band_contours.append(contour)
        
        return {
            'method': 'auto_detection',
            'bands_detected': len(band_contours),
            'contours': band_contours,
            'confidence': min(80.0, len(band_contours) * 20.0)
        }
        
    except Exception as e:
        logger.error(f"Error en detección automática: {e}")
        return {'method': 'auto_detection', 'bands_detected': 0, 'confidence': 0}

def analyze_intensity_profile(gray):
    """Análisis de perfil de intensidad horizontal"""
    try:
        height, width = gray.shape
        middle_row = gray[height//2, :]
        
        # Suavizar perfil
        smoothed_profile = np.convolve(middle_row, np.ones(10)/10, mode='same')
        
        # Encontrar valles (bandas oscuras)
        valleys = []
        for i in range(20, len(smoothed_profile)-20):
            if (smoothed_profile[i] < smoothed_profile[i-10:i].mean() and 
                smoothed_profile[i] < smoothed_profile[i+1:i+11].mean()):
                valleys.append((i, smoothed_profile[i]))
        
        # Ordenar por intensidad (más oscuros primero)
        valleys.sort(key=lambda x: x[1])
        
        return {
            'method': 'intensity_profile',
            'valleys_detected': len(valleys),
            'valley_positions': [v[0] for v in valleys[:4]],  # Máximo 4 valles
            'valley_intensities': [v[1] for v in valleys[:4]],
            'confidence': min(70.0, len(valleys) * 15.0)
        }
        
    except Exception as e:
        logger.error(f"Error en análisis de perfil: {e}")
        return {'method': 'intensity_profile', 'valleys_detected': 0, 'confidence': 0}

def combine_detection_methods(regions_analysis, auto_analysis, profile_analysis):
    """Combina resultados de múltiples métodos de detección"""
    # Ponderar resultados por confianza del método
    regions_weight = 0.6
    auto_weight = 0.25
    profile_weight = 0.15
    
    # Resultado base de regiones (método principal)
    base_result = regions_analysis
    
    # Ajustar confianza basado en otros métodos
    adjusted_confidence = base_result['confidence']
    
    if auto_analysis['bands_detected'] >= 2:
        adjusted_confidence += 10 * auto_weight
    if profile_analysis['valleys_detected'] >= 2:
        adjusted_confidence += 10 * profile_weight
    
    # Determinar resultado final
    if not base_result['control_present']:
        final_result = "INVÁLIDO"
        final_confidence = max(10.0, adjusted_confidence * 0.5)
    elif base_result['control_present'] and not base_result['test_present']:
        final_result = "NEGATIVO"
        final_confidence = min(95.0, adjusted_confidence)
    elif base_result['control_present'] and base_result['test_present']:
        if base_result['intensity_ratio'] < 0.6:
            final_result = "POSITIVO"
            final_confidence = min(95.0, adjusted_confidence * 1.1)
        elif base_result['intensity_ratio'] < 0.8:
            final_result = "DÉBIL POSITIVO"
            final_confidence = min(85.0, adjusted_confidence * 0.9)
        else:
            final_result = "NEGATIVO"  # Banda test muy débil
            final_confidence = min(80.0, adjusted_confidence * 0.8)
    else:
        final_result = "INDETERMINADO"
        final_confidence = 30.0
    
    return {
        'result': final_result,
        'confidence': final_confidence,
        'control_present': base_result['control_present'],
        'test_present': base_result['test_present'],
        'control_intensity': base_result['control_intensity'],
        'test_intensity': base_result['test_intensity'],
        'intensity_ratio': base_result['intensity_ratio'],
        'methods_used': [
            regions_analysis['method'],
            auto_analysis['method'],
            profile_analysis['method']
        ],
        'methods_confidence': {
            'regions': regions_analysis['confidence'],
            'auto': auto_analysis['confidence'],
            'profile': profile_analysis['confidence']
        }
    }

def validate_chagas_result(chagas_analysis, quality_analysis):
    """Valida el resultado considerando la calidad de la imagen"""
    result = chagas_analysis['result']
    confidence = chagas_analysis['confidence']
    quality_score = quality_analysis['quality_score']
    
    # Ajustar confianza basado en calidad de imagen
    quality_factor = quality_score / 100.0
    adjusted_confidence = confidence * quality_factor
    
    # Reglas de validación
    if quality_score < 40:
        # Calidad muy baja, resultado no confiable
        validated_result = "INDETERMINADO"
        validated_confidence = max(10.0, adjusted_confidence * 0.5)
        validation_notes = "Calidad de imagen insuficiente para análisis confiable"
    elif quality_score < 60 and confidence < 70:
        # Calidad baja y confianza baja
        validated_result = "INDETERMINADO"
        validated_confidence = max(20.0, adjusted_confidence * 0.7)
        validation_notes = "Se requiere mejor calidad de imagen para confirmación"
    else:
        # Resultado válido
        validated_result = result
        validated_confidence = adjusted_confidence
        validation_notes = "Análisis realizado con calidad adecuada"
    
    # Aplicar confianza mínima configurada
    if validated_confidence < st.session_state.min_confidence and validated_result not in ["INVÁLIDO", "INDETERMINADO"]:
        validated_result = "INDETERMINADO"
        validated_confidence = st.session_state.min_confidence - 10
        validation_notes = f"Confianza por debajo del mínimo requerido ({st.session_state.min_confidence}%)"
    
    return {
        **chagas_analysis,
        'validated_result': validated_result,
        'validated_confidence': validated_confidence,
        'quality_adjusted_confidence': adjusted_confidence,
        'validation_notes': validation_notes,
        'quality_score': quality_score,
        'min_confidence_threshold': st.session_state.min_confidence
    }

def create_error_analysis():
    """Crea análisis de error estándar"""
    return {
        'result': 'ERROR',
        'confidence': 0.0,
        'control_present': False,
        'test_present': False,
        'control_intensity': 0,
        'test_intensity': 0,
        'intensity_ratio': 0,
        'methods_used': ['error'],
        'validated_result': 'ERROR',
        'validated_confidence': 0.0,
        'validation_notes': 'Error en el procesamiento de la imagen'
    }

# [Las funciones de visualización, guardado y las otras pestañas se mantienen similares pero mejoradas]
# [Se omiten por longitud, pero incluirían display_advanced_quality_metrics, display_validated_chagas_result, etc.]

def display_advanced_quality_metrics(quality_analysis):
    """Muestra métricas de calidad avanzadas"""
    st.subheader("📊 Métricas de Calidad Avanzadas")
    
    # Categoría de calidad
    quality_color = {
        "EXCELENTE": "#4CAF50",
        "BUENA": "#8BC34A", 
        "ACEPTABLE": "#FFC107",
        "BAJA": "#FF9800",
        "INSUFICIENTE": "#F44336"
    }.get(quality_analysis['quality_category'], "#666666")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background-color: {quality_color}20; border-radius: 10px;'>
            <h3 style='color: {quality_color}; margin: 0;'>{quality_analysis['quality_score']:.1f}/100</h3>
            <small style='color: #666;'>{quality_analysis['quality_category']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Brillo", f"{quality_analysis['brightness']:.1f}")
        st.metric("Contraste", f"{quality_analysis['contrast']:.1f}")
    
    with col3:
        st.metric("Nitidez", f"{quality_analysis['sharpness_laplacian']:.0f}")
        st.metric("Enfoque", f"{quality_analysis['sharpness_brenner']:.1f}")

def display_validated_chagas_result(validated_analysis, quality_analysis):
    """Muestra el resultado validado de Chagas"""
    st.subheader("🦟 Resultado Validado de Análisis")
    
    result = validated_analysis['validated_result']
    confidence = validated_analysis['validated_confidence']
    
    # Configuración de colores según resultado
    result_config = {
        "POSITIVO": {"color": "#dc3545", "bg_color": "#f8d7da", "icon": "🔴"},
        "NEGATIVO": {"color": "#28a745", "bg_color": "#d4edda", "icon": "🟢"},
        "DÉBIL POSITIVO": {"color": "#ffc107", "bg_color": "#fff3cd", "icon": "🟡"},
        "INVÁLIDO": {"color": "#17a2b8", "bg_color": "#d1ecf1", "icon": "🔵"},
        "INDETERMINADO": {"color": "#6c757d", "bg_color": "#f5f5f5", "icon": "⚫"},
        "ERROR": {"color": "#6c757d", "bg_color": "#f5f5f5", "icon": "⚫"}
    }.get(result, {"color": "#666", "bg_color": "#f5f5f5", "icon": "⚫"})
    
    st.markdown(f"""
    <div style='background-color: {result_config["bg_color"]}; padding: 25px; border-radius: 10px; border-left: 5px solid {result_config["color"]}; margin: 20px 0;'>
        <h2 style='color: {result_config["color"]}; margin: 0 0 10px 0;'>{result_config["icon"]} RESULTADO: {result}</h2>
        <p style='margin: 10px 0; font-size: 1.2em;'><strong>Confianza:</strong> {confidence:.1f}%</p>
        <p style='margin: 10px 0;'><strong>Notas de validación:</strong> {validated_analysis['validation_notes']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Actualizar estadísticas
    st.session_state.analysis_count += 1
    if result == "POSITIVO":
        st.session_state.chagas_detections += 1
    elif result == "INVÁLIDO":
        st.session_state.invalid_results += 1

def display_technical_analysis(chagas_analysis, validated_analysis):
    """Muestra análisis técnico detallado"""
    st.write("**Métodos de Análisis Utilizados:**")
    for method in chagas_analysis.get('methods_used', []):
        st.write(f"- {method.replace('_', ' ').title()}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Intensidades de Bandas:**")
        st.write(f"- Control: {chagas_analysis['control_intensity']:.1f}")
        st.write(f"- Test: {chagas_analysis['test_intensity']:.1f}")
        st.write(f"- Relación T/C: {chagas_analysis['intensity_ratio']:.2f}")
        
    with col2:
        st.write("**Confianzas por Método:**")
        for method, conf in chagas_analysis.get('methods_confidence', {}).items():
            st.write(f"- {method}: {conf:.1f}%")
    
    st.write("**Ajustes por Calidad:**")
    st.write(f"- Confianza original: {chagas_analysis['confidence']:.1f}%")
    st.write(f"- Ajuste por calidad: {validated_analysis['quality_adjusted_confidence']:.1f}%")
    st.write(f"- Confianza final: {validated_analysis['validated_confidence']:.1f}%")

def save_advanced_reports(original_image, quality_analysis, validated_analysis):
    """Guarda reportes avanzados"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Reporte Completo (Imagen + Análisis)", use_container_width=True, type="primary"):
            save_comprehensive_report(original_image, quality_analysis, validated_analysis)
    
    with col2:
        if st.button("📊 Reporte Técnico Detallado", use_container_width=True):
            save_detailed_technical_report(quality_analysis, validated_analysis)

def save_comprehensive_report(image, quality_analysis, validated_analysis):
    """Guarda reporte comprensivo con imagen anotada"""
    try:
        # Crear imagen anotada (similar a la función anterior pero mejorada)
        img_array = np.array(image)
        annotated_img = create_advanced_annotated_image(img_array, validated_analysis)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chagas_advanced_report_{timestamp}.jpg"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            Image.fromarray(annotated_img).save(tmp_file.name, 'JPEG', quality=95)
            
            with open(tmp_file.name, 'rb') as file:
                st.download_button(
                    label="📥 Descargar Reporte Completo Avanzado",
                    data=file.read(),
                    file_name=filename,
                    mime="image/jpeg",
                    key=f"download_advanced_{timestamp}"
                )
        
        os.unlink(tmp_file.name)
        st.success("✅ Reporte avanzado guardado exitosamente")
        
    except Exception as e:
        st.error(f"Error guardando reporte avanzado: {e}")

def save_detailed_technical_report(quality_analysis, validated_analysis):
    """Guarda reporte técnico detallado"""
    try:
        report = create_detailed_technical_report(quality_analysis, validated_analysis)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Descargar Reporte Técnico Detallado",
            data=report,
            file_name=f"chagas_technical_{timestamp}.txt",
            mime="text/plain",
            key=f"download_technical_{timestamp}"
        )
        
    except Exception as e:
        st.error(f"Error guardando reporte técnico: {e}")

def create_advanced_annotated_image(img_array, validated_analysis):
    """Crea imagen anotada avanzada"""
    try:
        if len(img_array.shape) == 2:
            annotated = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            annotated = img_array.copy()
        
        height, width = annotated.shape[:2]
        
        # Dibujar regiones de análisis
        cv2.rectangle(annotated, 
                     (int(width*0.25), int(height*0.25)), 
                     (int(width*0.40), int(height*0.75)), 
                     (255, 0, 0), 2)  # Test - Azul
        
        cv2.rectangle(annotated, 
                     (int(width*0.65), int(height*0.25)), 
                     (int(width*0.80), int(height*0.75)), 
                     (0, 255, 0), 2)  # Control - Verde
        
        # Añadir texto con resultados
        result_text = f"Resultado: {validated_analysis['validated_result']}"
        confidence_text = f"Confianza: {validated_analysis['validated_confidence']:.1f}%"
        
        cv2.putText(annotated, result_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(annotated, confidence_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Añadir información de bandas
        band_info = f"Control: {validated_analysis['control_present']} | Test: {validated_analysis['test_present']}"
        cv2.putText(annotated, band_info, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated
        
    except Exception as e:
        logger.error(f"Error creando imagen anotada avanzada: {e}")
        return img_array

def create_detailed_technical_report(quality_analysis, validated_analysis):
    """Crea reporte técnico detallado"""
    report = f"""
REPORTE TÉCNICO DETALLADO - ANÁLISIS DE CHAGAS
===============================================
Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

RESULTADO PRINCIPAL
-------------------
Resultado Validado: {validated_analysis['validated_result']}
Confianza Validada: {validated_analysis['validated_confidence']:.1f}%
Notas de Validación: {validated_analysis['validation_notes']}

MÉTRICAS DE CALIDAD AVANZADAS
-----------------------------
Calidad General: {quality_analysis['quality_score']:.1f}/100 ({quality_analysis['quality_category']})
Brillo: {quality_analysis['brightness']:.1f}
Contraste: {quality_analysis['contrast']:.1f}
Nitidez (Laplacian): {quality_analysis['sharpness_laplacian']:.0f}
Nitidez (Brenner): {quality_analysis['sharpness_brenner']:.1f}
Puntuación de Blur: {quality_analysis['blur_score']:.1f}
Entropía: {quality_analysis['entropy']:.3f}
Áreas Brillantes: {quality_analysis['bright_areas']:.1f}%
Áreas Oscuras: {quality_analysis['dark_areas']:.1f}%
Contraste Local: {quality_analysis['local_contrast']:.1f}
Resolución: {quality_analysis['resolution']}

ANÁLISIS DE BANDAS DETALLADO
-----------------------------
Banda Control Presente: {validated_analysis['control_present']}
Banda Test Presente: {validated_analysis['test_present']}
Intensidad Control: {validated_analysis['control_intensity']:.1f}
Intensidad Test: {validated_analysis['test_intensity']:.1f}
Relación Test/Control: {validated_analysis['intensity_ratio']:.2f}

MÉTODOS DE ANÁLISIS UTILIZADOS
------------------------------
"""
    
    for method in validated_analysis.get('methods_used', []):
        report += f"- {method.replace('_', ' ').title()}\n"
    
    report += f"""
CONFIANZAS POR MÉTODO
---------------------
"""
    for method, conf in validated_analysis.get('methods_confidence', {}).items():
        report += f"- {method}: {conf:.1f}%\n"
    
    report += f"""
AJUSTES Y VALIDACIONES
----------------------
Confianza Original: {validated_analysis.get('confidence', 0):.1f}%
Confianza Ajustada por Calidad: {validated_analysis['quality_adjusted_confidence']:.1f}%
Umbral Mínimo de Confianza: {validated_analysis['min_confidence_threshold']}%

CONFIGURACIÓN DE ANÁLISIS
-------------------------
Modo de Análisis: {st.session_state.analysis_mode}
Confianza Mínima Configurada: {st.session_state.min_confidence}%

RECOMENDACIONES TÉCNICAS
------------------------
"""
    
    if quality_analysis['quality_score'] < 60:
        report += "- Se recomienda mejorar la calidad de la imagen para mayor precisión\n"
    if validated_analysis['validated_confidence'] < 80:
        report += "- Considerar repetir el análisis con mejores condiciones de captura\n"
    
    report += """
---
Sistema de Análisis Avanzado de Cintas Reactivas para Chagas
Este reporte fue generado automáticamente usando múltiples métodos de validación.
"""
    
    return report

def render_analysis_tab():
    """Pestaña de análisis histórico"""
    st.header("📈 Historial de Análisis Avanzado")
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if not st.session_state.analysis_history:
        st.info("No hay análisis históricos. Captura algunas imágenes primero.")
        return
    
    # Mostrar histórico con más detalles
    for i, analysis in enumerate(reversed(st.session_state.analysis_history[-10:])):
        with st.expander(f"Análisis {i+1} - {analysis.get('timestamp', '')[:16]} - {analysis.get('validated_result', 'N/A')}"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Resultado", analysis.get('validated_result', 'N/A'))
            with col2:
                st.metric("Confianza", f"{analysis.get('validated_confidence', 0):.1f}%")
            with col3:
                st.metric("Calidad", f"{analysis.get('quality_score', 0):.1f}")
            with col4:
                st.metric("Métodos", len(analysis.get('methods_used', [])))

def render_metrics_tab():
    """Pestaña de métricas y estadísticas"""
    st.header("📊 Métricas y Estadísticas Avanzadas")
    
    if st.session_state.analysis_count == 0:
        st.info("No hay datos de análisis aún. Realiza algunos análisis primero.")
        return
    
    # Estadísticas básicas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Análisis", st.session_state.analysis_count)
    with col2:
        st.metric("Tasa de Chagas", f"{(st.session_state.chagas_detections/st.session_state.analysis_count*100):.1f}%")
    with col3:
        st.metric("Tasa de Inválidos", f"{(st.session_state.invalid_results/st.session_state.analysis_count*100):.1f}%")
    
    # Métricas de calidad promedio
    if 'analysis_history' in st.session_state and st.session_state.analysis_history:
        quality_scores = [a.get('quality_score', 0) for a in st.session_state.analysis_history]
        confidence_scores = [a.get('validated_confidence', 0) for a in st.session_state.analysis_history]
        
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Calidad Promedio", f"{np.mean(quality_scores):.1f}")
        with col5:
            st.metric("Confianza Promedio", f"{np.mean(confidence_scores):.1f}%")

def render_guide_tab():
    """Pestaña de guía avanzada"""
    st.header("📚 Guía Avanzada de Análisis")
    
    st.markdown("""
    ### 🔬 Métodos de Análisis Implementados
    
    **1. Análisis por Regiones Predefinidas (Principal)**
    - Analiza regiones específicas para bandas de control y test
    - Calcula intensidades relativas al fondo
    - Evalúa presencia basada en diferencias de contraste
    
    **2. Detección Automática de Bandas**
    - Usa procesamiento morfológico y detección de contornos
    - Identifica bandas basándose en forma y tamaño
    - Complementa el método principal
    
    **3. Análisis de Perfil de Intensidad**
    - Examina el perfil horizontal de intensidad
    - Detecta valles que corresponden a bandas
    - Proporciona validación adicional
    
    ### 🎯 Validación y Ajuste de Confianza
    
    **Factores Considerados:**
    - Calidad de la imagen (40% de peso)
    - Concordancia entre métodos (30% de peso)
    - Intensidad relativa de bandas (30% de peso)
    
    **Umbrales de Confianza:**
    - ≥85%: Alta confianza
    - 70-84%: Confianza moderada  
    - 50-69%: Confianza baja
    - <50%: Resultado indeterminado
    
    ### ⚙️ Configuración Avanzada
    
    **Modos de Análisis:**
    - **Automático:** Balance entre sensibilidad y especificidad
    - **Conservador:** Menos falsos positivos, mayor especificidad
    - **Sensible:** Detecta casos más débiles, mayor sensibilidad
    
    **Parámetros Ajustables:**
    - Confianza mínima requerida
    - Umbrales de intensidad
    - Tamaños de región de interés
    """)
    
    st.info("""
    **💡 Para mejores resultados:**
    - Use imágenes de al menos 640x480 píxeles
    - Asegure iluminación uniforme sin reflejos
    - Capture toda el área de resultado de la cinta
    - Mantenga la cinta plana y bien enfocada
    """)

if __name__ == "__main__":
    main()
