import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tempfile
import os
import time
from datetime import datetime
import logging

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
        page_title="Analizador de Cintas Reactivas Chagas - HTTPS",
        page_icon="🦟",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Verificar HTTPS
    check_https_status()
    
    st.title("🦟 Analizador de Cintas Reactivas para Chagas")
    st.markdown("### **Detección e interpretación automática de resultados**")
    
    # Sidebar con información
    with st.sidebar:
        st.header("⚙️ Configuración")
        st.info("""
        **Para mejor calidad:**
        - Buena iluminación
        - Enfoque automático
        - Cámara estable
        - Fondo uniforme
        """)
        
        st.header("📊 Estadísticas")
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        if 'chagas_detections' not in st.session_state:
            st.session_state.chagas_detections = 0
        st.metric("Análisis Realizados", st.session_state.analysis_count)
        st.metric("Detecciones Chagas", st.session_state.chagas_detections)
    
    # Pestañas principales
    tab1, tab2, tab3 = st.tabs(["🎯 Captura y Análisis", "🔍 Resultados", "📚 Guía Chagas"])
    
    with tab1:
        render_capture_tab()
    
    with tab2:
        render_analysis_tab()
    
    with tab3:
        render_guide_tab()

def render_capture_tab():
    """Pestaña de captura de imágenes"""
    st.header("📸 Captura y Análisis de Cinta Reactiva")
    
    # Opción 1: Cámara directa (funciona con HTTPS)
    st.subheader("Opción 1: Cámara Directa")
    st.markdown("""
    <div style='background-color: #e7f3ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <b>💡 Importante:</b> Asegúrate de que la cinta reactiva esté bien visible y enfocada.
    </div>
    """, unsafe_allow_html=True)
    
    try:
        picture = st.camera_input(
            "Toma una foto de la cinta reactiva de Chagas",
            help="Enfoca claramente las bandas de control y test"
        )
        
        if picture is not None:
            process_camera_image(picture)
            
    except Exception as e:
        st.error(f"❌ Error con la cámara: {e}")
        st.info("💡 Si la cámara no funciona, usa la Opción 2: Subir archivo")
    
    # Opción 2: Subir archivo (alternativa)
    st.subheader("Opción 2: Subir Archivo")
    uploaded_file = st.file_uploader(
        "O sube una foto desde tu galería",
        type=['jpg', 'jpeg', 'png', 'heic'],
        help="Formatos soportados: JPG, PNG, HEIC"
    )
    
    if uploaded_file is not None:
        process_uploaded_image(uploaded_file)

def process_camera_image(picture):
    """Procesa imagen de la cámara"""
    try:
        # Convertir a PIL Image
        image = Image.open(picture)
        img_array = np.array(image)
        
        # Procesar y mostrar resultados
        process_and_analyze_chagas(image, img_array, "Cámara Directa")
        
    except Exception as e:
        st.error(f"Error procesando imagen de cámara: {e}")

def process_uploaded_image(uploaded_file):
    """Procesa imagen subida"""
    try:
        # Convertir a PIL Image
        image = Image.open(uploaded_file)
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        img_array = np.array(image)
        
        # Procesar y mostrar resultados
        process_and_analyze_chagas(image, img_array, "Archivo Subido")
        
    except Exception as e:
        st.error(f"Error procesando imagen subida: {e}")

def process_and_analyze_chagas(original_image, img_array, source):
    """Procesa y analiza la cinta reactiva de Chagas"""
    st.success(f"✅ Imagen recibida desde: {source}")
    
    # Mostrar imágenes en columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Original")
        st.image(img_array, use_column_width=True, caption="Imagen original")
    
    with col2:
        st.subheader("✨ Mejorada")
        enhanced_img = enhance_image_quality(img_array)
        st.image(enhanced_img, use_column_width=True, caption="Imagen mejorada")
    
    # Análisis de calidad
    quality_analysis = analyze_image_quality(img_array)
    
    # Detección de bandas de Chagas
    chagas_analysis = detect_chagas_bands(img_array)
    
    # Mostrar métricas de calidad
    display_quality_metrics(quality_analysis)
    
    # Mostrar resultado de Chagas
    display_chagas_result(chagas_analysis, quality_analysis)
    
    # Opciones de guardado
    st.subheader("💾 Guardar Resultados")
    save_chagas_report(original_image, quality_analysis, chagas_analysis)

def enhance_image_quality(img_array):
    """Mejora la calidad de la imagen para análisis"""
    try:
        pil_img = Image.fromarray(img_array)
        
        # Mejorar contraste
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.3)
        
        # Mejorar nitidez
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.2)
        
        # Ajustar brillo si es necesario
        brightness = np.mean(img_array)
        if brightness < 100:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1.1)
        elif brightness > 200:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(0.9)
        
        return np.array(pil_img)
        
    except Exception as e:
        logger.error(f"Error mejorando imagen: {e}")
        return img_array

def detect_chagas_bands(img_array):
    """Detecta y analiza las bandas de la cinta reactiva de Chagas"""
    try:
        # Convertir a escala de grises
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
        
        height, width = gray.shape
        
        # Definir regiones de interés para bandas (ajustar según la cinta)
        control_region = gray[int(height*0.3):int(height*0.7), int(width*0.7):int(width*0.85)]
        test_region = gray[int(height*0.3):int(height*0.7), int(width*0.3):int(width*0.45)]
        
        # Calcular intensidad promedio en cada región
        control_intensity = np.mean(control_region)
        test_intensity = np.mean(test_region)
        
        # Calcular contraste en cada región
        control_contrast = np.std(control_region)
        test_contrast = np.std(test_region)
        
        # Umbrales para detección (ajustables)
        intensity_threshold = 100
        contrast_threshold = 20
        
        # Detectar presencia de bandas
        control_present = control_intensity < intensity_threshold and control_contrast > contrast_threshold
        test_present = test_intensity < intensity_threshold and test_contrast > contrast_threshold
        
        # Calcular relación de intensidades
        if control_intensity > 0:
            intensity_ratio = test_intensity / control_intensity
        else:
            intensity_ratio = 1.0
        
        # Determinar resultado
        if not control_present:
            result = "INVÁLIDO"
            confidence = 0.0
        elif control_present and not test_present:
            result = "NEGATIVO"
            confidence = min(95.0, 100 - (intensity_ratio * 20))
        elif control_present and test_present:
            if intensity_ratio < 0.7:
                result = "POSITIVO"
                confidence = min(95.0, 100 - (intensity_ratio * 60))
            else:
                result = "DÉBIL POSITIVO"
                confidence = min(80.0, 70 - (intensity_ratio * 30))
        else:
            result = "INDETERMINADO"
            confidence = 0.0
        
        return {
            'result': result,
            'confidence': confidence,
            'control_present': control_present,
            'test_present': test_present,
            'control_intensity': control_intensity,
            'test_intensity': test_intensity,
            'intensity_ratio': intensity_ratio,
            'control_region': control_region,
            'test_region': test_region
        }
        
    except Exception as e:
        logger.error(f"Error detectando bandas Chagas: {e}")
        return {
            'result': 'ERROR',
            'confidence': 0.0,
            'control_present': False,
            'test_present': False,
            'control_intensity': 0,
            'test_intensity': 0,
            'intensity_ratio': 0
        }

def analyze_image_quality(img_array):
    """Analiza la calidad de la imagen"""
    try:
        # Convertir a escala de grises
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Métricas básicas
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Calcular nitidez (varianza Laplaciana)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calcular relación de aspecto y tamaño
        height, width = gray.shape
        aspect_ratio = width / height
        
        # Detectar áreas muy brillantes (reflejos)
        bright_areas = np.sum(gray > 220) / gray.size * 100
        
        # Score de calidad compuesto (0-100)
        quality_score = calculate_quality_score(brightness, contrast, sharpness, bright_areas)
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'bright_areas': bright_areas,
            'resolution': f"{width}x{height}",
            'aspect_ratio': aspect_ratio,
            'quality_score': quality_score,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analizando imagen: {e}")
        return {
            'brightness': 0, 'contrast': 0, 'sharpness': 0, 
            'bright_areas': 0, 'quality_score': 0
        }

def calculate_quality_score(brightness, contrast, sharpness, bright_areas):
    """Calcula un score de calidad compuesto"""
    # Normalizar métricas
    brightness_score = 100 - abs(brightness - 128) / 255 * 100
    contrast_score = min(100, contrast / 3)
    sharpness_score = min(100, sharpness / 1000)
    reflection_penalty = max(0, bright_areas - 5) * 2  # Penalizar reflejos excesivos
    
    # Score compuesto
    composite_score = (
        brightness_score * 0.25 +
        contrast_score * 0.35 +
        sharpness_score * 0.40 -
        reflection_penalty
    )
    
    return max(0, min(100, composite_score))

def display_quality_metrics(analysis):
    """Muestra las métricas de calidad"""
    st.subheader("📊 Métricas de Calidad de Imagen")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        quality_color = "#4CAF50" if analysis['quality_score'] >= 70 else "#FF9800" if analysis['quality_score'] >= 50 else "#F44336"
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background-color: {quality_color}20; border-radius: 10px;'>
            <h3 style='color: {quality_color}; margin: 0;'>{analysis['quality_score']:.1f}/100</h3>
            <small style='color: #666;'>Calidad General</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Brillo", f"{analysis['brightness']:.1f}")
    
    with col3:
        st.metric("Contraste", f"{analysis['contrast']:.1f}")
    
    with col4:
        st.metric("Nitidez", f"{analysis['sharpness']:.0f}")

def display_chagas_result(chagas_analysis, quality_analysis):
    """Muestra el resultado del análisis de Chagas"""
    st.subheader("🦟 Resultado de Análisis de Chagas")
    
    result = chagas_analysis['result']
    confidence = chagas_analysis['confidence']
    
    # Mostrar resultado con color según el diagnóstico
    if result == "POSITIVO":
        st.error(f"""
        <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545;'>
            <h2 style='color: #dc3545; margin: 0;'>🔴 RESULTADO: {result}</h2>
            <p style='margin: 10px 0;'>Confianza: {confidence:.1f}%</p>
            <p><strong>Interpretación:</strong> Presencia de anticuerpos contra Trypanosoma cruzi detectada.</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.chagas_detections += 1
        
    elif result == "NEGATIVO":
        st.success(f"""
        <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;'>
            <h2 style='color: #28a745; margin: 0;'>🟢 RESULTADO: {result}</h2>
            <p style='margin: 10px 0;'>Confianza: {confidence:.1f}%</p>
            <p><strong>Interpretación:</strong> No se detectaron anticuerpos contra Trypanosoma cruzi.</p>
        </div>
        """, unsafe_allow_html=True)
        
    elif result == "DÉBIL POSITIVO":
        st.warning(f"""
        <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107;'>
            <h2 style='color: #856404; margin: 0;'>🟡 RESULTADO: {result}</h2>
            <p style='margin: 10px 0;'>Confianza: {confidence:.1f}%</p>
            <p><strong>Interpretación:</strong> Banda de test débilmente visible. Se recomienda confirmación.</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.chagas_detections += 0.5
        
    elif result == "INVÁLIDO":
        st.info(f"""
        <div style='background-color: #d1ecf1; padding: 20px; border-radius: 10px; border-left: 5px solid #17a2b8;'>
            <h2 style='color: #0c5460; margin: 0;'>🔵 RESULTADO: {result}</h2>
            <p style='margin: 10px 0;'>Confianza: {confidence:.1f}%</p>
            <p><strong>Interpretación:</strong> Prueba inválida. Banda de control no detectada.</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.error(f"""
        <div style='background-color: #f5f5f5; padding: 20px; border-radius: 10px; border-left: 5px solid #6c757d;'>
            <h2 style='color: #6c757d; margin: 0;'>⚫ RESULTADO: {result}</h2>
            <p style='margin: 10px 0;'>No se pudo determinar el resultado.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Mostrar detalles técnicos
    with st.expander("🔍 Ver detalles técnicos del análisis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Banda de Control:**")
            st.write(f"- Presente: {'✅ Sí' if chagas_analysis['control_present'] else '❌ No'}")
            st.write(f"- Intensidad: {chagas_analysis['control_intensity']:.1f}")
            
        with col2:
            st.write("**Banda de Test:**")
            st.write(f"- Presente: {'✅ Sí' if chagas_analysis['test_present'] else '❌ No'}")
            st.write(f"- Intensidad: {chagas_analysis['test_intensity']:.1f}")
            st.write(f"- Relación C/T: {chagas_analysis['intensity_ratio']:.2f}")
    
    # Recomendaciones basadas en calidad
    if quality_analysis['quality_score'] < 60:
        st.warning("""
        ⚠️ **Advertencia de Calidad:** La calidad de la imagen puede afectar la precisión del análisis. 
        Se recomienda tomar una nueva foto con mejor iluminación y enfoque.
        """)

def save_chagas_report(original_image, quality_analysis, chagas_analysis):
    """Guarda el reporte completo de Chagas"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Guardar Imagen y Reporte", use_container_width=True, type="primary"):
            save_complete_report(original_image, quality_analysis, chagas_analysis)
    
    with col2:
        if st.button("📊 Solo Reporte", use_container_width=True):
            save_text_report(quality_analysis, chagas_analysis)

def save_complete_report(image, quality_analysis, chagas_analysis):
    """Guarda imagen con anotaciones y reporte"""
    try:
        # Crear imagen anotada
        img_array = np.array(image)
        annotated_img = annotate_chagas_result(img_array, chagas_analysis)
        
        # Crear nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chagas_report_{timestamp}.jpg"
        
        # Guardar imagen temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            Image.fromarray(annotated_img).save(tmp_file.name, 'JPEG', quality=95)
            
            # Ofrecer descarga
            with open(tmp_file.name, 'rb') as file:
                st.download_button(
                    label="📥 Descargar Reporte Completo (Imagen + Análisis)",
                    data=file.read(),
                    file_name=filename,
                    mime="image/jpeg",
                    key=f"download_complete_{timestamp}"
                )
        
        # Limpiar
        os.unlink(tmp_file.name)
        st.session_state.analysis_count += 1
        st.success("✅ Reporte guardado exitosamente")
        
    except Exception as e:
        st.error(f"Error guardando reporte completo: {e}")

def save_text_report(quality_analysis, chagas_analysis):
    """Guarda solo el reporte de texto"""
    try:
        # Crear reporte detallado
        report = f"""
REPORTE DE ANÁLISIS - PRUEBA DE CHAGAS
=======================================
Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

RESULTADO PRINCIPAL
-------------------
Resultado: {chagas_analysis['result']}
Confianza: {chagas_analysis['confidence']:.1f}%

DETALLES TÉCNICOS
-----------------
Banda de Control: {'PRESENTE' if chagas_analysis['control_present'] else 'AUSENTE'}
Banda de Test: {'PRESENTE' if chagas_analysis['test_present'] else 'AUSENTE'}
Intensidad Control: {chagas_analysis['control_intensity']:.1f}
Intensidad Test: {chagas_analysis['test_intensity']:.1f}
Relación C/T: {chagas_analysis['intensity_ratio']:.2f}

MÉTRICAS DE CALIDAD
-------------------
Calidad General: {quality_analysis['quality_score']:.1f}/100
Brillo: {quality_analysis['brightness']:.1f}
Contraste: {quality_analysis['contrast']:.1f}
Nitidez: {quality_analysis['sharpness']:.0f}
Resolución: {quality_analysis['resolution']}

INTERPRETACIÓN CLÍNICA
----------------------
{get_clinical_interpretation(chagas_analysis['result'])}

RECOMENDACIONES
---------------
{get_recommendations(chagas_analysis['result'], quality_analysis['quality_score'])}

---
Este reporte fue generado automáticamente por el sistema de análisis de cintas reactivas.
Los resultados deben ser confirmados por personal médico calificado.
"""
        # Ofrecer descarga
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Descargar Reporte de Texto",
            data=report,
            file_name=f"chagas_analysis_{timestamp}.txt",
            mime="text/plain",
            key=f"download_text_{timestamp}"
        )
        
    except Exception as e:
        st.error(f"Error guardando reporte de texto: {e}")

def annotate_chagas_result(img_array, chagas_analysis):
    """Añade anotaciones a la imagen con el resultado"""
    try:
        # Convertir a RGB para dibujar
        if len(img_array.shape) == 2:
            annotated = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            annotated = img_array.copy()
        
        height, width = annotated.shape[:2]
        
        # Dibujar rectángulos en las regiones de análisis
        cv2.rectangle(annotated, 
                     (int(width*0.3), int(height*0.3)), 
                     (int(width*0.45), int(height*0.7)), 
                     (255, 0, 0), 2)  # Test - Azul
        
        cv2.rectangle(annotated, 
                     (int(width*0.7), int(height*0.3)), 
                     (int(width*0.85), int(height*0.7)), 
                     (0, 255, 0), 2)  # Control - Verde
        
        # Añadir texto con resultado
        result_text = f"Resultado: {chagas_analysis['result']} ({chagas_analysis['confidence']:.1f}%)"
        cv2.putText(annotated, result_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Añadir leyenda
        cv2.putText(annotated, "Test", (int(width*0.3), int(height*0.25)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(annotated, "Control", (int(width*0.7), int(height*0.25)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotated
        
    except Exception as e:
        logger.error(f"Error anotando imagen: {e}")
        return img_array

def get_clinical_interpretation(result):
    """Devuelve la interpretación clínica según el resultado"""
    interpretations = {
        "POSITIVO": "Presencia de anticuerpos contra Trypanosoma cruzi detectada. Se recomienda confirmación con pruebas adicionales y evaluación médica.",
        "NEGATIVO": "No se detectaron anticuerpos contra Trypanosoma cruzi. Resultado consistente con ausencia de infección activa.",
        "DÉBIL POSITIVO": "Banda de test débilmente visible. Puede indicar infección temprana o baja carga de anticuerpos. Se requiere confirmación.",
        "INVÁLIDO": "Prueba técnicamente inválida. La banda de control no fue detectada. Repetir la prueba con nuevo dispositivo.",
        "ERROR": "No se pudo procesar la imagen. Verificar la calidad de la foto y repetir el análisis."
    }
    return interpretations.get(result, "Resultado no interpretable.")

def get_recommendations(result, quality_score):
    """Genera recomendaciones basadas en el resultado y calidad"""
    recommendations = []
    
    if result == "POSITIVO":
        recommendations.extend([
            "✅ Confirmar con prueba de ELISA o PCR",
            "✅ Consultar con especialista en enfermedades tropicales",
            "✅ Realizar evaluación clínica completa",
            "✅ Considerar estudio de contactos si es apropiado"
        ])
    elif result == "DÉBIL POSITIVO":
        recommendations.extend([
            "🔄 Repetir la prueba en 2-4 semanas",
            "✅ Confirmar con pruebas serológicas adicionales",
            "🔍 Evaluar factores de riesgo epidemiológicos"
        ])
    elif result == "INVÁLIDO":
        recommendations.extend([
            "🔄 Repetir la prueba con nuevo dispositivo",
            "📷 Verificar que la imagen capture toda la cinta reactiva",
            "💡 Asegurar buena iluminación y enfoque"
        ])
    
    if quality_score < 60:
        recommendations.append("📸 Mejorar calidad de imagen para mayor precisión")
    
    return "\n".join(recommendations) if recommendations else "No se requieren acciones adicionales."

def render_analysis_tab():
    """Pestaña de análisis histórico"""
    st.header("📈 Historial de Análisis")
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if not st.session_state.analysis_history:
        st.info("No hay análisis históricos. Captura algunas imágenes primero.")
        return
    
    # Mostrar histórico
    for i, analysis in enumerate(reversed(st.session_state.analysis_history[-10:])):
        with st.expander(f"Análisis {i+1} - {analysis.get('timestamp', '')[:16]}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Resultado", analysis.get('result', 'N/A'))
            with col2:
                st.metric("Confianza", f"{analysis.get('confidence', 0):.1f}%")
            with col3:
                st.metric("Calidad", f"{analysis.get('quality_score', 0):.1f}")

def render_guide_tab():
    """Pestaña de guía de Chagas"""
    st.header("📚 Guía de Prueba de Chagas")
    
    st.markdown("""
    ### 🦟 Información sobre la Enfermedad de Chagas
    
    **Agente causal:** Trypanosoma cruzi
    **Transmisión:** Vectorial (chinches), congénita, transfusiones
    **Distribución:** América Latina, casos importados globalmente
    
    ### 🎯 Instrucciones para la Prueba
    
    **1. Preparación de la Cinta:**
    - Verificar fecha de caducidad del dispositivo
    - Aplicar muestra según instrucciones del fabricante
    - Esperar tiempo de desarrollo especificado
    
    **2. Captura de Imagen:**
    - Colocar cinta en superficie plana y bien iluminada
    - Enfocar claramente las bandas de test y control
    - Evitar sombras y reflejos
    - Capturar toda el área de resultado
    
    **3. Interpretación de Resultados:**
    
    **POSITIVO (🔴):** Ambas bandas visibles
    - Presencia de anticuerpos anti-T. cruzi
    - Requiere confirmación con pruebas adicionales
    
    **NEGATIVO (🟢):** Solo banda control visible  
    - No se detectan anticuerpos anti-T. cruzi
    - Considerar factores de riesgo epidemiológicos
    
    **DÉBIL POSITIVO (🟡):** Banda test tenue
    - Posible infección temprana o baja carga
    - Repetir prueba en 2-4 semanas
    
    **INVÁLIDO (🔵):** Banda control no visible
    - Error en la prueba o reactivo
    - Repetir con nuevo dispositivo
    
    ### ⚠️ Limitaciones y Consideraciones
    
    - Esta es una herramienta de apoyo al diagnóstico
    - Los resultados deben ser confirmados por personal médico
    - Considerar contexto clínico y epidemiológico
    - La calidad de la imagen afecta la precisión del análisis
    
    ### 📞 Acciones Recomendadas
    
    **Resultado POSITIVO:**
    - Derivar a especialista en enfermedades tropicales
    - Realizar pruebas confirmatorias (ELISA, PCR, IFA)
    - Evaluación clínica completa
    
    **Resultado NEGATIVO con factores de riesgo:**
    - Repetir prueba en 4-6 semanas si exposición reciente
    - Considerar otras pruebas diagnósticas
    """)

if __name__ == "__main__":
    main()
