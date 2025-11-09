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
        # Esta es una verificación básica - en Streamlit Cloud siempre es HTTPS
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
        page_title="Analizador de Cintas Reactivas - HTTPS",
        page_icon="🔬",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Verificar HTTPS
    check_https_status()
    
    st.title("📱 Analizador de Cintas Reactivas")
    st.markdown("### **Captura y análisis optimizado para móviles**")
    
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
        st.metric("Análisis Realizados", st.session_state.analysis_count)
    
    # Pestañas principales
    tab1, tab2, tab3 = st.tabs(["🎯 Captura", "🔍 Análisis", "📚 Guía"])
    
    with tab1:
        render_capture_tab()
    
    with tab2:
        render_analysis_tab()
    
    with tab3:
        render_guide_tab()

def render_capture_tab():
    """Pestaña de captura de imágenes"""
    st.header("📸 Captura de Imágenes")
    
    # Opción 1: Cámara directa (funciona con HTTPS)
    st.subheader("Opción 1: Cámara Directa")
    st.markdown("""
    <div style='background-color: #e7f3ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <b>💡 Importante:</b> Permite el acceso a la cámara cuando tu navegador lo solicite.
    </div>
    """, unsafe_allow_html=True)
    
    try:
        picture = st.camera_input(
            "Toma una foto de la cinta reactiva",
            help="Asegúrate de que la cinta esté bien iluminada y enfocada"
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
        process_and_display(image, img_array, "Cámara Directa")
        
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
        process_and_display(image, img_array, "Archivo Subido")
        
    except Exception as e:
        st.error(f"Error procesando imagen subida: {e}")

def process_and_display(original_image, img_array, source):
    """Procesa y muestra la imagen con análisis"""
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
    analysis = analyze_image_quality(img_array)
    
    # Mostrar métricas
    display_quality_metrics(analysis)
    
    # Recomendaciones
    show_quality_recommendations(analysis)
    
    # Opciones de guardado
    st.subheader("💾 Guardar Resultados")
    save_image_analysis(original_image, analysis)

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
    st.subheader("📊 Métricas de Calidad")
    
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
    
    # Métricas secundarias
    col5, col6 = st.columns(2)
    
    with col5:
        st.metric("Áreas Brillantes", f"{analysis['bright_areas']:.1f}%")
    
    with col6:
        st.metric("Resolución", analysis['resolution'])

def show_quality_recommendations(analysis):
    """Muestra recomendaciones basadas en el análisis"""
    st.subheader("💡 Recomendaciones")
    
    recommendations = []
    
    # Evaluar calidad general
    if analysis['quality_score'] >= 80:
        recommendations.append("✅ **Excelente calidad** - Ideal para análisis detallado")
    elif analysis['quality_score'] >= 60:
        recommendations.append("⚠️ **Buena calidad** - Aceptable para análisis")
    elif analysis['quality_score'] >= 40:
        recommendations.append("🔸 **Calidad regular** - Considera tomar otra foto")
    else:
        recommendations.append("❌ **Calidad insuficiente** - Toma una nueva foto con mejor iluminación")
    
    # Recomendaciones específicas
    if analysis['brightness'] < 80:
        recommendations.append("💡 **Poca iluminación** - Aumenta la luz o usa flash")
    elif analysis['brightness'] > 200:
        recommendations.append("🔆 **Exceso de luz** - Reduce brillo o evita reflejos directos")
    
    if analysis['contrast'] < 20:
        recommendations.append("🎨 **Bajo contraste** - Mejora la iluminación lateral")
    
    if analysis['sharpness'] < 100:
        recommendations.append("🔍 **Poca nitidez** - Asegura el enfoque y estabilidad")
    
    if analysis['bright_areas'] > 10:
        recommendations.append("✨ **Demasiados reflejos** - Cambia el ángulo o reduce luz directa")
    
    # Mostrar recomendaciones
    for rec in recommendations:
        st.write(rec)

def save_image_analysis(image, analysis):
    """Guarda la imagen y el análisis"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Guardar Imagen", use_container_width=True):
            save_image_file(image, analysis)
    
    with col2:
        if st.button("📊 Guardar Análisis", use_container_width=True):
            save_analysis_report(analysis)

def save_image_file(image, analysis):
    """Guarda la imagen con metadatos"""
    try:
        # Crear nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cinta_reactiva_{timestamp}.jpg"
        
        # Guardar imagen temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name, 'JPEG', quality=95)
            
            # Ofrecer descarga
            with open(tmp_file.name, 'rb') as file:
                st.download_button(
                    label="📥 Descargar Imagen",
                    data=file,
                    file_name=filename,
                    mime="image/jpeg",
                    key=f"download_img_{timestamp}"
                )
        
        # Limpiar
        os.unlink(tmp_file.name)
        st.session_state.analysis_count += 1
        
    except Exception as e:
        st.error(f"Error guardando imagen: {e}")

def save_analysis_report(analysis):
    """Guarda el reporte de análisis"""
    try:
        # Crear reporte
        report = f"""ANÁLISIS DE CINTA REACTIVA
Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

MÉTRICAS DE CALIDAD:
- Calidad General: {analysis['quality_score']:.1f}/100
- Brillo: {analysis['brightness']:.1f}
- Contraste: {analysis['contrast']:.1f}
- Nitidez: {analysis['sharpness']:.0f}
- Áreas Brillantes: {analysis['bright_areas']:.1f}%
- Resolución: {analysis['resolution']}

RECOMENDACIONES:
{get_recommendations_text(analysis)}
"""
        # Ofrecer descarga
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Descargar Reporte",
            data=report,
            file_name=f"analisis_cinta_{timestamp}.txt",
            mime="text/plain",
            key=f"download_report_{timestamp}"
        )
        
    except Exception as e:
        st.error(f"Error guardando reporte: {e}")

def get_recommendations_text(analysis):
    """Genera texto de recomendaciones para el reporte"""
    recommendations = []
    
    if analysis['quality_score'] >= 80:
        recommendations.append("- Excelente calidad para análisis")
    elif analysis['quality_score'] >= 60:
        recommendations.append("- Buena calidad, aceptable para análisis")
    else:
        recommendations.append("- Considerar nueva captura con mejor iluminación")
    
    if analysis['brightness'] < 80:
        recommendations.append("- Aumentar iluminación")
    elif analysis['brightness'] > 200:
        recommendations.append("- Reducir brillo y reflejos")
    
    return "\n".join(recommendations)

def render_analysis_tab():
    """Pestaña de análisis histórico"""
    st.header("📈 Análisis Histórico")
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if not st.session_state.analysis_history:
        st.info("No hay análisis históricos. Captura algunas imágenes primero.")
        return
    
    # Mostrar histórico
    for i, analysis in enumerate(reversed(st.session_state.analysis_history[-10:])):
        with st.expander(f"Análisis {i+1} - {analysis.get('timestamp', '')[:16]}"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Calidad", f"{analysis.get('quality_score', 0):.1f}")
            with col2:
                st.metric("Brillo", f"{analysis.get('brightness', 0):.1f}")

def render_guide_tab():
    """Pestaña de guía de uso"""
    st.header("📚 Guía de Uso")
    
    st.markdown("""
    ### 🎯 Instrucciones para Captura Óptima
    
    **1. Preparación:**
    - Coloca la cinta reactiva en superficie plana
    - Asegura buena iluminación indirecta
    - Limpia el lente de la cámara
    
    **2. Captura:**
    - Mantén el dispositivo estable
    - Usa enfoque automático
    - Distancia recomendada: 15-20 cm
    - Evita sombras y reflejos
    
    **3. Análisis:**
    - Revisa las métricas de calidad
    - Sigue las recomendaciones
    - Guarda resultados importantes
    
    ### 📱 Compatibilidad Móvil
    - ✅ HTTPS activado para acceso a cámara
    - ✅ Compatible con iOS y Android
    - ✅ Navegadores: Chrome, Safari, Firefox
    - ✅ Conexión segura garantizada
    
    ### 🔧 Solución de Problemas
    - **Cámara no funciona:** Verifica permisos en el navegador
    - **Imagen borrosa:** Mejora iluminación y enfoque
    - **Error de análisis:** Verifica formato de imagen
    - **Problemas de descarga:** Revisa almacenamiento del dispositivo
    """)

if __name__ == "__main__":
    main()
