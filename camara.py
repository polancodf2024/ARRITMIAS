import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tempfile
import os
import time
from datetime import datetime

def main():
    st.set_page_config(
        page_title="Analizador de Cintas Reactivas",
        page_icon="🔬",
        layout="centered"
    )
    
    st.title("📱 Analizador Móvil de Cintas Reactivas")
    
    # Mensaje informativo sobre permisos
    st.markdown("""
    <div style='background-color: #e7f3ff; padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3;'>
    <h4>🛠️ Configuración Requerida</h4>
    <p>Para usar la cámara en tu móvil:</p>
    <ol>
    <li><b>Abre esta página en Chrome o Safari</b></li>
    <li><b>Permite el acceso a la cámara</b> cuando el navegador lo solicite</li>
    <li><b>Si no funciona</b>, usa la opción de subir archivo</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Opción 1: Cámara directa (puede no funcionar en algunos navegadores)
    st.subheader("📸 Opción 1: Tomar Foto Directamente")
    
    try:
        picture = st.camera_input("Toma una foto de la cinta reactiva")
        
        if picture is not None:
            process_image(picture, "Cámara Directa")
            
    except Exception as e:
        st.error(f"❌ La cámara no está disponible: {e}")
        st.info("💡 Usa la opción de subir archivo below")
    
    # Opción 2: Subir archivo (siempre funciona)
    st.subheader("📁 Opción 2: Subir Foto Existente")
    
    uploaded_file = st.file_uploader(
        "Selecciona una foto de tu galería",
        type=['jpg', 'jpeg', 'png'],
        help="Toma la foto con tu app de cámara y luego súbela aquí"
    )
    
    if uploaded_file is not None:
        process_image(uploaded_file, "Archivo Subido")

def process_image(file, source_type):
    """Procesa la imagen sin importar la fuente"""
    
    # Mostrar información de la fuente
    st.success(f"✅ Imagen recibida desde: {source_type}")
    
    # Convertir a imagen PIL
    image = Image.open(file)
    img_array = np.array(image)
    
    # Mostrar en columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Original")
        st.image(img_array, use_column_width=True)
    
    with col2:
        st.subheader("✨ Mejorada")
        enhanced_img = enhance_image(img_array)
        st.image(enhanced_img, use_column_width=True)
    
    # Análisis
    analysis = analyze_image(img_array)
    
    # Mostrar métricas
    st.subheader("📊 Métricas de Calidad")
    
    cols = st.columns(4)
    metrics = [
        ("Calidad", f"{analysis['quality_score']:.1f}/100", "#4CAF50"),
        ("Brillo", f"{analysis['brightness']:.1f}", "#FF9800"),
        ("Contraste", f"{analysis['contrast']:.1f}", "#2196F3"),
        ("Nitidez", f"{analysis['sharpness']:.3f}", "#9C27B0")
    ]
    
    for col, (label, value, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {color}20; border-radius: 10px;'>
                <h4 style='color: {color}; margin: 0;'>{value}</h4>
                <small style='color: #666;'>{label}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Recomendaciones basadas en análisis
    show_recommendations(analysis)
    
    # Opción para guardar
    if st.button("💾 Guardar Análisis y Foto", type="primary"):
        save_results(image, analysis)

def enhance_image(img_array):
    """Mejora la calidad de la imagen"""
    try:
        pil_img = Image.fromarray(img_array)
        
        # Mejorar contraste
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.3)
        
        # Mejorar nitidez
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.2)
        
        # Mejorar brillo si es necesario
        brightness = np.mean(img_array)
        if brightness < 100:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1.2)
        
        return np.array(pil_img)
    
    except Exception as e:
        st.error(f"Error mejorando imagen: {e}")
        return img_array

def analyze_image(img_array):
    """Analiza la imagen y devuelve métricas"""
    try:
        # Convertir a escala de grises si es color
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Métricas básicas
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Calcular nitidez
        gy, gx = np.gradient(gray.astype(float))
        sharpness = np.sqrt(gx**2 + gy**2).mean()
        
        # Score de calidad (0-100)
        quality_score = min(100, max(0, 
            contrast * 0.4 + 
            sharpness * 2.5 + 
            (255 - abs(brightness - 128)) * 0.2
        ))
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'quality_score': quality_score,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        st.error(f"Error analizando imagen: {e}")
        return {'brightness': 0, 'contrast': 0, 'sharpness': 0, 'quality_score': 0}

def show_recommendations(analysis):
    """Muestra recomendaciones basadas en el análisis"""
    st.subheader("💡 Recomendaciones")
    
    recommendations = []
    
    if analysis['quality_score'] < 60:
        recommendations.append("❌ **Calidad insuficiente**: Toma otra foto con mejor iluminación y enfoque")
    elif analysis['quality_score'] < 80:
        recommendations.append("⚠️ **Calidad aceptable**: Podría mejorar con más luz")
    else:
        recommendations.append("✅ **Excelente calidad**: Adecuada para análisis detallado")
    
    if analysis['brightness'] < 100:
        recommendations.append("💡 **Poca luz**: Aumenta la iluminación o usa flash")
    elif analysis['brightness'] > 200:
        recommendations.append("🔆 **Demasiada luz**: Reduce brillo o evita reflejos")
    
    if analysis['contrast'] < 25:
        recommendations.append("🎨 **Bajo contraste**: Mejora la iluminación lateral")
    
    for rec in recommendations:
        st.write(rec)

def save_results(image, analysis):
    """Guarda los resultados"""
    try:
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cinta_reactiva_{timestamp}.jpg"
        
        # Guardar imagen
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name, 'JPEG', quality=95)
            
            # Ofrecer descarga
            with open(tmp_file.name, 'rb') as file:
                st.download_button(
                    label="📥 Descargar Foto Analizada",
                    data=file,
                    file_name=filename,
                    mime="image/jpeg"
                )
        
        # Limpiar archivo temporal
        os.unlink(tmp_file.name)
        
        st.success("✅ Análisis completado. Usa el botón de descarga above")
        
    except Exception as e:
        st.error(f"❌ Error guardando resultados: {e}")

if __name__ == "__main__":
    main()
