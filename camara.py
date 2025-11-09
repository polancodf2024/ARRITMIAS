import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tempfile
import os
import time
from datetime import datetime

def mobile_camera_setup():
    """Configuración optimizada para móviles"""
    st.markdown("""
    <style>
    .mobile-header {
        font-size: 2rem;
        text-align: center;
        color: #1f77b4;
    }
    .mobile-button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

def main_mobile():
    st.set_page_config(
        page_title="Análisis Cintas Reactivas Móvil",
        page_icon="📱",
        layout="centered"
    )
    
    mobile_camera_setup()
    
    st.markdown('<h1 class="mobile-header">📱 Analizador Móvil</h1>', unsafe_allow_html=True)
    
    # Solución para móviles - Usar input de archivo
    st.warning("🔍 En dispositivos móviles, usa la cámara directamente y sube la foto")
    
    uploaded_file = st.camera_input("Toma una foto de la cinta reactiva")
    
    if uploaded_file is not None:
        # Procesar la imagen
        bytes_data = uploaded_file.getvalue()
        
        # Convertir a array numpy
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Mostrar imagen original
        st.subheader("📸 Foto Capturada")
        st.image(img_array, use_column_width=True)
        
        # Mejorar imagen
        enhanced_img = enhance_image_mobile(img_array)
        st.subheader("🔄 Imagen Mejorada")
        st.image(enhanced_img, use_column_width=True)
        
        # Análisis
        analysis = analyze_image_mobile(img_array)
        
        st.subheader("📊 Análisis de Calidad")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Calidad", f"{analysis['quality']}/100")
        with col2:
            st.metric("Brillo", f"{analysis['brightness']:.1f}")
        with col3:
            st.metric("Contraste", f"{analysis['contrast']:.1f}")
        
        # Recomendaciones
        st.subheader("💡 Recomendaciones")
        if analysis['quality'] < 70:
            st.error("❌ Calidad insuficiente. Toma otra foto con mejor iluminación")
        else:
            st.success("✅ Calidad adecuada para análisis")
        
        # Guardar opción
        if st.button("💾 Guardar Análisis"):
            save_analysis_mobile(analysis, img_array)

def enhance_image_mobile(img_array):
    """Mejorar imagen para móviles"""
    try:
        pil_img = Image.fromarray(img_array)
        
        # Mejorar contraste
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.3)
        
        # Mejorar nitidez
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.2)
        
        return np.array(pil_img)
    except:
        return img_array

def analyze_image_mobile(img_array):
    """Análisis simplificado para móviles"""
    try:
        # Convertir a escala de grises
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
            contrast * 0.5 + 
            sharpness * 3 + 
            (255 - abs(brightness - 128)) * 0.2
        ))
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness,
            'quality': quality_score,
            'timestamp': datetime.now().isoformat()
        }
    except:
        return {'brightness': 0, 'contrast': 0, 'sharpness': 0, 'quality': 0}

def save_analysis_mobile(analysis, image):
    """Guardar análisis en móvil"""
    try:
        # Crear imagen con timestamp
        pil_img = Image.fromarray(image)
        
        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            pil_img.save(tmp_file.name, 'JPEG', quality=95)
            
            # Ofrecer descarga
            with open(tmp_file.name, 'rb') as file:
                st.download_button(
                    label="📥 Descargar Imagen y Análisis",
                    data=file,
                    file_name=f"cinta_analisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    mime="image/jpeg"
                )
        
        # Limpiar
        os.unlink(tmp_file.name)
        
        st.success("✅ Análisis guardado correctamente")
    except Exception as e:
        st.error(f"❌ Error guardando: {e}")

# Ejecutar versión móvil
if __name__ == "__main__":
    main_mobile()
