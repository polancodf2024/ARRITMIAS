import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tempfile
import os
import time
import logging
from datetime import datetime
import platform
import psutil
from typing import Optional, Dict, List, Tuple
import sys

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cinta_reactiva_analyzer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AdvancedCameraManager:
    def __init__(self):
        self.available_cameras = []
        self.current_camera = None
        self.cap = None
        self.camera_capabilities = {}
        
    def detect_cameras_with_retry(self, max_retries: int = 3) -> List[Dict]:
        self.available_cameras = []
        
        for camera_id in range(6):
            for attempt in range(max_retries):
                try:
                    cap = cv2.VideoCapture(camera_id)
                    if cap.isOpened():
                        time.sleep(0.5)
                        ret, frame = cap.read()
                        
                        if ret and frame is not None and frame.size > 0:
                            properties = self._get_camera_properties(cap, camera_id)
                            capabilities = self._detect_camera_capabilities(cap, camera_id)
                            properties['capabilities'] = capabilities
                            self.available_cameras.append(properties)
                            self.camera_capabilities[camera_id] = capabilities
                            logger.info(f"Cámara {camera_id} detectada")
                            break
                    
                    cap.release()
                    
                except Exception as e:
                    logger.error(f"Error detectando cámara {camera_id}: {e}")
                    if 'cap' in locals():
                        cap.release()
                
                time.sleep(0.2)
            
        return self.available_cameras
    
    def _detect_camera_capabilities(self, cap, camera_id: int) -> Dict:
        capabilities = {
            'flash': False,
            'anti_reflection': False,
            'macro_focus': False,
            'manual_focus': False,
        }
        
        try:
            test_properties = [
                (cv2.CAP_PROP_FLASH, 1, 'flash'),
                (cv2.CAP_PROP_AUTOFOCUS, 1, 'macro_focus'),
                (cv2.CAP_PROP_FOCUS, 40, 'manual_focus'),
            ]
            
            for prop, test_value, capability in test_properties:
                original_value = cap.get(prop)
                if cap.set(prop, test_value):
                    time.sleep(0.1)
                    new_value = cap.get(prop)
                    if abs(new_value - test_value) < 0.1:
                        capabilities[capability] = True
                    cap.set(prop, original_value)
            
            capabilities['anti_reflection'] = self._simulate_anti_reflection_detection(cap)
            capabilities['flash'] = self._test_flash_capability(cap)
            
        except Exception as e:
            logger.error(f"Error detectando capacidades: {e}")
        
        return capabilities
    
    def _simulate_anti_reflection_detection(self, cap) -> bool:
        try:
            has_advanced_controls = (
                cap.get(cv2.CAP_PROP_BRIGHTNESS) != -1 and
                cap.get(cv2.CAP_PROP_CONTRAST) != -1 and
                cap.get(cv2.CAP_PROP_SATURATION) != -1
            )
            return has_advanced_controls
        except:
            return False
    
    def _test_flash_capability(self, cap) -> bool:
        try:
            flash_properties = [cv2.CAP_PROP_FLASH, 39, 800]
            
            for prop in flash_properties:
                original = cap.get(prop)
                if cap.set(prop, 1):
                    time.sleep(0.1)
                    if cap.get(prop) == 1:
                        cap.set(prop, original)
                        return True
                cap.set(prop, original)
                
            return False
        except:
            return False
    
    def _get_camera_properties(self, cap, camera_id: int) -> Dict:
        try:
            properties = {
                'id': camera_id,
                'name': f"Cámara {camera_id}",
                'resolution': {
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                },
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'backend': cap.getBackendName(),
                'status': 'active'
            }
            
            if platform.system() == 'Windows':
                properties['name'] = f"Cámara Windows {camera_id}"
            elif platform.system() == 'Darwin':
                properties['name'] = f"Cámara macOS {camera_id}"
            else:
                properties['name'] = f"Cámara Linux {camera_id}"
                
            return properties
            
        except Exception as e:
            logger.error(f"Error obteniendo propiedades: {e}")
            return {
                'id': camera_id,
                'name': f"Cámara {camera_id}",
                'status': 'error',
                'error': str(e)
            }
    
    def initialize_camera(self, camera_id: int, resolution: Tuple[int, int] = (1920, 1080)) -> bool:
        try:
            if self.cap is not None:
                self.release_camera()
            
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"No se pudo abrir cámara {camera_id}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            
            time.sleep(1)
            
            self.current_camera = camera_id
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando cámara: {e}")
            return False
    
    def configure_advanced_macro_mode(self, use_flash: bool = True, use_anti_reflection: bool = True) -> Dict:
        if self.cap is None:
            return {'success': False, 'message': 'Cámara no inicializada'}
        
        config_results = {
            'flash_activated': False,
            'anti_reflection_activated': False,
            'macro_configured': False,
            'focus_optimized': False
        }
        
        try:
            if use_flash and self.camera_capabilities.get(self.current_camera, {}).get('flash', False):
                config_results['flash_activated'] = self._activate_flash()
            
            if use_anti_reflection and self.camera_capabilities.get(self.current_camera, {}).get('anti_reflection', False):
                config_results['anti_reflection_activated'] = self._activate_anti_reflection()
            
            config_results['macro_configured'] = self._configure_macro_focus()
            config_results['focus_optimized'] = self._optimize_focus_for_macro()
            
            return {
                'success': any(config_results.values()),
                'details': config_results
            }
            
        except Exception as e:
            logger.error(f"Error configurando modo macro: {e}")
            return {'success': False, 'error': str(e)}
    
    def _activate_flash(self) -> bool:
        try:
            flash_properties = [
                (cv2.CAP_PROP_FLASH, 1),
                (39, 1),
                (800, 1),
            ]
            
            for prop, value in flash_properties:
                if self.cap.set(prop, value):
                    time.sleep(0.5)
                    if self.cap.get(prop) == value:
                        logger.info("Flash activado exitosamente")
                        return True
            
            return self._activate_torch_mode()
            
        except Exception as e:
            logger.error(f"Error activando flash: {e}")
            return False
    
    def _activate_torch_mode(self) -> bool:
        try:
            torch_properties = [
                (cv2.CAP_PROP_FLASH, 2),
                (39, 2),
            ]
            
            for prop, value in torch_properties:
                if self.cap.set(prop, value):
                    time.sleep(0.3)
                    if self.cap.get(prop) == value:
                        logger.info("Modo linterna activado")
                        return True
            return False
        except:
            return False
    
    def _activate_anti_reflection(self) -> bool:
        try:
            strategies = [
                (cv2.CAP_PROP_BRIGHTNESS, 0.4),
                (cv2.CAP_PROP_CONTRAST, 0.7),
                (cv2.CAP_PROP_SATURATION, 0.6),
                (cv2.CAP_PROP_EXPOSURE, -0.5),
                (cv2.CAP_PROP_GAIN, 0),
            ]
            
            success_count = 0
            for prop, value in strategies:
                if self.cap.set(prop, value):
                    time.sleep(0.1)
                    success_count += 1
            
            return success_count > 2
            
        except Exception as e:
            logger.error(f"Error activando anti-reflejo: {e}")
            return False
    
    def _configure_macro_focus(self) -> bool:
        try:
            focus_strategies = [
                (cv2.CAP_PROP_AUTOFOCUS, 1),
                (cv2.CAP_PROP_FOCUS, 40),
                (38, 1),
            ]
            
            success_count = 0
            for prop, value in focus_strategies:
                if self.cap.set(prop, value):
                    success_count += 1
            
            time.sleep(1)
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error configurando enfoque: {e}")
            return False
    
    def _optimize_focus_for_macro(self) -> bool:
        try:
            for focus_attempt in range(3):
                focus_values = [30, 40, 50, 35, 45]
                
                for focus_val in focus_values:
                    if self.cap.set(cv2.CAP_PROP_FOCUS, focus_val):
                        time.sleep(0.3)
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            sharpness = self._calculate_sharpness(test_frame)
                            if sharpness > 50:
                                return True
                
                time.sleep(0.5)
            
            return False
            
        except Exception as e:
            logger.error(f"Error optimizando enfoque: {e}")
            return False
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var
        except:
            return 0
    
    def capture_advanced_macro_photo(self, stabilization_time: int = 3) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        
        try:
            start_time = time.time()
            
            while time.time() - start_time < stabilization_time:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    pass
                time.sleep(0.1)
            
            best_frame = None
            best_sharpness = 0
            
            for capture_attempt in range(3):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    sharpness = self._calculate_sharpness(frame)
                    if sharpness > best_sharpness:
                        best_sharpness = sharpness
                        best_frame = frame
                time.sleep(0.2)
            
            if best_frame is not None and self._validate_image_quality(best_frame):
                rgb_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
                return rgb_frame
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error capturando foto: {e}")
            return None
    
    def _validate_image_quality(self, image: np.ndarray) -> bool:
        try:
            if image is None or image.size == 0:
                return False
            
            mean_brightness = np.mean(image)
            if mean_brightness < 10 or mean_brightness > 245:
                return False
            
            std_dev = np.std(image)
            if std_dev < 10:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validando calidad: {e}")
            return False
    
    def release_camera(self):
        try:
            if self.cap is not None:
                flash_properties = [cv2.CAP_PROP_FLASH, 39, 800]
                for prop in flash_properties:
                    try:
                        self.cap.set(prop, 0)
                    except:
                        pass
                
                self.cap.release()
                self.cap = None
                self.current_camera = None
        except Exception as e:
            logger.error(f"Error liberando cámara: {e}")

class EnhancedImageProcessor:
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        try:
            pil_img = Image.fromarray(image)
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(1.1)
            return np.array(pil_img)
        except Exception as e:
            logger.error(f"Error mejorando imagen: {e}")
            return image
    
    @staticmethod
    def analyze_test_strip(image: np.ndarray) -> Dict:
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'brightness': float(np.mean(image)),
                'contrast': float(np.std(image)),
                'sharpness': 0.0,
                'color_balance': {},
                'quality_score': 0
            }
            
            for i, channel in enumerate(['Red', 'Green', 'Blue']):
                analysis['color_balance'][channel] = float(np.mean(image[:, :, i]))
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gy, gx = np.gradient(gray.astype(float))
            analysis['sharpness'] = float(np.sqrt(gx**2 + gy**2).mean())
            
            analysis['quality_score'] = min(100, max(0, 
                analysis['contrast'] * 0.4 + 
                analysis['sharpness'] * 2.5 + 
                (255 - abs(analysis['brightness'] - 128)) * 0.15
            ))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando imagen: {e}")
            return {}

class SystemMonitor:
    @staticmethod
    def get_system_info() -> Dict:
        try:
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'opencv_version': cv2.__version__
            }
        except Exception as e:
            logger.error(f"Error obteniendo info del sistema: {e}")
            return {}

def initialize_session_state():
    if 'camera_manager' not in st.session_state:
        st.session_state.camera_manager = AdvancedCameraManager()
    if 'image_processor' not in st.session_state:
        st.session_state.image_processor = EnhancedImageProcessor()
    if 'system_monitor' not in st.session_state:
        st.session_state.system_monitor = SystemMonitor()
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'camera_config' not in st.session_state:
        st.session_state.camera_config = {
            'use_flash': True,
            'use_anti_reflection': True,
            'high_quality': True,
            'auto_detect': True
        }

def main():
    st.set_page_config(
        page_title="Analizador de Cintas Reactivas",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .flash-active {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">🔬 Analizador de Cintas Reactivas</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        render_sidebar()
    
    tab1, tab2, tab3 = st.tabs(["🎯 Captura", "🔍 Análisis", "📊 Historial"])
    
    with tab1:
        render_capture_tab()
    
    with tab2:
        render_analysis_tab()
    
    with tab3:
        render_history_tab()

def render_sidebar():
    st.header("⚡ Configuración")
    
    st.session_state.camera_config['use_flash'] = st.checkbox(
        "Activar Flash Automático", 
        value=st.session_state.camera_config['use_flash']
    )
    
    st.session_state.camera_config['use_anti_reflection'] = st.checkbox(
        "Activar Anti-Reflejo", 
        value=st.session_state.camera_config['use_anti_reflection']
    )
    
    st.session_state.camera_config['high_quality'] = st.checkbox(
        "Alta Calidad", 
        value=st.session_state.camera_config['high_quality']
    )
    
    st.header("📊 Sistema")
    system_info = st.session_state.system_monitor.get_system_info()
    
    if system_info:
        st.metric("Python", system_info['python_version'])
        st.metric("OpenCV", system_info['opencv_version'])

def render_capture_tab():
    st.header("📸 Captura de Imágenes")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔍 Escanear Cámaras", type="primary"):
            with st.spinner("Buscando cámaras..."):
                cameras = st.session_state.camera_manager.detect_cameras_with_retry()
                
            if cameras:
                st.success(f"✅ Se detectaron {len(cameras)} cámaras")
                for cam in cameras:
                    if cam.get('status') == 'active':
                        with st.expander(f"📷 {cam['name']}"):
                            st.write(f"Resolución: {cam['resolution']['width']}x{cam['resolution']['height']}")
    
    with col2:
        camera_id = st.number_input("ID Cámara", min_value=0, max_value=5, value=0)
        stabilization = st.slider("Estabilización (seg)", 1, 10, 3)
        resolution = st.selectbox("Resolución", [
            (640, 480),
            (1280, 720), 
            (1920, 1080)
        ], index=2 if st.session_state.camera_config['high_quality'] else 1)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚡ CAPTURAR FOTO", type="primary", use_container_width=True):
            try:
                if st.session_state.camera_manager.initialize_camera(camera_id, resolution):
                    config_result = st.session_state.camera_manager.configure_advanced_macro_mode(
                        use_flash=st.session_state.camera_config['use_flash'],
                        use_anti_reflection=st.session_state.camera_config['use_anti_reflection']
                    )
                    
                    if config_result['success']:
                        details = config_result.get('details', {})
                        
                        if details.get('flash_activated', False):
                            st.markdown('<div class="flash-active">', unsafe_allow_html=True)
                            st.success("⚡ FLASH ACTIVADO")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        if details.get('anti_reflection_activated', False):
                            st.success("🛡️ ANTI-REFLECTION ACTIVADO")
                        
                        image = st.session_state.camera_manager.capture_advanced_macro_photo(stabilization)
                        
                        if image is not None:
                            st.session_state.current_image = image
                            st.session_state.current_image_enhanced = st.session_state.image_processor.enhance_image(image)
                            st.session_state.current_analysis = st.session_state.image_processor.analyze_test_strip(image)
                            
                            st.session_state.analysis_history.append({
                                'timestamp': datetime.now().isoformat(),
                                'analysis': st.session_state.current_analysis
                            })
                            
                            st.success("✅ Foto capturada exitosamente!")
                        else:
                            st.error("❌ Error al capturar la imagen")
                    else:
                        st.error("❌ Error en configuración de cámara")
                else:
                    st.error("❌ No se pudo inicializar la cámara")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with col2:
        if st.button("🛑 LIBERAR CÁMARA", use_container_width=True):
            st.session_state.camera_manager.release_camera()
            st.info("🔒 Cámara liberada")
    
    if 'current_image' in st.session_state:
        st.markdown("---")
        st.subheader("🖼️ Vista Previa")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(st.session_state.current_image, 
                    caption="Imagen Original", 
                    use_column_width=True)
        
        with col2:
            st.image(st.session_state.current_image_enhanced,
                    caption="Imagen Mejorada",
                    use_column_width=True)

def render_analysis_tab():
    st.header("🔍 Análisis Detallado")
    
    if 'current_analysis' not in st.session_state:
        st.info("📸 Capture una imagen primero para ver el análisis")
        return
    
    analysis = st.session_state.current_analysis
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Puntaje Calidad", f"{analysis.get('quality_score', 0):.1f}/100")
    
    with col2:
        st.metric("Brillo", f"{analysis.get('brightness', 0):.1f}")
    
    with col3:
        st.metric("Contraste", f"{analysis.get('contrast', 0):.1f}")
    
    with col4:
        st.metric("Nitidez", f"{analysis.get('sharpness', 0):.3f}")
    
    st.subheader("🎨 Balance de Color")
    if 'color_balance' in analysis:
        color_cols = st.columns(3)
        colors = ['Red', 'Green', 'Blue']
        for i, color in enumerate(colors):
            with color_cols[i]:
                value = analysis['color_balance'].get(color, 0)
                st.metric(f"{color}", f"{value:.1f}")

def render_history_tab():
    st.header("📊 Historial de Análisis")
    
    if not st.session_state.analysis_history:
        st.info("📝 No hay análisis en el historial")
        return
    
    for i, analysis_data in enumerate(reversed(st.session_state.analysis_history[-5:])):
        with st.expander(f"Análisis {i+1} - {analysis_data['timestamp'][:19]}"):
            if 'analysis' in analysis_data:
                analysis = analysis_data['analysis']
                st.metric("Calidad", f"{analysis.get('quality_score', 0):.1f}")
                st.metric("Brillo", f"{analysis.get('brightness', 0):.1f}")
                st.metric("Contraste", f"{analysis.get('contrast', 0):.1f}")

if __name__ == "__main__":
    main()
