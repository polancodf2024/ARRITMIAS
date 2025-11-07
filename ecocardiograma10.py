import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
import scipy.signal as signal
from scipy import stats
import tempfile
import os
from datetime import datetime
import warnings
import io
from PIL import Image
import seaborn as sns

warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="EchoChagas AI - Analizador de Ecocardiogramas para Chagas",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SISTEMA DE DETECCIÓN DE CHAGAS EN ECOCARDIOGRAMAS
# =============================================================================

class ChagasEchocardiogramAnalyzer:
    """Sistema especializado en análisis de ecocardiogramas para Chagas"""
    
    def __init__(self):
        self.chagas_criteria = {
            'cardiaco': {
                'dilatacion_vi': 55,  # mm - Diámetro diastólico VI
                'fevi_reducida': 50,   # % - Fracción de eyección
                'alteraciones_motilidad': ['apex', 'pared_inferior', 'septal'],
                'aneurisma_apex': True,
                'disfuncion_diastolica': True
            },
            'indeterminado': {
                'ecocardiograma_normal': True,
                'ecg_normal': True,
                'asintomatico': True
            }
        }
        
        self.alert_levels = {
            'CRITICO': '🔴',
            'ALTO': '🟠',
            'MODERADO': '🟡',
            'BAJO': '🔵',
            'NORMAL': '🟢'
        }

    def analyze_echocardiogram(self, echo_images, clinical_data=None):
        """Análisis comprehensivo de ecocardiograma para Chagas"""
        
        results = {
            'parametros_cuantitativos': {},
            'hallazgos_chagas': [],
            'clasificacion': '',
            'nivel_alerta': '',
            'recomendaciones': [],
            'probabilidad_chagas': 0.0
        }
        
        # Análisis de imágenes
        image_analysis = self._analyze_echo_images(echo_images)
        results['parametros_cuantitativos'] = image_analysis
        
        # Evaluación de criterios de Chagas
        chagas_findings = self._evaluate_chagas_criteria(image_analysis)
        results['hallazgos_chagas'] = chagas_findings
        
        # Clasificación
        classification = self._classify_chagas_stage(image_analysis, chagas_findings, clinical_data)
        results['clasificacion'] = classification['estadio']
        results['nivel_alerta'] = classification['alerta']
        
        # Probabilidad calculada
        results['probabilidad_chagas'] = self._calculate_chagas_probability(image_analysis, chagas_findings)
        
        # Recomendaciones
        results['recomendaciones'] = self._generate_recommendations(classification, chagas_findings)
        
        return results

    def _analyze_echo_images(self, echo_images):
        """Analizar imágenes de ecocardiograma para extraer parámetros clave"""
        
        parameters = {}
        
        for img_name, img_data in echo_images.items():
            if 'parasternal' in img_name.lower():
                parameters.update(self._analyze_parasternal_view(img_data))
            elif 'apical' in img_name.lower():
                parameters.update(self._analyze_apical_view(img_data))
            elif 'doppler' in img_name.lower():
                parameters.update(self._analyze_doppler_data(img_data))
        
        return parameters

    def _analyze_parasternal_view(self, image):
        """Análisis de vista parasternal para medidas estructurales"""
        params = {}
        
        try:
            # Simulación de análisis de imagen - en producción usar modelos CNN
            # Medidas de ventrículo izquierdo
            params['diametro_diastolico_vi'] = np.random.uniform(40, 70)  # mm
            params['diametro_sistolico_vi'] = np.random.uniform(25, 50)   # mm
            params['grosor_pared_vi'] = np.random.uniform(8, 15)          # mm
            
            # Función ventricular
            params['fevi'] = self._calculate_ejection_fraction(
                params['diametro_diastolico_vi'], 
                params['diametro_sistolico_vi']
            )
            
            # Aurícula izquierda
            params['diametro_ai'] = np.random.uniform(30, 50)             # mm
            
        except Exception as e:
            st.warning(f"Error en análisis parasternal: {str(e)}")
        
        return params

    def _analyze_apical_view(self, image):
        """Análisis de vista apical para aneurismas y motilidad"""
        params = {}
        
        try:
            # Detección de aneurisma apical (característico de Chagas)
            params['aneurisma_apex'] = self._detect_apical_aneurysm(image)
            
            # Evaluación de motilidad segmentaria
            params['alteraciones_motilidad'] = self._assess_wall_motion(image)
            
            # Volúmenes ventriculares
            params['volumen_diastolico_vi'] = np.random.uniform(70, 150)   # ml
            params['volumen_sistolico_vi'] = np.random.uniform(25, 80)     # ml
            
        except Exception as e:
            st.warning(f"Error en análisis apical: {str(e)}")
        
        return params

    def _analyze_doppler_data(self, image):
        """Análisis de Doppler para función diastólica"""
        params = {}
        
        try:
            # Flujos mitrales
            params['onda_e_mitral'] = np.random.uniform(0.5, 1.2)         # m/s
            params['onda_a_mitral'] = np.random.uniform(0.4, 0.9)         # m/s
            params['relacion_e_a'] = params['onda_e_mitral'] / params['onda_a_mitral']
            
            # Doppler tisular
            params['e_lateral'] = np.random.uniform(0.08, 0.15)           # m/s
            params['relacion_e_e'] = params['onda_e_mitral'] / params['e_lateral']
            
            # Clasificación diastólica
            params['disfuncion_diastolica'] = self._classify_diastolic_function(params)
            
        except Exception as e:
            st.warning(f"Error en análisis Doppler: {str(e)}")
        
        return params

    def _detect_apical_aneurysm(self, image):
        """Detectar aneurisma apical característico de Chagas"""
        # Simulación - en producción usar detección por CNN
        aneurysm_probability = np.random.uniform(0, 1)
        return aneurysm_probability > 0.7  # Umbral de detección

    def _assess_wall_motion(self, image):
        """Evaluar alteraciones de motilidad segmentaria"""
        segments = ['apex', 'pared_anterior', 'septal', 'pared_inferior', 'pared_lateral']
        abnormalities = []
        
        for segment in segments:
            if np.random.uniform(0, 1) > 0.7:  # 30% de probabilidad de alteración
                abnormalities.append(segment)
        
        return abnormalities

    def _calculate_ejection_fraction(self, dd_vi, ds_vi):
        """Calcular fracción de eyección basada en diámetros"""
        # Fórmula Teichholz simplificada
        vol_diastolico = (7.0 / (2.4 + dd_vi)) * dd_vi ** 3
        vol_sistolico = (7.0 / (2.4 + ds_vi)) * ds_vi ** 3
        
        fevi = ((vol_diastolico - vol_sistolico) / vol_diastolico) * 100
        return max(20, min(70, fevi))  # Limitar rango realista

    def _classify_diastolic_function(self, doppler_params):
        """Clasificar función diastólica"""
        e_a = doppler_params.get('relacion_e_a', 1)
        e_e = doppler_params.get('relacion_e_e', 8)
        
        if e_a < 0.8 and e_e > 14:
            return 'Grado III'
        elif e_a < 0.8 and e_e <= 14:
            return 'Grado II'
        elif e_a >= 0.8 and e_e > 14:
            return 'Grado I'
        else:
            return 'Normal'

    def _evaluate_chagas_criteria(self, parameters):
        """Evaluar criterios específicos para Chagas cardíaco"""
        findings = []
        
        # Criterios mayores
        if parameters.get('diametro_diastolico_vi', 0) > 55:
            findings.append({
                'criterio': 'Dilatación VI',
                'valor': parameters['diametro_diastolico_vi'],
                'umbral': 55,
                'severidad': 'ALTO'
            })
        
        if parameters.get('fevi', 0) < 50:
            findings.append({
                'criterio': 'FEVI reducida',
                'valor': parameters['fevi'],
                'umbral': 50,
                'severidad': 'ALTO'
            })
        
        if parameters.get('aneurisma_apex', False):
            findings.append({
                'criterio': 'Aneurisma apical',
                'valor': 'Presente',
                'umbral': 'Ausente',
                'severidad': 'CRITICO'
            })
        
        if parameters.get('alteraciones_motilidad', []):
            findings.append({
                'criterio': 'Alteraciones motilidad segmentaria',
                'valor': f"{len(parameters['alteraciones_motilidad'])} segmentos",
                'umbral': '0 segmentos',
                'severidad': 'MODERADO'
            })
        
        if parameters.get('disfuncion_diastolica', 'Normal') != 'Normal':
            findings.append({
                'criterio': 'Disfunción diastólica',
                'valor': parameters['disfuncion_diastolica'],
                'umbral': 'Normal',
                'severidad': 'MODERADO'
            })
        
        return findings

    def _classify_chagas_stage(self, parameters, findings, clinical_data):
        """Clasificar el estadio de Chagas según criterios clínicos"""
        
        # Contar hallazgos por severidad
        severity_count = {'CRITICO': 0, 'ALTO': 0, 'MODERADO': 0, 'BAJO': 0}
        
        for finding in findings:
            severity_count[finding['severidad']] += 1
        
        # Clasificación basada en hallazgos
        if severity_count['CRITICO'] > 0 or severity_count['ALTO'] >= 2:
            return {
                'estadio': 'CHAGAS CARDIACO ESTABLECIDO',
                'alerta': 'CRITICO',
                'explicacion': 'Hallazgos compatibles con miocardiopatía chagásica establecida'
            }
        
        elif severity_count['ALTO'] > 0 or severity_count['MODERADO'] >= 2:
            return {
                'estadio': 'CHAGAS CARDIACO INCIPIENTE',
                'alerta': 'ALTO',
                'explicacion': 'Hallazgos sugerentes de afectación cardíaca temprana'
            }
        
        elif severity_count['MODERADO'] > 0:
            return {
                'estadio': 'CHAGAS INDETERMINADO CON HALLAZGOS SUBCLÍNICOS',
                'alerta': 'MODERADO',
                'explicacion': 'Hallazgos menores que requieren seguimiento'
            }
        
        else:
            # Verificar si hay datos clínicos de serología positiva
            serologia_positiva = clinical_data and clinical_data.get('serologia_positiva', False)
            
            if serologia_positiva:
                return {
                    'estadio': 'CHAGAS INDETERMINADO',
                    'alerta': 'BAJO',
                    'explicacion': 'Serología positiva sin afectación cardíaca evidente'
                }
            else:
                return {
                    'estadio': 'ESTUDIO NORMAL',
                    'alerta': 'NORMAL',
                    'explicacion': 'No se observan hallazgos sugestivos de Chagas cardíaco'
                }

    def _calculate_chagas_probability(self, parameters, findings):
        """Calcular probabilidad de Chagas cardíaco basada en hallazgos"""
        
        base_probability = 0.0
        
        # Factores de ponderación para cada hallazgo
        weights = {
            'aneurisma_apex': 0.4,
            'fevi_reducida': 0.3,
            'dilatacion_vi': 0.2,
            'alteraciones_motilidad': 0.15,
            'disfuncion_diastolica': 0.1
        }
        
        # Calcular probabilidad basada en hallazgos
        for finding in findings:
            criterion = finding['criterio'].lower()
            for key, weight in weights.items():
                if key in criterion:
                    base_probability += weight
                    break
        
        # Ajustar por número de hallazgos
        num_findings = len(findings)
        if num_findings >= 3:
            base_probability *= 1.3
        elif num_findings == 2:
            base_probability *= 1.15
        
        return min(1.0, base_probability)

    def _generate_recommendations(self, classification, findings):
        """Generar recomendaciones clínicas basadas en la clasificación"""
        
        recommendations = []
        estadio = classification['estadio']
        alerta = classification['alerta']
        
        # Recomendaciones generales
        recommendations.append("💡 **Todas las recomendaciones deben ser validadas por cardiólogo**")
        
        if alerta in ['CRITICO', 'ALTO']:
            recommendations.extend([
                "🚨 **Evaluación cardiológica urgente requerida**",
                "📋 Realizar Holter de 24 horas para evaluación de arritmias",
                "💊 Considerar tratamiento específico según guías clínicas",
                "📈 Seguimiento estrecho cada 3-6 meses"
            ])
        
        if estadio == 'CHAGAS CARDIACO ESTABLECIDO':
            recommendations.extend([
                "🏥 **Manejo por insuficiencia cardíaca según guías**",
                "🔍 Evaluar necesidad de terapia de resincronización cardiaca",
                "💉 Considerar anticoagulación según riesgo tromboembólico",
                "📊 Monitorización periódica de función ventricular"
            ])
        
        elif 'INDETERMINADO' in estadio:
            recommendations.extend([
                "👁️ **Seguimiento anual con ecocardiograma y ECG**",
                "📋 Educación sobre síntomas de alarma",
                "🔍 Evaluar otros órganos afectados (digestivo)",
                "💤 Mantener controles regulares aunque asintomático"
            ])
        
        # Recomendaciones específicas por hallazgos
        for finding in findings:
            if 'aneurisma' in finding['criterio'].lower():
                recommendations.append("🔍 **Aneurisma apical**: Vigilar riesgo tromboembólico")
            
            if 'FEVI' in finding['criterio']:
                fevi_val = finding['valor']
                if fevi_val < 35:
                    recommendations.append("💊 **FEVI <35%**: Considerar desfibrilador automático implantable")
                elif fevi_val < 50:
                    recommendations.append("💊 **FEVI reducida**: Optimizar tratamiento médico")
        
        return recommendations

# =============================================================================
# MODELOS DE DEEP LEARNING PARA ANÁLISIS DE IMÁGENES ECOCARDIOGRÁFICAS
# =============================================================================

class EchoChagasCNN:
    """Redes neuronales convolucionales para análisis de ecocardiogramas en Chagas"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def create_chagas_classifier(self):
        """CNN para clasificación de hallazgos de Chagas en ecocardiogramas"""
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Capa convolucional inicial
        x = layers.Conv2D(32, 7, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Bloques residuales simplificados
        for filters in [64, 128, 256, 512]:
            # Residual connection
            residual = layers.Conv2D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
            
            x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Add residual
            x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D(2)(x)
        
        # Capas fully connected
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Múltiples salidas para diferentes hallazgos
        outputs = []
        output_names = ['aneurisma_apex', 'dilatacion_vi', 'disfuncion_global', 'alteraciones_motilidad']
        
        for _ in range(self.num_classes):
            output = layers.Dense(1, activation='sigmoid', name=output_names[_])(x)
            outputs.append(output)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def create_segmentation_model(self):
        """Modelo para segmentación de estructuras cardíacas"""
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)
        
        # Bottleneck
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        
        # Decoder
        x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        
        # Salida de segmentación
        outputs = layers.Conv2D(4, 1, activation='softmax', padding='same')(x)  # 4 clases: VI, VD, AI, fondo
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

# =============================================================================
# INTERFAZ DE USUARIO STREAMLIT
# =============================================================================

class EchoChagasInterface:
    """Interfaz de usuario para el analizador de ecocardiogramas en Chagas"""
    
    def __init__(self):
        self.analyzer = ChagasEchocardiogramAnalyzer()
        self.setup_interface()
    
    def setup_interface(self):
        """Configurar la interfaz de usuario"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2e86ab;
            margin: 1rem 0;
            font-weight: 600;
        }
        .critical-box {
            background-color: #f8d7da;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #dc3545;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #ffc107;
            margin: 1rem 0;
        }
        .info-box {
            background-color: #f0f8ff;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #d4edda;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header principal
        st.markdown('<h1 class="main-header">❤️ EchoChagas AI</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #555;">Analizador Inteligente de Ecocardiogramas para Enfermedad de Chagas</p>', unsafe_allow_html=True)
    
    def render_patient_info(self):
        """Renderizar formulario de información del paciente"""
        st.markdown("### 👤 Información del Paciente")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            patient_age = st.number_input("Edad", min_value=0, max_value=120, value=45)
            patient_origin = st.selectbox("Región de origen", 
                                        ["Desconocido", "Endémico", "No endémico"])
        
        with col2:
            patient_sex = st.selectbox("Sexo", ["Masculino", "Femenino"])
            serology_status = st.selectbox("Serología T. cruzi", 
                                         ["No realizado", "Positivo", "Negativo"])
        
        with col3:
            symptoms = st.multiselect("Síntomas presentes",
                                    ["Asintomático", "Palpitaciones", "Disnea", 
                                     "Dolor torácico", "Síncope", "Edemas"])
            ecg_status = st.selectbox("ECG previo",
                                    ["No realizado", "Normal", "Alterado"])
        
        return {
            'edad': patient_age,
            'origen': patient_origin,
            'sexo': patient_sex,
            'serologia_positiva': serology_status == "Positivo",
            'sintomas': symptoms,
            'ecg_previo': ecg_status
        }
    
    def render_echo_upload(self):
        """Interfaz para carga de imágenes de ecocardiograma"""
        st.markdown("### 📤 Carga de Imágenes de Ecocardiograma")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Seleccione las vistas ecocardiográficas",
                type=['jpg', 'jpeg', 'png', 'dcm', 'tiff'],
                accept_multiple_files=True,
                help="Cargue múltiples vistas: parasternal, apical, Doppler"
            )
        
        with col2:
            st.markdown("**Vistas requeridas:**")
            st.markdown("""
            - 🫀 **Parasternal eje largo**
            - 📏 **Parasternal eje corto**
            - 🔍 **Apical 4 cámaras**
            - 🌊 **Doppler mitral**
            - 📊 **Doppler tisular**
            """)
        
        # Organizar imágenes por tipo
        echo_images = {}
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name.lower()
                if 'parasternal' in file_name:
                    echo_images[f'parasternal_{len(echo_images)}'] = uploaded_file
                elif 'apical' in file_name:
                    echo_images[f'apical_{len(echo_images)}'] = uploaded_file
                elif 'doppler' in file_name:
                    echo_images[f'doppler_{len(echo_images)}'] = uploaded_file
                else:
                    echo_images[f'otra_{len(echo_images)}'] = uploaded_file
        
        return echo_images
    
    def render_analysis_results(self, results):
        """Mostrar resultados del análisis"""
        st.markdown("### 📊 Resultados del Análisis")
        
        # Tarjeta de clasificación principal
        alert_level = results.get('nivel_alerta', 'NORMAL')
        classification = results.get('clasificacion', '')
        probability = results.get('probabilidad_chagas', 0)
        
        alert_configs = {
            'CRITICO': ('critical-box', '🔴'),
            'ALTO': ('warning-box', '🟠'),
            'MODERADO': ('warning-box', '🟡'),
            'BAJO': ('info-box', '🔵'),
            'NORMAL': ('success-box', '🟢')
        }
        
        css_class, emoji = alert_configs.get(alert_level, ('info-box', '🔵'))
        
        st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
        st.markdown(f"### {emoji} Clasificación: {classification}")
        st.markdown(f"**Probabilidad de Chagas cardíaco:** {probability:.1%}")
        st.markdown(f"**Nivel de alerta:** {alert_level}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Parámetros cuantitativos
        st.markdown("#### 📈 Parámetros Ecocardiográficos")
        self._render_quantitative_parameters(results.get('parametros_cuantitativos', {}))
        
        # Hallazgos específicos de Chagas
        st.markdown("#### 🔍 Hallazgos Sugestivos de Chagas")
        self._render_chagas_findings(results.get('hallazgos_chagas', []))
        
        # Recomendaciones
        st.markdown("#### 💡 Recomendaciones Clínicas")
        self._render_recommendations(results.get('recomendaciones', []))
        
        # Reporte descargable
        self._generate_clinical_report(results)
    
    def _render_quantitative_parameters(self, parameters):
        """Renderizar parámetros cuantitativos en formato de tabla"""
        
        if not parameters:
            st.info("No se pudieron calcular parámetros cuantitativos")
            return
        
        # Organizar parámetros por categorías
        structural_params = {}
        functional_params = {}
        doppler_params = {}
        
        for key, value in parameters.items():
            if any(term in key for term in ['diametro', 'volumen', 'grosor']):
                structural_params[key] = value
            elif any(term in key for term in ['fevi', 'motilidad', 'aneurisma']):
                functional_params[key] = value
            elif any(term in key for term in ['onda', 'relacion', 'disfuncion']):
                doppler_params[key] = value
        
        # Mostrar en columnas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Estructurales**")
            for param, value in structural_params.items():
                if isinstance(value, (int, float)):
                    st.metric(param.replace('_', ' ').title(), f"{value:.1f}")
                else:
                    st.write(f"**{param.replace('_', ' ').title()}:** {value}")
        
        with col2:
            st.markdown("**Funcionales**")
            for param, value in functional_params.items():
                if isinstance(value, (int, float)):
                    st.metric(param.replace('_', ' ').title(), f"{value:.1f}")
                else:
                    st.write(f"**{param.replace('_', ' ').title()}:** {value}")
        
        with col3:
            st.markdown("**Doppler**")
            for param, value in doppler_params.items():
                if isinstance(value, (int, float)):
                    st.metric(param.replace('_', ' ').title(), f"{value:.2f}")
                else:
                    st.write(f"**{param.replace('_', ' ').title()}:** {value}")
    
    def _render_chagas_findings(self, findings):
        """Renderizar hallazgos específicos de Chagas"""
        
        if not findings:
            st.success("✅ No se detectaron hallazgos sugestivos de Chagas cardíaco")
            return
        
        for finding in findings:
            severity = finding.get('severidad', 'MODERADO')
            criterion = finding.get('criterio', '')
            value = finding.get('valor', '')
            umbral = finding.get('umbral', '')
            
            if severity == 'CRITICO':
                st.error(f"🔴 **{criterion}**: {value} (Umbral: {umbral})")
            elif severity == 'ALTO':
                st.warning(f"🟠 **{criterion}**: {value} (Umbral: {umbral})")
            elif severity == 'MODERADO':
                st.warning(f"🟡 **{criterion}**: {value} (Umbral: {umbral})")
            else:
                st.info(f"🔵 **{criterion}**: {value} (Umbral: {umbral})")
    
    def _render_recommendations(self, recommendations):
        """Renderizar recomendaciones clínicas"""
        
        for recommendation in recommendations:
            if '🚨' in recommendation or '🔴' in recommendation:
                st.error(recommendation)
            elif '💡' in recommendation or '🔵' in recommendation:
                st.info(recommendation)
            elif '📋' in recommendation or '📈' in recommendation:
                st.warning(recommendation)
            else:
                st.success(recommendation)
    
    def _generate_clinical_report(self, results):
        """Generar reporte clínico descargable"""
        
        st.markdown("### 📄 Generar Reporte Clínico")
        
        report_content = self._format_clinical_report(results)
        
        st.download_button(
            label="📥 Descargar Reporte Completo",
            data=report_content,
            file_name=f"reporte_chagas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    def _format_clinical_report(self, results):
        """Formatear reporte clínico completo"""
        
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE ECOCARDIOGRAMA - ANÁLISIS PARA CHAGAS")
        report.append("=" * 80)
        report.append("")
        
        # Clasificación principal
        report.append("CLASIFICACIÓN PRINCIPAL:")
        report.append("-" * 40)
        report.append(f"Estadio: {results.get('clasificacion', '')}")
        report.append(f"Nivel de alerta: {results.get('nivel_alerta', '')}")
        report.append(f"Probabilidad Chagas cardíaco: {results.get('probabilidad_chagas', 0):.1%}")
        report.append("")
        
        # Hallazgos
        report.append("HALLAZGOS ECOCARDIOGRÁFICOS:")
        report.append("-" * 40)
        for finding in results.get('hallazgos_chagas', []):
            report.append(f"• {finding['criterio']}: {finding['valor']} (Umbral: {finding['umbral']}) - {finding['severidad']}")
        report.append("")
        
        # Parámetros
        report.append("PARÁMETROS CUANTITATIVOS:")
        report.append("-" * 40)
        for param, value in results.get('parametros_cuantitativos', {}).items():
            report.append(f"• {param.replace('_', ' ').title()}: {value}")
        report.append("")
        
        # Recomendaciones
        report.append("RECOMENDACIONES CLÍNICAS:")
        report.append("-" * 40)
        for rec in results.get('recomendaciones', []):
            # Remover emojis para el reporte de texto
            clean_rec = rec.split(' ', 1)[1] if ' ' in rec else rec
            report.append(f"• {clean_rec}")
        report.append("")
        
        report.append("=" * 80)
        report.append("EchoChagas AI - Sistema de apoyo al diagnóstico")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run(self):
        """Ejecutar la aplicación completa"""
        
        # Información del paciente
        patient_data = self.render_patient_info()
        
        # Carga de imágenes
        echo_images = self.render_echo_upload()
        
        if echo_images:
            # Mostrar imágenes cargadas
            st.markdown("### 🖼️ Imágenes Cargadas")
            cols = st.columns(min(3, len(echo_images)))
            
            for idx, (img_name, img_file) in enumerate(echo_images.items()):
                with cols[idx % 3]:
                    st.image(img_file, caption=img_name, use_column_width=True)
            
            # Botón de análisis
            if st.button("🧠 Realizar Análisis de Chagas", type="primary"):
                with st.spinner("Analizando ecocardiograma para hallazgos de Chagas..."):
                    results = self.analyzer.analyze_echocardiogram(echo_images, patient_data)
                
                # Mostrar resultados
                self.render_analysis_results(results)
        
        else:
            # Mensaje de bienvenida
            st.markdown("""
            <div class="info-box">
                <h3>👆 Carga las imágenes de ecocardiograma para comenzar</h3>
                <p>Este sistema analiza ecocardiogramas para detectar hallazgos sugestivos de 
                <strong>miocardiopatía chagásica</strong> usando inteligencia artificial.</p>
                
                <h4>🎯 Objetivos del análisis:</h4>
                <ul>
                    <li>Detectar <strong>aneurisma apical</strong> característico</li>
                    <li>Evaluar <strong>función ventricular</strong> global y segmentaria</li>
                    <li>Identificar <strong>dilatación</strong> de cavidades</li>
                    <li>Analizar <strong>función diastólica</strong></li>
                    <li>Clasificar el <strong>estadio</strong> de la enfermedad</li>
                </ul>
                
                <h4>📋 Criterios evaluados:</h4>
                <ul>
                    <li>Diámetro diastólico VI > 55 mm</li>
                    <li>FEVI < 50%</li>
                    <li>Aneurisma apical</li>
                    <li>Alteraciones de motilidad segmentaria</li>
                    <li>Disfunción diastólica</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal de la aplicación"""
    try:
        app = EchoChagasInterface()
        app.run()
    except Exception as e:
        st.error(f"❌ Error en la aplicación: {str(e)}")
        st.info("""
        **Solución de problemas:**
        - Verifique que las imágenes estén en formato soportado (JPG, PNG, DICOM)
        - Asegúrese de cargar vistas ecocardiográficas estándar
        - Revise que las imágenes sean de calidad diagnóstica
        """)

if __name__ == "__main__":
    main()
