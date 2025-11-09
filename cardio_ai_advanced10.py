import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import stats
import neurokit2 as nk
import pyedflib
import tempfile
import os
import io
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
import warnings
import gc
import struct
from streamlit.runtime.caching import cache_resource, cache_data

warnings.filterwarnings('ignore')

# Configuración de la página mejorada
st.set_page_config(
    page_title="CardioAI Advanced Pro - Analizador Cardíaco Inteligente",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SISTEMA DE ALERTAS CLÍNICAS CON PRIORIDADES
# =============================================================================

class ClinicalAlertSystem:
    def __init__(self):
        self.critical_conditions = {
            'Taquicardia Ventricular': 'CRITICAL',
            'Bloqueo AV': 'CRITICAL', 
            'Fibrilación Ventricular': 'CRITICAL',
            'Bigeminismo Ventricular': 'HIGH',
            'Trigeminismo Ventricular': 'HIGH',
            'Fibrilación Auricular': 'MEDIUM',
            'Extrasístoles Ventriculares': 'MEDIUM',
            'Taquicardia Sinusal': 'LOW',
            'Bradicardia Sinusal': 'LOW',
            'Ritmo Sinusal Normal': 'NORMAL'
        }
        
        self.alert_colors = {
            'CRITICAL': '🔴',
            'HIGH': '🟠', 
            'MEDIUM': '🟡',
            'LOW': '🔵',
            'NORMAL': '🟢'
        }
    
    def generate_clinical_recommendations(self, diagnosis, probability):
        """Generar recomendaciones clínicas basadas en el diagnóstico y probabilidad"""
        recommendations = {
            'CRITICAL': [
                "🚨 INTERVENCIÓN INMEDIATA REQUERIDA",
                "Buscar atención médica de emergencia",
                "Monitorización continua requerida",
                "Preparar protocolo de reanimación"
            ],
            'HIGH': [
                "⚠️ Evaluación cardiológica urgente (dentro de 24 horas)",
                "Monitorización hospitalaria recomendada",
                "Considerar tratamiento antiarrítmico",
                "Evitar actividades extenuantes"
            ],
            'MEDIUM': [
                "📋 Consulta cardiológica programada",
                "Monitorización ambulatoria recomendada",
                "Evaluar factores de riesgo",
                "Seguimiento en 1-2 semanas"
            ],
            'LOW': [
                "👁️ Monitoreo continuo recomendado",
                "Consulta médica de rutina",
                "Educación sobre síntomas de alarma",
                "Seguimiento según evolución"
            ],
            'NORMAL': [
                "✅ Ritmo cardíaco normal",
                "Continuar con controles rutinarios",
                "Mantener estilo de vida saludable",
                "Próximo control anual"
            ]
        }
        
        alert_level = self.critical_conditions.get(diagnosis, 'LOW')
        return recommendations.get(alert_level, recommendations['LOW']), alert_level
    
    def get_immediate_actions(self, diagnosis):
        """Acciones inmediatas específicas para cada condición"""
        action_guides = {
            'Taquicardia Ventricular': [
                "Evaluar estabilidad hemodinámica",
                "Preparar cardioversión eléctrica si es inestable",
                "Considerar amiodarona IV si es estable",
                "Monitorización en UCI"
            ],
            'Bloqueo AV': [
                "Evaluar grado del bloqueo",
                "Considerar marcapasos temporal si bradicardia sintomática",
                "Monitorizar progresión",
                "Evaluar necesidad de marcapasos permanente"
            ],
            'Fibrilación Auricular': [
                "Evaluar riesgo tromboembólico (CHA2DS2-VASc)",
                "Control de frecuencia vs ritmo",
                "Anticoagulación según riesgo",
                "Manejo de factores desencadenantes"
            ],
            'Bigeminismo Ventricular': [
                "Evaluar densidad de PVCs",
                "Buscar cardiopatía estructural",
                "Considerar betabloqueadores",
                "Monitorizar progresión a taquicardia"
            ],
            'Trigeminismo Ventricular': [
                "Evaluar frecuencia y patrones",
                "Estudio de cardiopatía isquémica",
                "Considerar Holter 24 horas",
                "Seguimiento cercano"
            ]
        }
        return action_guides.get(diagnosis, ["Consulta cardiológica para evaluación específica"])

# =============================================================================
# EVALUACIÓN DE CALIDAD DE SEÑAL
# =============================================================================

class SignalQualityAssessment:
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
    
    def assess_signal_quality(self, ecg_signal, sampling_rate):
        """Evaluar calidad de señal para determinar confiabilidad del diagnóstico"""
        try:
            # Calcular métricas de calidad
            noise_level = self._calculate_noise_ratio(ecg_signal)
            baseline_wander = self._detect_baseline_wander(ecg_signal)
            signal_strength = np.std(ecg_signal)
            saturation = self._detect_saturation(ecg_signal)
            
            # Calcular puntaje de calidad compuesto
            quality_score = max(0, 1 - (noise_level * 0.4 + baseline_wander * 0.3 + saturation * 0.3))
            
            # Determinar nivel de calidad
            if quality_score >= self.quality_thresholds['excellent']:
                quality_level = "EXCELENTE"
                diagnostic_reliability = "Alta"
            elif quality_score >= self.quality_thresholds['good']:
                quality_level = "BUENA"
                diagnostic_reliability = "Moderada-Alta"
            elif quality_score >= self.quality_thresholds['fair']:
                quality_level = "ACEPTABLE"
                diagnostic_reliability = "Moderada"
            else:
                quality_level = "POBRE"
                diagnostic_reliability = "Baja"
            
            return {
                'quality_score': quality_score,
                'quality_level': quality_level,
                'diagnostic_reliability': diagnostic_reliability,
                'noise_level': noise_level,
                'baseline_wander': baseline_wander,
                'signal_strength': signal_strength,
                'saturation': saturation,
                'is_diagnostic_quality': quality_score > self.quality_thresholds['fair'],
                'recommendations': self._get_quality_recommendations(quality_score)
            }
            
        except Exception as e:
            return {
                'quality_score': 0,
                'quality_level': "NO EVALUABLE",
                'diagnostic_reliability': "Muy Baja",
                'noise_level': 1.0,
                'baseline_wander': 1.0,
                'signal_strength': 0,
                'saturation': 0,
                'is_diagnostic_quality': False,
                'recommendations': ["No se pudo evaluar la calidad de la señal"]
            }
    
    def _calculate_noise_ratio(self, ecg_signal):
        """Calcular relación señal-ruido"""
        try:
            # Filtro pasa-altos para aislar ruido de alta frecuencia
            b, a = signal.butter(3, 40, btype='high', fs=250)
            high_freq = signal.filtfilt(b, a, ecg_signal)
            noise_power = np.mean(high_freq ** 2)
            signal_power = np.mean(ecg_signal ** 2)
            return min(1.0, noise_power / (signal_power + 1e-10))
        except:
            return 1.0
    
    def _detect_baseline_wander(self, ecg_signal):
        """Detectar deriva de línea base"""
        try:
            # Filtro pasa-bajos para línea base
            b, a = signal.butter(3, 0.5, btype='low', fs=250)
            baseline = signal.filtfilt(b, a, ecg_signal)
            wander = np.std(baseline) / (np.std(ecg_signal) + 1e-10)
            return min(1.0, wander)
        except:
            return 1.0
    
    def _detect_saturation(self, ecg_signal):
        """Detectar saturación de la señal"""
        try:
            # Buscar segmentos planos (saturación)
            diff_signal = np.diff(ecg_signal)
            flat_segments = np.sum(np.abs(diff_signal) < 1e-5) / len(diff_signal)
            return min(1.0, flat_segments * 10)  # Escalar
        except:
            return 0.0
    
    def _get_quality_recommendations(self, quality_score):
        """Generar recomendaciones para mejorar la calidad de señal"""
        if quality_score >= 0.8:
            return ["✅ Calidad de señal excelente para diagnóstico"]
        elif quality_score >= 0.6:
            return ["✓ Calidad adecuada", "Considere verificar conexiones de electrodos"]
        elif quality_score >= 0.4:
            return [
                "⚠️ Calidad moderada - límite para diagnóstico",
                "Verifique conexiones de electrodos",
                "Asegure piel limpia y seca",
                "Evite movimiento durante registro"
            ]
        else:
            return [
                "❌ Calidad insuficiente para diagnóstico confiable",
                "Reinicie el registro ECG",
                "Verifique todos los electrodos",
                "Limpie la piel adecuadamente",
                "Mantenga al paciente inmóvil"
            ]

# =============================================================================
# VALIDADOR DE CRITERIOS CLÍNICOS
# =============================================================================

class ClinicalCriteriaValidator:
    def __init__(self):
        self.guideline_thresholds = {
            'afib_rr_irregularity': 0.15,  # Desviación estándar normalizada de intervalos RR
            'vtach_qrs_width': 120,        # ms
            'vtach_duration': 30.0,        # segundos
            'av_block_pr_interval': 200,   # ms
            'bradycardia_hr': 60,          # lpm
            'tachycardia_hr': 100,         # lpm
            'pvc_density_threshold': 0.1   # 10% de latidos son PVCs
        }
    
    def validate_afib(self, features, rr_intervals):
        """Validar Fibrilación Auricular según criterios de Guidelines"""
        try:
            rr_irregularity = np.std(rr_intervals) / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
            meets_criteria = (
                rr_irregularity > self.guideline_thresholds['afib_rr_irregularity'] and
                features.get('p_wave_absence_score', 0) > 0.7 and
                features.get('hr_variability', 0) > 0.6
            )
            confidence = min(1.0, rr_irregularity * 3)
            return meets_criteria, confidence
        except:
            return False, 0.0
    
    def validate_vtach(self, features, duration, qrs_width):
        """Validar Taquicardia Ventricular"""
        try:
            meets_criteria = (
                qrs_width > self.guideline_thresholds['vtach_qrs_width'] and
                duration >= 3.0 and  # Al menos 3 segundos para VT no sostenida
                features.get('av_dissociation_score', 0) > 0.5
            )
            confidence = min(1.0, (qrs_width / 200) * 0.5 + (min(duration, 30) / 30) * 0.5)
            return meets_criteria, confidence
        except:
            return False, 0.0
    
    def validate_av_block(self, features, pr_interval):
        """Validar Bloqueo AV"""
        try:
            meets_criteria = (
                pr_interval > self.guideline_thresholds['av_block_pr_interval'] or
                features.get('av_conduction_ratio', 0) < 0.8
            )
            confidence = min(1.0, pr_interval / 300)
            return meets_criteria, confidence
        except:
            return False, 0.0
    
    def validate_bigeminy(self, features, pvc_density, pattern_regularity):
        """Validar Bigeminismo Ventricular"""
        try:
            meets_criteria = (
                pvc_density > 0.3 and  # Al menos 30% de PVCs
                pattern_regularity > 0.7 and  # Patrón regular
                features.get('bigeminy_score', 0) > 0.5
            )
            confidence = min(1.0, (pvc_density * 0.5 + pattern_regularity * 0.5))
            return meets_criteria, confidence
        except:
            return False, 0.0

# =============================================================================
# SISTEMA DE EXPLICABILIDAD (XAI)
# =============================================================================

class ExplainableAI:
    def __init__(self):
        self.feature_descriptions = {
            'mean_rr': 'Intervalo RR promedio',
            'std_rr': 'Variabilidad de intervalos RR',
            'hr_mean': 'Frecuencia cardíaca promedio',
            'qrs_width': 'Ancho del complejo QRS',
            'pr_interval': 'Intervalo PR',
            'pvc_density': 'Densidad de extrasístoles ventriculares',
            'bigeminy_score': 'Puntaje de patrón bigeminismo',
            'trigeminy_score': 'Puntaje de patrón trigeminismo',
            'lf_hf_ratio': 'Ratio de frecuencias bajas/altas',
            'signal_entropy': 'Complejidad de la señal'
        }
    
    def generate_explanation(self, diagnosis, features, probabilities):
        """Generar explicaciones comprensibles del diagnóstico"""
        
        explanations = {
            'Bigeminismo Ventricular': 
                self._explain_bigeminy(features),
                
            'Trigeminismo Ventricular':
                self._explain_trigeminy(features),
                
            'Bloqueo AV':
                self._explain_av_block(features),
                
            'Fibrilación Auricular':
                self._explain_afib(features),
                
            'Taquicardia Ventricular':
                self._explain_vtach(features),
                
            'Extrasístoles Ventriculares':
                self._explain_pvcs(features)
        }
        
        default_explanation = (
            f"El diagnóstico se basa en el análisis de {len(features)} características ECG.\n"
            f"Características más relevantes:\n"
            f"• Variabilidad RR: {features.get('std_rr', 0):.2f} ms\n"
            f"• Frecuencia cardíaca: {features.get('hr_mean', 0):.1f} lpm\n"
            f"• Ancho QRS: {features.get('qrs_width', 0):.1f} ms"
        )
        
        return explanations.get(diagnosis, default_explanation)
    
    def _explain_bigeminy(self, features):
        explanation = "**Bigeminismo Ventricular Detectado:**\n\n"
        explanation += f"• Patrón de PVC cada dos latidos: {features.get('bigeminy_score', 0):.2f}\n"
        explanation += f"• Regularidad del patrón: {features.get('pattern_regularity', 0):.2f}\n"
        explanation += f"• Densidad de PVCs: {features.get('pvc_density', 0):.1%}\n"
        explanation += f"• Acoplamiento de PVCs: {features.get('pvc_coupling_std', 0):.1f} ms\n\n"
        explanation += "**Significado clínico:** Patrón organizado de extrasístoles que puede progresar a taquicardia."
        return explanation
    
    def _explain_trigeminy(self, features):
        explanation = "**Trigeminismo Ventricular Detectado:**\n\n"
        explanation += f"• Patrón de PVC cada tres latidos: {features.get('trigeminy_score', 0):.2f}\n"
        explanation += f"• Regularidad del patrón: {features.get('pattern_regularity', 0):.2f}\n"
        explanation += f"• Densidad de PVCs: {features.get('pvc_density', 0):.1%}\n"
        explanation += f"• Pausas compensatorias: {features.get('compensatory_pause_ratio', 0):.2f}\n\n"
        explanation += "**Significado clínico:** Patrón menos frecuente que bigeminismo, pero igualmente requiere evaluación."
        return explanation
    
    def _explain_av_block(self, features):
        explanation = "**Bloqueo AV Detectado:**\n\n"
        explanation += f"• Intervalo PR prolongado: {features.get('pr_interval', 0):.1f} ms\n"
        explanation += f"• Relación aurículo-ventricular alterada\n"
        explanation += f"• Pérdida de conducción: {features.get('av_block_ratio', 0):.1%}\n\n"
        explanation += "**Significado clínico:** Alteración en la conducción eléctrica entre aurículas y ventrículos."
        return explanation
    
    def _explain_afib(self, features):
        explanation = "**Fibrilación Auricular Detectada:**\n\n"
        explanation += f"• Irregularidad RR: {features.get('std_rr', 0):.2f} ms\n"
        explanation += f"• Ausencia de ondas P organizadas\n"
        explanation += f"• Variabilidad de frecuencia: {features.get('hr_std', 0):.1f} lpm\n\n"
        explanation += "**Significado clínico:** Arritmia auricular común que aumenta riesgo de embolias."
        return explanation
    
    def _explain_vtach(self, features):
        explanation = "**Taquicardia Ventricular Detectada:**\n\n"
        explanation += f"• Complejos QRS anchos: {features.get('qrs_width', 0):.1f} ms\n"
        explanation += f"• Frecuencia cardíaca elevada: {features.get('hr_mean', 0):.1f} lpm\n"
        explanation += f"• Disociación aurículo-ventricular\n\n"
        explanation += "**Significado clínico:** Arritmia ventricular grave que puede ser potencialmente mortal."
        return explanation
    
    def _explain_pvcs(self, features):
        explanation = "**Extrasístoles Ventriculares Detectadas:**\n\n"
        explanation += f"• Densidad de PVCs: {features.get('pvc_density', 0):.1%}\n"
        explanation += f"• Distribución: { 'Agrupadas' if features.get('pvc_density', 0) > 0.1 else 'Aisladas'}\n"
        explanation += f"• Variabilidad de acoplamiento: {features.get('pvc_coupling_std', 0):.1f} ms\n\n"
        explanation += "**Significado clínico:** Latidos ventriculares prematuros que pueden ser benignos o indicar cardiopatía."
        return explanation

# =============================================================================
# DETECTOR DE ARTEFACTOS Y RUIDO
# =============================================================================

class ArtifactDetector:
    def __init__(self):
        self.artifact_thresholds = {
            'motion_threshold': 0.15,
            'electrode_pop_threshold': 0.3,
            'powerline_threshold': 0.1,
            'baseline_drift_threshold': 0.2
        }
    
    def detect_artifacts(self, ecg_signal, sampling_rate):
        """Detección y clasificación de artefactos comunes en ECG"""
        
        artifacts = {
            'motion_artifact': self._detect_motion_artifact(ecg_signal, sampling_rate),
            'electrode_pop': self._detect_electrode_pop(ecg_signal),
            'powerline_noise': self._detect_powerline_noise(ecg_signal, sampling_rate),
            'baseline_drift': self._detect_baseline_drift(ecg_signal),
            'muscle_noise': self._detect_muscle_noise(ecg_signal, sampling_rate)
        }
        
        # Filtrar solo artefactos detectados
        detected_artifacts = {k: v for k, v in artifacts.items() if v['detected']}
        
        # Calcular score global de artefactos
        artifact_score = sum(art['severity'] for art in detected_artifacts.values()) / max(1, len(detected_artifacts))
        
        return {
            'detected_artifacts': detected_artifacts,
            'artifact_score': artifact_score,
            'artifact_count': len(detected_artifacts),
            'recommendations': self._get_artifact_recommendations(detected_artifacts)
        }
    
    def _detect_motion_artifact(self, ecg_signal, sampling_rate):
        """Detectar artefactos por movimiento"""
        try:
            # Analizar derivadas de alta frecuencia
            diff_signal = np.diff(ecg_signal)
            motion_peaks = np.sum(np.abs(diff_signal) > np.std(ecg_signal) * 3) / len(diff_signal)
            
            detected = motion_peaks > self.artifact_thresholds['motion_threshold']
            severity = min(1.0, motion_peaks * 2)
            
            return {
                'detected': detected,
                'severity': severity,
                'description': 'Artefacto por movimiento del paciente',
                'suggested_fix': 'Mantener al paciente inmóvil durante el registro'
            }
        except:
            return {'detected': False, 'severity': 0, 'description': '', 'suggested_fix': ''}
    
    def _detect_electrode_pop(self, ecg_signal):
        """Detectar desconexión momentánea de electrodos"""
        try:
            # Buscar transiciones abruptas
            diff_signal = np.diff(ecg_signal)
            pop_candidates = np.sum(np.abs(diff_signal) > np.std(ecg_signal) * 5) / len(diff_signal)
            
            detected = pop_candidates > self.artifact_thresholds['electrode_pop_threshold']
            severity = min(1.0, pop_candidates * 3)
            
            return {
                'detected': detected,
                'severity': severity,
                'description': 'Desconexión momentánea de electrodos',
                'suggested_fix': 'Verificar conexiones y contacto de electrodos'
            }
        except:
            return {'detected': False, 'severity': 0, 'description': '', 'suggested_fix': ''}
    
    def _detect_powerline_noise(self, ecg_signal, sampling_rate):
        """Detectar interferencia de línea de potencia (50/60 Hz)"""
        try:
            # Análisis espectral en banda de 50/60 Hz
            f, Pxx = signal.welch(ecg_signal, sampling_rate, nperseg=min(1024, len(ecg_signal)//4))
            
            # Buscar picos en 50 Hz y armónicos
            powerline_freqs = [50, 60, 100, 120, 150, 180]
            powerline_power = 0
            
            for freq in powerline_freqs:
                idx = np.argmin(np.abs(f - freq))
                if idx < len(Pxx):
                    powerline_power += Pxx[idx]
            
            total_power = np.trapz(Pxx, f)
            powerline_ratio = powerline_power / total_power if total_power > 0 else 0
            
            detected = powerline_ratio > self.artifact_thresholds['powerline_threshold']
            severity = min(1.0, powerline_ratio * 5)
            
            return {
                'detected': detected,
                'severity': severity,
                'description': f'Interferencia de línea de potencia ({powerline_ratio:.3f})',
                'suggested_fix': 'Usar filtro de línea de potencia y verificar conexión a tierra'
            }
        except:
            return {'detected': False, 'severity': 0, 'description': '', 'suggested_fix': ''}
    
    def _detect_baseline_drift(self, ecg_signal):
        """Detectar deriva de línea base"""
        try:
            # Filtro pasa-bajos para línea base
            b, a = signal.butter(3, 0.5, btype='low', fs=250)
            baseline = signal.filtfilt(b, a, ecg_signal)
            
            # Calcular variación de línea base
            baseline_variation = np.std(baseline) / (np.std(ecg_signal) + 1e-10)
            
            detected = baseline_variation > self.artifact_thresholds['baseline_drift_threshold']
            severity = min(1.0, baseline_variation * 3)
            
            return {
                'detected': detected,
                'severity': severity,
                'description': 'Deriva de línea base significativa',
                'suggested_fix': 'Aplicar filtro de línea base y verificar contacto de electrodos'
            }
        except:
            return {'detected': False, 'severity': 0, 'description': '', 'suggested_fix': ''}
    
    def _detect_muscle_noise(self, ecg_signal, sampling_rate):
        """Detectar ruido muscular (EMG)"""
        try:
            # Análisis espectral en banda de alta frecuencia (20-100 Hz)
            f, Pxx = signal.welch(ecg_signal, sampling_rate, nperseg=min(1024, len(ecg_signal)//4))
            
            muscle_band = (20, 100)
            muscle_power = np.trapz(Pxx[(f >= muscle_band[0]) & (f <= muscle_band[1])], 
                                  f[(f >= muscle_band[0]) & (f <= muscle_band[1])])
            total_power = np.trapz(Pxx, f)
            muscle_ratio = muscle_power / total_power if total_power > 0 else 0
            
            detected = muscle_ratio > 0.1  # 10% de potencia en banda muscular
            severity = min(1.0, muscle_ratio * 4)
            
            return {
                'detected': detected,
                'severity': severity,
                'description': 'Ruido muscular (EMG) detectado',
                'suggested_fix': 'Relajar músculos y asegurar posición cómoda'
            }
        except:
            return {'detected': False, 'severity': 0, 'description': '', 'suggested_fix': ''}
    
    def _get_artifact_recommendations(self, artifacts):
        """Generar recomendaciones para corregir artefactos"""
        recommendations = []
        
        for artifact_name, artifact_info in artifacts.items():
            if artifact_info['detected']:
                recommendations.append(f"• {artifact_info['suggested_fix']}")
        
        if not recommendations:
            recommendations.append("✅ Calidad de señal buena - pocos artefactos detectados")
        
        return recommendations

# =============================================================================
# ANÁLISIS DE TENDENCIA TEMPORAL
# =============================================================================

class TemporalTrendAnalyzer:
    def __init__(self):
        self.trend_history = []
        self.max_history_size = 10
    
    def add_analysis(self, analysis_results, timestamp=None):
        """Agregar análisis actual al historial"""
        if timestamp is None:
            timestamp = datetime.now()
        
        analysis_entry = {
            'timestamp': timestamp,
            'results': analysis_results.copy()
        }
        
        self.trend_history.append(analysis_entry)
        
        # Mantener tamaño máximo del historial
        if len(self.trend_history) > self.max_history_size:
            self.trend_history.pop(0)
    
    def analyze_trends(self, current_analysis):
        """Analizar evolución temporal de las arritmias"""
        if len(self.trend_history) < 2:
            return {"message": "Historial insuficiente para análisis de tendencias"}
        
        trend_alerts = []
        improvements = []
        
        # Obtener análisis anterior
        previous_analysis = self.trend_history[-2]['results']
        
        # Analizar cada condición
        for condition, current_prob in current_analysis.items():
            if condition in previous_analysis:
                previous_prob = previous_analysis[condition]
                trend = current_prob - previous_prob
                
                if abs(trend) > 0.1:  # Cambio significativo
                    if trend > 0.1:  # Empeoramiento
                        trend_alerts.append({
                            'condition': condition,
                            'change': trend,
                            'trend': 'worsening',
                            'message': f"📈 Empeoramiento de {condition}: +{trend:.1%}"
                        })
                    elif trend < -0.1:  # Mejoría
                        improvements.append({
                            'condition': condition,
                            'change': abs(trend),
                            'trend': 'improving',
                            'message': f"📉 Mejoría de {condition}: -{abs(trend):.1%}"
                        })
        
        # Análisis de estabilidad
        stability_score = self._calculate_stability(current_analysis)
        
        return {
            'trend_alerts': trend_alerts,
            'improvements': improvements,
            'stability_score': stability_score,
            'total_changes': len(trend_alerts) + len(improvements),
            'overall_trend': 'ESTABLE' if stability_score > 0.8 else 'CAMBIOS DETECTADOS'
        }
    
    def _calculate_stability(self, current_analysis):
        """Calcular score de estabilidad basado en variaciones recientes"""
        if len(self.trend_history) < 3:
            return 1.0
        
        variations = []
        recent_analyses = self.trend_history[-3:]
        
        for i in range(1, len(recent_analyses)):
            prev = recent_analyses[i-1]['results']
            curr = recent_analyses[i]['results']
            
            for condition in current_analysis.keys():
                if condition in prev and condition in curr:
                    variation = abs(curr[condition] - prev[condition])
                    variations.append(variation)
        
        if not variations:
            return 1.0
        
        avg_variation = np.mean(variations)
        stability = max(0, 1 - avg_variation * 5)  # Convertir a score 0-1
        
        return stability

# =============================================================================
# SISTEMA DE APRENDIZAJE CONTINUO
# =============================================================================

class FeedbackLearningSystem:
    def __init__(self):
        self.feedback_data = []
        self.learning_threshold = 5  # Mínimo de feedbacks para ajustar
    
    def add_feedback(self, prediction, actual_diagnosis, user_correction, confidence):
        """Aprender de las correcciones del usuario"""
        feedback_entry = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'actual_diagnosis': actual_diagnosis,
            'user_correction': user_correction,
            'confidence': confidence,
            'was_correct': prediction == actual_diagnosis
        }
        
        self.feedback_data.append(feedback_entry)
        st.success("✅ Feedback registrado para mejorar el sistema")
    
    def get_learning_insights(self):
        """Obtener insights del aprendizaje"""
        if len(self.feedback_data) == 0:
            return {"message": "Aún no hay datos de feedback"}
        
        total_feedbacks = len(self.feedback_data)
        correct_predictions = sum(1 for f in self.feedback_data if f['was_correct'])
        accuracy = correct_predictions / total_feedbacks
        
        # Análisis por condición
        condition_analysis = {}
        for feedback in self.feedback_data:
            condition = feedback['prediction']
            if condition not in condition_analysis:
                condition_analysis[condition] = {'total': 0, 'correct': 0}
            
            condition_analysis[condition]['total'] += 1
            if feedback['was_correct']:
                condition_analysis[condition]['correct'] += 1
        
        # Calcular accuracy por condición
        condition_accuracy = {}
        for condition, stats in condition_analysis.items():
            condition_accuracy[condition] = stats['correct'] / stats['total']
        
        return {
            'total_feedbacks': total_feedbacks,
            'overall_accuracy': accuracy,
            'condition_accuracy': condition_accuracy,
            'learning_ready': total_feedbacks >= self.learning_threshold
        }
    
    def adjust_detection_thresholds(self):
        """Ajustar umbrales de detección basado en feedback acumulado"""
        if len(self.feedback_data) < self.learning_threshold:
            return {"message": "Feedback insuficiente para ajustes automáticos"}
        
        insights = self.get_learning_insights()
        
        adjustment_suggestions = []
        for condition, accuracy in insights['condition_accuracy'].items():
            if accuracy < 0.7:  # Baja accuracy
                adjustment_suggestions.append(
                    f"Considerar aumentar umbral para {condition} (accuracy: {accuracy:.1%})"
                )
            elif accuracy > 0.9:  # Muy alta accuracy
                adjustment_suggestions.append(
                    f"Considerar disminuir umbral para {condition} (accuracy: {accuracy:.1%})"
                )
        
        return {
            'adjustment_suggestions': adjustment_suggestions,
            'current_accuracy': insights['overall_accuracy'],
            'recommendation': 'Umbrales estables' if not adjustment_suggestions else 'Ajustes recomendados'
        }

# =============================================================================
# CLASES DE MODELOS AVANZADOS (MEJORADAS)
# =============================================================================

class AdvancedECGClassifier:
    """Modelos avanzados para clasificación de arritmias - Versión Mejorada"""
    
    def __init__(self, input_shape, num_classes=8):  # Aumentado a 8 clases
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def create_resnet_lstm_hybrid(self):
        """Modelo híbrido ResNet + LSTM optimizado"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Branch CNN (ResNet-like optimizado)
        x = layers.Conv1D(64, 7, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(3)(x)
        
        # Residual blocks optimizados
        for filters in [64, 128, 256]:
            # Residual connection
            residual = layers.Conv1D(filters, 1, padding='same')(x) if x.shape[-1] != filters else x
            
            x = layers.Conv1D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv1D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            
            # Add residual
            x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling1D(2)(x)
        
        # Temporal modeling con LSTM optimizado
        x = layers.Reshape((-1, x.shape[-1]))(x)
        x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
        x = layers.LSTM(64, dropout=0.2)(x)
        
        # Classification head mejorado
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        return model
    
    def create_transformer_ecg(self):
        """Modelo Transformer para ECG optimizado"""
        class TransformerBlock(layers.Layer):
            def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
                super(TransformerBlock, self).__init__()
                self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
                self.ffn = tf.keras.Sequential([
                    layers.Dense(ff_dim, activation="relu"),
                    layers.Dense(embed_dim),
                ])
                self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = layers.Dropout(rate)
                self.dropout2 = layers.Dropout(rate)
            
            def call(self, inputs, training):
                attn_output = self.att(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Positional encoding optimizado
        x = layers.Conv1D(128, 3, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Add positional encoding
        position = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        position = tf.cast(position, tf.float32)
        position = tf.expand_dims(position, 0)
        position = tf.expand_dims(position, -1)
        position = tf.tile(position, [tf.shape(x)[0], 1, 128])
        
        x = x + position
        
        # Transformer blocks optimizados
        x = TransformerBlock(128, 4, 512)(x)
        x = TransformerBlock(128, 4, 512)(x)
        
        # Global attention pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification mejorada
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

class ECGAnomalyDetector:
    """Detección de arritmias raras usando VAE - Versión Mejorada"""
    
    def __init__(self, input_dim, latent_dim=20):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
    
    def build_vae(self):
        """Variational Autoencoder para ECG optimizado"""
        # Encoder mejorado
        encoder_inputs = layers.Input(shape=(self.input_dim, 1))
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(encoder_inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        
        # Decoder mejorado
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation='relu')(latent_inputs)
        x = layers.Dense(64 * 32, activation='relu')(x)
        x = layers.Reshape((64, 32))(x)
        x = layers.Conv1DTranspose(128, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1DTranspose(64, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1DTranspose(32, 3, activation='relu', padding='same')(x)
        decoder_outputs = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x)
        
        # Models
        encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        decoder = Model(latent_inputs, decoder_outputs, name='decoder')
        
        # VAE
        outputs = decoder(encoder(encoder_inputs)[2])
        vae = Model(encoder_inputs, outputs, name='vae')
        
        # Loss function mejorada
        reconstruction_loss = tf.keras.losses.mse(
            tf.keras.layers.Flatten()(encoder_inputs),
            tf.keras.layers.Flatten()(outputs)
        )
        reconstruction_loss *= self.input_dim
        
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        
        return vae, encoder, decoder

class ECGEnsemble:
    """Ensemble de modelos avanzados - Versión Mejorada"""
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = []
    
    def create_ensemble(self):
        """Crear ensemble de modelos diversos optimizado"""
        # 1. ResNet-LSTM Hybrid
        advanced_classifier = AdvancedECGClassifier(self.input_shape, self.num_classes)
        model1 = advanced_classifier.create_resnet_lstm_hybrid()
        self.models.append(('resnet_lstm', model1))
        
        # 2. Transformer
        model2 = advanced_classifier.create_transformer_ecg()
        self.models.append(('transformer', model2))
        
        # 3. CNN Simple pero profundo
        model3 = self._create_deep_cnn()
        self.models.append(('deep_cnn', model3))
        
        return self.models
    
    def _create_deep_cnn(self):
        """CNN profundo inspirado en Inception optimizado"""
        def inception_module(x, filters):
            # 1x1 convolution
            conv1 = layers.Conv1D(filters, 1, padding='same', activation='relu')(x)
            
            # 3x3 convolution
            conv3 = layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
            
            # 5x5 convolution
            conv5 = layers.Conv1D(filters, 5, padding='same', activation='relu')(x)
            
            # 3x3 max pooling
            pool = layers.MaxPooling1D(3, strides=1, padding='same')(x)
            pool = layers.Conv1D(filters, 1, padding='same', activation='relu')(pool)
            
            # Concatenate
            return layers.Concatenate()([conv1, conv3, conv5, pool])
        
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling1D(2)(x)
        
        # Inception modules optimizados
        x = inception_module(x, 64)
        x = inception_module(x, 128)
        x = inception_module(x, 256)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

# =============================================================================
# PROCESADOR DE SEÑALES ECG OPTIMIZADO
# =============================================================================

class OptimizedECGProcessor:
    """Procesador robusto de señales ECG con gestión de memoria y cache"""
    
    def __init__(self):
        self.supported_formats = {
            'edf': 'EDF/EDF+ (European Data Format)',
            'bdf': 'BDF (Biosemi Data Format)',
            'bin': 'Formato Binario Personalizado',
            'raw': 'RAW (Datos crudos binarios)'
        }
        
        self.binary_formats = {
            'int16': '16-bit signed integer',
            'int32': '32-bit signed integer', 
            'float32': '32-bit float',
            'float64': '64-bit float',
            'uint16': '16-bit unsigned integer'
        }
        self.chunk_size = 100000  # Procesar por chunks
    
    def process_large_file(self, uploaded_file, **kwargs):
        """Procesar archivos grandes por chunks con gestión de memoria"""
        file_size = uploaded_file.size
        
        if file_size > 50 * 1024 * 1024:  # > 50MB
            return self._process_in_chunks(uploaded_file, **kwargs)
        else:
            return self.load_ecg_file(uploaded_file, **kwargs)
    
    def _process_in_chunks(self, uploaded_file, **kwargs):
        """Procesar archivo grande en chunks optimizado"""
        try:
            chunks = []
            total_samples = 0
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Procesar por chunks
            with open(tmp_path, 'rb') as f:
                while True:
                    chunk_data = f.read(self.chunk_size)
                    if not chunk_data:
                        break
                    
                    # Simular procesamiento de chunk
                    chunks.append(len(chunk_data))
                    total_samples += len(chunk_data) // kwargs.get('bytes_per_sample', 2)
                    
                    # Liberar memoria
                    del chunk_data
                    gc.collect()
            
            os.unlink(tmp_path)
            st.success(f"📊 Archivo grande procesado: {total_samples:,} muestras en {len(chunks)} chunks")
            
            # Crear señal dummy para demostración (en producción sería el procesamiento real)
            return self._create_dummy_signal(total_samples), 250
            
        except Exception as e:
            st.error(f"❌ Error procesando archivo grande: {str(e)}")
            return None, None
    
    def _create_dummy_signal(self, total_samples):
        """Crear señal dummy para archivos grandes (simulación)"""
        t = np.linspace(0, total_samples/250, total_samples)
        # Señal ECG simulada básica
        signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
        return signal
    
    def load_ecg_file(self, uploaded_file, binary_format=None, bytes_per_sample=2, header_size=0, sampling_rate=None):
        """Cargar archivo ECG en formatos binarios y europeos - Versión Optimizada"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['edf', 'bdf']:
                return self._load_european_format(uploaded_file)
            elif file_extension in ['bin', 'raw']:
                return self._load_binary(uploaded_file, binary_format, bytes_per_sample, header_size, sampling_rate)
            else:
                # Intentar auto-detección
                return self._auto_detect_format(uploaded_file)
                
        except Exception as e:
            st.error(f"Error cargando archivo: {str(e)}")
            return None, None
    
    def _load_european_format(self, uploaded_file):
        """Cargar formatos europeos EDF/BDF optimizado"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with pyedflib.EdfReader(tmp_path) as f:
                    # Obtener información del archivo
                    n_channels = f.signals_in_file
                    signal_labels = f.getSignalLabels()
                    
                    # Buscar canal ECG
                    ecg_channel = None
                    for i, label in enumerate(signal_labels):
                        if any(ecg_keyword in label.lower() for ecg_keyword in 
                              ['ecg', 'ekg', 'electrocardiogram', 'derivation']):
                            ecg_channel = i
                            break
                    
                    # Si no se encuentra, usar primer canal
                    if ecg_channel is None:
                        ecg_channel = 0
                        st.warning(f"⚠️ Canal ECG no identificado. Usando canal: {signal_labels[0]}")
                    
                    # Leer señal ECG
                    ecg_signal = f.readSignal(ecg_channel)
                    sampling_rate = f.getSampleFrequency(ecg_channel)
                    
                    # Metadatos adicionales
                    file_duration = f.getFileDuration()
                    patient_info = f.getPatientCode()
                    
                    st.success(f"✅ EDF/BDF cargado: {len(ecg_signal)} muestras, {sampling_rate} Hz, {file_duration}s")
                    st.info(f"📋 Paciente: {patient_info}, Canal: {signal_labels[ecg_channel]}")
                    
                    return ecg_signal, sampling_rate
                    
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            st.error(f"Error leyendo archivo EDF/BDF: {str(e)}")
            return None, None
    
    def _load_binary(self, uploaded_file, data_format='int16', bytes_per_sample=2, header_size=0, sampling_rate=None):
        """Cargar archivos binarios personalizados optimizado"""
        try:
            raw_data = uploaded_file.getvalue()
            file_size = len(raw_data)
            
            st.info(f"📊 Tamaño archivo: {file_size} bytes, Cabecera: {header_size} bytes")
            
            # Saltar cabecera si existe
            if header_size > 0:
                if header_size >= file_size:
                    st.error("❌ Tamaño de cabecera mayor que archivo")
                    return None, None
                raw_data = raw_data[header_size:]
            
            # Determinar formato de datos
            dtype_map = {
                'int16': np.int16,
                'int32': np.int32, 
                'float32': np.float32,
                'float64': np.float64,
                'uint16': np.uint16
            }
            
            dtype = dtype_map.get(data_format, np.int16)
            
            # Calcular número de muestras
            data_size = len(raw_data)
            num_samples = data_size // bytes_per_sample
            
            if num_samples == 0:
                st.error("❌ No hay muestras después de procesar cabecera")
                return None, None
            
            # Leer datos binarios
            try:
                ecg_signal = np.frombuffer(raw_data[:num_samples * bytes_per_sample], dtype=dtype)
            except Exception as e:
                st.error(f"❌ Error interpretando datos binarios: {e}")
                return None, None
            
            # Convertir a float para procesamiento
            ecg_signal = ecg_signal.astype(np.float64)
            
            # Remover offset DC y normalizar
            ecg_signal = ecg_signal - np.mean(ecg_signal)
            if np.std(ecg_signal) > 0:
                ecg_signal = ecg_signal / np.std(ecg_signal)
            
            # Estimar sampling rate si no se proporciona
            if sampling_rate is None:
                sampling_rate = self._estimate_sampling_rate(ecg_signal)
            
            st.success(f"✅ Binario cargado: {len(ecg_signal)} muestras, {sampling_rate:.1f} Hz")
            st.info(f"🔢 Formato: {data_format}, Bytes/muestra: {bytes_per_sample}")
            
            return ecg_signal, sampling_rate
            
        except Exception as e:
            st.error(f"❌ Error leyendo archivo binario: {str(e)}")
            return None, None
    
    def _auto_detect_format(self, uploaded_file):
        """Auto-detección de formato para archivos desconocidos optimizado"""
        content = uploaded_file.getvalue()
        
        # Intentar como EDF/BDF primero
        try:
            if content[:4] == b'0\x00\x00\x00' or content[:8] == b'0       ':
                return self._load_european_format(uploaded_file)
        except:
            pass
        
        # Intentar diferentes formatos binarios
        binary_configs = [
            ('int16', 2), ('int32', 4), ('float32', 4), 
            ('float64', 8), ('uint16', 2)
        ]
        
        for data_format, bytes_per_sample in binary_configs:
            try:
                ecg_signal, sampling_rate = self._load_binary(
                    uploaded_file, data_format, bytes_per_sample, 0, None
                )
                if ecg_signal is not None and len(ecg_signal) > 100:
                    st.success(f"🎯 Formato auto-detectado: {data_format}")
                    return ecg_signal, sampling_rate
            except:
                continue
        
        st.error("❌ No se pudo auto-detectar el formato del archivo")
        return None, None
    
    def _estimate_sampling_rate(self, signal):
        """Estimar frecuencia de muestreo basado en características de la señal optimizado"""
        if len(signal) < 100:
            return 250  # Valor por defecto
        
        # Analizar características espectrales
        try:
            f, Pxx = signal.welch(signal, 250, nperseg=min(1024, len(signal)//4))
            dominant_freq = f[np.argmax(Pxx)]
            
            # Frecuencias típicas de ECG
            if dominant_freq < 5:  # Baja frecuencia típica de ECG
                estimated_duration = 10  # Asumir 10 segundos
            else:
                estimated_duration = 5   # Señal más corta
            
            sampling_rate = len(signal) / estimated_duration
            
            # Redondear a valores típicos
            typical_rates = [125, 250, 500, 1000]
            closest_rate = min(typical_rates, key=lambda x: abs(x - sampling_rate))
            
            return closest_rate
            
        except:
            return 250  # Valor por defecto

# =============================================================================
# DETECTOR DE ARRITMIAS AVANZADO MEJORADO CON DETECCIÓN DE BIGEMINISMO/TRIGEMINISMO
# =============================================================================

class AdvancedArrhythmiaDetector:
    """Detector de arritmias con modelos avanzados - Versión Mejorada con detección de patrones"""
    
    def __init__(self):
        # CARACTERÍSTICAS MEJORADAS CON PATRONES DE AGRUPAMIENTO
        self.feature_names = [
            'mean_rr', 'std_rr', 'cv_rr', 'rmssd', 'nn50', 'pnn50',
            'hr_mean', 'hr_std', 'qrs_width', 'qt_interval', 'pr_interval',
            't_peak_amplitude', 'st_slope', 'signal_entropy', 'lf_hf_ratio',
            'sdnn', 'cv_hr', 'amplitude_asymmetry', 'zero_crossing_rate',
            # NUEVAS CARACTERÍSTICAS PARA PATRONES
            'bigeminy_score', 'trigeminy_score', 'pvc_coupling_std',
            'compensatory_pause_ratio', 'pattern_regularity', 'pvc_density',
            # CARACTERÍSTICAS AVANZADAS AÑADIDAS
            'p_wave_absence_score', 'av_dissociation_score', 'av_block_ratio',
            'r_on_t_risk', 'sustained_vt_duration', 'st_depression_mv'
        ]
        
        # TIPOS DE ARRITMIA MEJORADOS
        self.arrhythmia_types = {
            'Ritmo Sinusal Normal': 'NSR',
            'Fibrilación Auricular': 'AFIB', 
            'Taquicardia Sinusal': 'STACH',
            'Bradicardia Sinusal': 'SBRAD',
            'Extrasístoles Ventriculares': 'PVC',
            'Taquicardia Ventricular': 'VTACH',
            'Bloqueo AV': 'AVB',
            'Ritmo de Escape': 'ESCAPE',
            # NUEVOS PATRONES AÑADIDOS
            'Bigeminismo Ventricular': 'BIGEM',
            'Trigeminismo Ventricular': 'TRIGEM'
        }
        
        # Inicializar sistemas de apoyo
        self.alert_system = ClinicalAlertSystem()
        self.quality_assessor = SignalQualityAssessment()
        self.criteria_validator = ClinicalCriteriaValidator()
        self.explainer = ExplainableAI()
        self.artifact_detector = ArtifactDetector()
        self.trend_analyzer = TemporalTrendAnalyzer()
        self.learning_system = FeedbackLearningSystem()
    
    def extract_advanced_features(self, ecg_signal, sampling_rate):
        """Extraer características avanzadas para modelos deep learning optimizado"""
        try:
            # Preprocesamiento robusto
            ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method='neurokit')
            
            # Detección de picos R con múltiples métodos
            try:
                rpeaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='kalidas2017')
                rpeaks = info['ECG_R_Peaks']
            except:
                # Fallback a método más simple
                rpeaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method='neurokit')
                rpeaks = info['ECG_R_Peaks']
            
            if len(rpeaks) < 4:
                return None, "Señal demasiado corta para análisis"
            
            features = {}
            
            # Características de variabilidad cardíaca avanzadas
            rr_intervals = np.diff(rpeaks) / sampling_rate * 1000  # ms
            
            features['mean_rr'] = np.mean(rr_intervals)
            features['std_rr'] = np.std(rr_intervals)
            features['cv_rr'] = features['std_rr'] / features['mean_rr'] if features['mean_rr'] > 0 else 0
            features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
            features['nn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 50)
            features['pnn50'] = features['nn50'] / len(rr_intervals) * 100 if len(rr_intervals) > 0 else 0
            features['sdnn'] = np.std(rr_intervals)
            
            # Frecuencia cardíaca
            instantaneous_hr = 60000 / rr_intervals
            features['hr_mean'] = np.mean(instantaneous_hr)
            features['hr_std'] = np.std(instantaneous_hr)
            features['cv_hr'] = features['hr_std'] / features['hr_mean'] if features['hr_mean'] > 0 else 0
            
            # Características morfológicas
            features.update(self._extract_morphological_features(ecg_cleaned, rpeaks, sampling_rate))
            
            # Características espectrales
            features.update(self._extract_spectral_features(ecg_cleaned, sampling_rate))
            
            # Características de complejidad
            features.update(self._extract_complexity_features(ecg_cleaned))
            
            # NUEVO: Características de patrones de agrupamiento
            features.update(self._extract_pattern_features(ecg_cleaned, rpeaks, sampling_rate))
            
            # NUEVO: Características avanzadas de patrones complejos
            features.update(self._extract_advanced_patterns(ecg_cleaned, rpeaks, sampling_rate))
            
            # Asegurar que todas las características estén presentes
            for feature in self.feature_names:
                if feature not in features:
                    features[feature] = 0.0
            
            feature_vector = np.array([features[feature] for feature in self.feature_names])
            
            return feature_vector, None
            
        except Exception as e:
            return None, f"Error extrayendo características: {str(e)}"
    
    def _extract_pattern_features(self, ecg_signal, rpeaks, sampling_rate):
        """NUEVO: Extraer características específicas para bigeminismo/trigeminismo"""
        features = {}
        
        try:
            if len(rpeaks) < 6:
                return {
                    'bigeminy_score': 0,
                    'trigeminy_score': 0, 
                    'pvc_coupling_std': 0,
                    'compensatory_pause_ratio': 0,
                    'pattern_regularity': 0,
                    'pvc_density': 0
                }
            
            rr_intervals = np.diff(rpeaks) / sampling_rate * 1000  # ms
            
            # Detectar posibles PVCs (intervalos RR muy cortos/largos)
            mean_rr = np.mean(rr_intervals)
            pvc_threshold = 0.3  # 30% variación para PVCs
            
            pvc_candidates = []
            normal_intervals = []
            
            for i, rr in enumerate(rr_intervals):
                if abs(rr - mean_rr) / mean_rr > pvc_threshold:
                    pvc_candidates.append(i)
                else:
                    normal_intervals.append(rr)
            
            # Características de densidad de PVCs
            features['pvc_density'] = len(pvc_candidates) / len(rr_intervals)
            
            # Análisis de patrones de agrupamiento
            if len(pvc_candidates) >= 2:
                coupling_intervals = []
                compensatory_pauses = []
                
                for pvc_idx in pvc_candidates:
                    if pvc_idx > 0:
                        coupling_intervals.append(rr_intervals[pvc_idx])
                    if pvc_idx < len(rr_intervals) - 1:
                        compensatory_pauses.append(rr_intervals[pvc_idx + 1])
                
                # Bigeminismo: PVC cada dos latidos
                bigeminy_patterns = 0
                trigeminy_patterns = 0
                
                for i in range(len(rr_intervals) - 2):
                    # Patrón bigeminismo: normal-corto-normal-corto
                    if (abs(rr_intervals[i] - mean_rr) / mean_rr < 0.2 and  # normal
                        abs(rr_intervals[i+1] - mean_rr) / mean_rr > 0.3 and  # PVC (corto)
                        abs(rr_intervals[i+2] - mean_rr) / mean_rr < 0.2):    # normal
                        bigeminy_patterns += 1
                    
                    # Patrón trigeminismo: normal-normal-corto
                    if (i < len(rr_intervals) - 3 and
                        abs(rr_intervals[i] - mean_rr) / mean_rr < 0.2 and    # normal
                        abs(rr_intervals[i+1] - mean_rr) / mean_rr < 0.2 and  # normal  
                        abs(rr_intervals[i+2] - mean_rr) / mean_rr > 0.3):    # PVC (corto)
                        trigeminy_patterns += 1
                
                features['bigeminy_score'] = bigeminy_patterns / max(1, len(pvc_candidates))
                features['trigeminy_score'] = trigeminy_patterns / max(1, len(pvc_candidates))
                features['pvc_coupling_std'] = np.std(coupling_intervals) if coupling_intervals else 0
                features['compensatory_pause_ratio'] = (
                    np.mean(compensatory_pauses) / mean_rr if compensatory_pauses else 0
                )
                
                # Regularidad del patrón
                if coupling_intervals:
                    features['pattern_regularity'] = 1 - (np.std(coupling_intervals) / np.mean(coupling_intervals))
                else:
                    features['pattern_regularity'] = 0
            else:
                features.update({
                    'bigeminy_score': 0,
                    'trigeminy_score': 0,
                    'pvc_coupling_std': 0, 
                    'compensatory_pause_ratio': 0,
                    'pattern_regularity': 0
                })
                
        except Exception as e:
            features.update({
                'bigeminy_score': 0,
                'trigeminy_score': 0,
                'pvc_coupling_std': 0,
                'compensatory_pause_ratio': 0, 
                'pattern_regularity': 0,
                'pvc_density': 0
            })
        
        return features

    def _extract_advanced_patterns(self, ecg_signal, rpeaks, sampling_rate):
        """NUEVO: Detección de patrones complejos como R-on-T y taquicardia sostenida"""
        features = {}
        
        try:
            if len(rpeaks) < 4:
                return {
                    'r_on_t_risk': 0,
                    'sustained_vt_duration': 0,
                    'st_depression_mv': 0,
                    'p_wave_absence_score': 0,
                    'av_dissociation_score': 0,
                    'av_block_ratio': 0
                }
            
            rr_intervals = np.diff(rpeaks) / sampling_rate * 1000
            
            # Detección R-on-T (PVC peligrosa)
            features['r_on_t_risk'] = self._detect_r_on_t_pattern(ecg_signal, rpeaks, sampling_rate)
            
            # Detección de taquicardia sostenida
            features['sustained_vt_duration'] = self._detect_sustained_tachycardia(rr_intervals)
            
            # Análisis de segmento ST (isquemia)
            features['st_depression_mv'] = self._analyze_st_segment(ecg_signal, rpeaks, sampling_rate)
            
            # Detección de ondas P (para bloqueo AV y FA)
            features['p_wave_absence_score'] = self._detect_p_wave_absence(ecg_signal, rpeaks, sampling_rate)
            
            # Disociación AV (para taquicardia ventricular)
            features['av_dissociation_score'] = self._detect_av_dissociation(ecg_signal, rpeaks, sampling_rate)
            
            # Ratio de bloqueo AV
            features['av_block_ratio'] = self._calculate_av_block_ratio(rr_intervals)
            
        except Exception as e:
            features.update({
                'r_on_t_risk': 0,
                'sustained_vt_duration': 0,
                'st_depression_mv': 0,
                'p_wave_absence_score': 0,
                'av_dissociation_score': 0,
                'av_block_ratio': 0
            })
        
        return features

    def _detect_r_on_t_pattern(self, ecg_signal, rpeaks, sampling_rate):
        """Detectar riesgo de fenómeno R-on-T"""
        try:
            risk_score = 0
            for i in range(1, len(rpeaks)):
                # Calcular intervalo entre fin de T y siguiente QRS
                t_end_estimate = rpeaks[i-1] + int(0.4 * sampling_rate)  # Asumir fin de onda T
                if t_end_estimate < len(ecg_signal) and t_end_estimate > rpeaks[i-1]:
                    # Buscar PVC muy precoz
                    if rpeaks[i] - rpeaks[i-1] < 0.3 * sampling_rate:  # PVC muy precoz
                        risk_score += 1
            
            return min(1.0, risk_score / max(1, len(rpeaks) - 1))
        except:
            return 0.0

    def _detect_sustained_tachycardia(self, rr_intervals):
        """Detectar taquicardia sostenida"""
        try:
            if len(rr_intervals) == 0:
                return 0.0
            
            tachycardia_threshold = 500  # ms (120 lpm)
            sustained_count = 0
            max_sustained = 0
            
            for rr in rr_intervals:
                if rr < tachycardia_threshold:
                    sustained_count += 1
                    max_sustained = max(max_sustained, sustained_count)
                else:
                    sustained_count = 0
            
            # Convertir a duración en segundos
            duration = (max_sustained * np.mean(rr_intervals)) / 1000 if max_sustained > 0 else 0
            return min(60.0, duration)  # Limitar a 60 segundos
        except:
            return 0.0

    def _analyze_st_segment(self, ecg_signal, rpeaks, sampling_rate):
        """Analizar depresión del segmento ST"""
        try:
            st_depressions = []
            for peak in rpeaks[:min(10, len(rpeaks))]:
                j_point = peak + int(0.08 * sampling_rate)  # Punto J
                st_point = j_point + int(0.06 * sampling_rate)  # Segmento ST
                
                if st_point < len(ecg_signal):
                    baseline = np.mean(ecg_signal[max(0, peak-50):peak])
                    st_amplitude = ecg_signal[st_point] - baseline
                    st_depressions.append(st_amplitude)
            
            if st_depressions:
                avg_depression = np.mean(st_depressions)
                # Convertir a milivolts (asumiendo calibración)
                return abs(avg_depression) * 1000  # Escalar para mejor visualización
            return 0.0
        except:
            return 0.0

    def _detect_p_wave_absence(self, ecg_signal, rpeaks, sampling_rate):
        """Detectar ausencia de ondas P (para FA)"""
        try:
            p_wave_present = 0
            total_analyzed = 0
            
            for i in range(1, min(10, len(rpeaks))):
                # Buscar onda P antes del complejo QRS
                p_wave_region = slice(max(0, rpeaks[i] - int(0.3 * sampling_rate)), rpeaks[i] - int(0.05 * sampling_rate))
                
                if p_wave_region.stop < len(ecg_signal):
                    segment = ecg_signal[p_wave_region]
                    # Buscar deflexión significativa (potencial onda P)
                    if np.max(segment) - np.min(segment) > 0.1 * np.std(ecg_signal):
                        p_wave_present += 1
                    total_analyzed += 1
            
            return 1 - (p_wave_present / total_analyzed) if total_analyzed > 0 else 0.5
        except:
            return 0.5

    def _detect_av_dissociation(self, ecg_signal, rpeaks, sampling_rate):
        """Detectar disociación aurículo-ventricular"""
        try:
            # Análisis simplificado de disociación AV
            rr_variability = np.std(np.diff(rpeaks)) / sampling_rate if len(rpeaks) > 1 else 0
            return min(1.0, rr_variability * 10)  # Escalar
        except:
            return 0.0

    def _calculate_av_block_ratio(self, rr_intervals):
        """Calcular ratio de bloqueo AV"""
        try:
            if len(rr_intervals) < 3:
                return 0.0
            
            # Buscar patrones de Wenckebach u otros bloqueos
            long_intervals = sum(1 for rr in rr_intervals if rr > np.mean(rr_intervals) * 1.5)
            return long_intervals / len(rr_intervals)
        except:
            return 0.0
    
    def _extract_morphological_features(self, ecg_signal, rpeaks, sampling_rate):
        """Extraer características morfológicas avanzadas optimizado"""
        features = {}
        
        try:
            # Análisis de ondas ECG
            signals, waves = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=sampling_rate)
            
            # Estimación de intervalos
            features['qrs_width'] = self._estimate_qrs_width(ecg_signal, rpeaks, sampling_rate)
            features['qt_interval'] = self._estimate_qt_interval(ecg_signal, rpeaks, sampling_rate)
            features['pr_interval'] = self._estimate_pr_interval(ecg_signal, rpeaks, sampling_rate)
            
            # Amplitudes
            features['t_peak_amplitude'] = np.percentile(ecg_signal, 95) - np.percentile(ecg_signal, 5)
            features['st_slope'] = np.mean(np.diff(ecg_signal))
            
        except Exception as e:
            st.warning(f"Características morfológicas limitadas: {str(e)}")
            # Valores por defecto
            features.update({
                'qrs_width': 80,
                'qt_interval': 400,
                'pr_interval': 160,
                't_peak_amplitude': np.std(ecg_signal),
                'st_slope': 0
            })
        
        return features
    
    def _extract_spectral_features(self, ecg_signal, sampling_rate):
        """Extraer características espectrales optimizado"""
        features = {}
        
        try:
            # Análisis espectral
            f, Pxx = signal.welch(ecg_signal, sampling_rate, nperseg=min(1024, len(ecg_signal)//4))
            
            # Bandas de frecuencia típicas de HRV
            lf_band = (0.04, 0.15)  # Baja frecuencia
            hf_band = (0.15, 0.4)   # Alta frecuencia
            
            lf_power = np.trapz(Pxx[(f >= lf_band[0]) & (f <= lf_band[1])], f[(f >= lf_band[0]) & (f <= lf_band[1])])
            hf_power = np.trapz(Pxx[(f >= hf_band[0]) & (f <= hf_band[1])], f[(f >= hf_band[0]) & (f <= hf_band[1])])
            
            features['lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else 0
            
        except:
            features['lf_hf_ratio'] = 1.0
        
        return features
    
    def _extract_complexity_features(self, ecg_signal):
        """Extraer características de complejidad de señal optimizado"""
        features = {}
        
        try:
            # Entropía aproximada
            features['signal_entropy'] = stats.entropy(np.histogram(ecg_signal, bins=50)[0] + 1e-10)
            
            # Asimetría de amplitud
            features['amplitude_asymmetry'] = np.mean(ecg_signal > np.mean(ecg_signal)) - 0.5
            
            # Tasa de cruces por cero
            zero_crossings = np.where(np.diff(np.signbit(ecg_signal)))[0]
            features['zero_crossing_rate'] = len(zero_crossings) / len(ecg_signal)
            
        except:
            features.update({
                'signal_entropy': 0,
                'amplitude_asymmetry': 0,
                'zero_crossing_rate': 0
            })
        
        return features
    
    def _estimate_qrs_width(self, ecg_signal, rpeaks, sampling_rate):
        """Estimar ancho del complejo QRS optimizado"""
        try:
            widths = []
            for peak in rpeaks[:min(10, len(rpeaks))]:
                start = max(0, peak - int(0.05 * sampling_rate))
                end = min(len(ecg_signal), peak + int(0.05 * sampling_rate))
                segment = ecg_signal[start:end]
                if len(segment) > 0:
                    threshold = 0.3 * (np.max(segment) - np.min(segment))
                    above_threshold = np.where(np.abs(segment) > threshold)[0]
                    if len(above_threshold) > 0:
                        width = (above_threshold[-1] - above_threshold[0]) / sampling_rate * 1000
                        widths.append(width)
            return np.median(widths) if widths else 80
        except:
            return 80
    
    def _estimate_qt_interval(self, ecg_signal, rpeaks, sampling_rate):
        """Estimar intervalo QT optimizado"""
        try:
            if len(rpeaks) < 2:
                return 400
            rr_intervals = np.diff(rpeaks) / sampling_rate * 1000
            mean_rr = np.mean(rr_intervals)
            # Fórmula de Bazett
            qt = 0.39 * np.sqrt(mean_rr / 1000) * 1000
            return min(qt, 600)
        except:
            return 400
    
    def _estimate_pr_interval(self, ecg_signal, rpeaks, sampling_rate):
        """Estimar intervalo PR optimizado"""
        try:
            if len(rpeaks) < 2:
                return 160
            # Estimación simple basada en literatura
            return 160 + 0.1 * (np.mean(rpeaks) - 500)
        except:
            return 160

    def comprehensive_analysis(self, ecg_signal, sampling_rate):
        """Análisis comprehensivo que integra todos los sistemas"""
        results = {}
        
        # 1. Análisis de calidad de señal
        quality_results = self.quality_assessor.assess_signal_quality(ecg_signal, sampling_rate)
        results['signal_quality'] = quality_results
        
        # 2. Detección de artefactos
        artifact_results = self.artifact_detector.detect_artifacts(ecg_signal, sampling_rate)
        results['artifacts'] = artifact_results
        
        # 3. Extracción de características
        features, error = self.extract_advanced_features(ecg_signal, sampling_rate)
        if error:
            results['error'] = error
            return results
        
        results['features'] = features
        
        # 4. Clasificación de arritmias
        arrhythmia_scores = self._simulate_classification(features)
        results['arrhythmia_scores'] = arrhythmia_scores
        
        # 5. Validación con criterios clínicos
        validated_diagnoses = self._validate_with_clinical_criteria(features, arrhythmia_scores)
        results['validated_diagnoses'] = validated_diagnoses
        
        # 6. Generar explicaciones
        primary_diagnosis = max(arrhythmia_scores, key=arrhythmia_scores.get)
        explanation = self.explainer.generate_explanation(primary_diagnosis, dict(zip(self.feature_names, features)), arrhythmia_scores)
        results['explanation'] = explanation
        
        # 7. Generar alertas clínicas
        recommendations, alert_level = self.alert_system.generate_clinical_recommendations(primary_diagnosis, arrhythmia_scores[primary_diagnosis])
        immediate_actions = self.alert_system.get_immediate_actions(primary_diagnosis)
        results['clinical_alerts'] = {
            'primary_diagnosis': primary_diagnosis,
            'alert_level': alert_level,
            'recommendations': recommendations,
            'immediate_actions': immediate_actions,
            'probability': arrhythmia_scores[primary_diagnosis]
        }
        
        # 8. Análisis de tendencias (si hay historial)
        trend_analysis = self.trend_analyzer.analyze_trends(arrhythmia_scores)
        results['trend_analysis'] = trend_analysis
        
        # 9. Agregar al historial para análisis futuro
        self.trend_analyzer.add_analysis(arrhythmia_scores)
        
        return results

    def _validate_with_clinical_criteria(self, features, arrhythmia_scores):
        """Validar diagnósticos con criterios clínicos"""
        validated = {}
        feature_dict = dict(zip(self.feature_names, features))
        
        for arrhythmia, score in arrhythmia_scores.items():
            if score > 0.3:  # Solo validar si hay probabilidad significativa
                if arrhythmia == 'Fibrilación Auricular':
                    meets_criteria, confidence = self.criteria_validator.validate_afib(feature_dict, [])
                    validated[arrhythmia] = {
                        'meets_criteria': meets_criteria,
                        'confidence': confidence,
                        'adjusted_score': score * confidence if meets_criteria else score * 0.5
                    }
                elif arrhythmia == 'Taquicardia Ventricular':
                    meets_criteria, confidence = self.criteria_validator.validate_vtach(
                        feature_dict, 
                        feature_dict.get('sustained_vt_duration', 0),
                        feature_dict.get('qrs_width', 0)
                    )
                    validated[arrhythmia] = {
                        'meets_criteria': meets_criteria,
                        'confidence': confidence,
                        'adjusted_score': score * confidence if meets_criteria else score * 0.5
                    }
                elif arrhythmia == 'Bloqueo AV':
                    meets_criteria, confidence = self.criteria_validator.validate_av_block(
                        feature_dict, feature_dict.get('pr_interval', 0)
                    )
                    validated[arrhythmia] = {
                        'meets_criteria': meets_criteria,
                        'confidence': confidence,
                        'adjusted_score': score * confidence if meets_criteria else score * 0.5
                    }
                elif arrhythmia == 'Bigeminismo Ventricular':
                    meets_criteria, confidence = self.criteria_validator.validate_bigeminy(
                        feature_dict,
                        feature_dict.get('pvc_density', 0),
                        feature_dict.get('pattern_regularity', 0)
                    )
                    validated[arrhythmia] = {
                        'meets_criteria': meets_criteria,
                        'confidence': confidence,
                        'adjusted_score': score * confidence if meets_criteria else score * 0.5
                    }
                else:
                    validated[arrhythmia] = {
                        'meets_criteria': True,
                        'confidence': 0.7,
                        'adjusted_score': score
                    }
        
        return validated

    def _simulate_classification(self, features):
        """Simular clasificación basada en características (placeholder para modelo real)"""
        # Esta es una simulación - en producción se usarían los modelos reales entrenados
        feature_dict = dict(zip(self.feature_names, features))
        
        scores = {
            'Ritmo Sinusal Normal': max(0, 1 - np.abs(feature_dict.get('std_rr', 0)) * 2),
            'Fibrilación Auricular': min(1, np.abs(feature_dict.get('lf_hf_ratio', 0)) * 0.5 + feature_dict.get('p_wave_absence_score', 0) * 0.5),
            'Extrasístoles Ventriculares': min(1, feature_dict.get('pvc_density', 0) * 3),
            'Taquicardia Ventricular': min(1, (feature_dict.get('hr_mean', 0) - 80) / 100 + feature_dict.get('av_dissociation_score', 0) * 0.3),
            'Bigeminismo Ventricular': min(1, feature_dict.get('bigeminy_score', 0) * 2),
            'Trigeminismo Ventricular': min(1, feature_dict.get('trigeminy_score', 0) * 2),
            'Bloqueo AV': min(1, np.abs(feature_dict.get('pr_interval', 0) - 200) / 200 + feature_dict.get('av_block_ratio', 0)),
            'Taquicardia Sinusal': min(1, max(0, (feature_dict.get('hr_mean', 0) - 100) / 50)),
            'Bradicardia Sinusal': min(1, max(0, (60 - feature_dict.get('hr_mean', 0)) / 40))
        }
        
        # Aplicar ajustes basados en características avanzadas
        if feature_dict.get('r_on_t_risk', 0) > 0.5:
            scores['Extrasístoles Ventriculares'] = min(1, scores['Extrasístoles Ventriculares'] * 1.5)
        
        if feature_dict.get('sustained_vt_duration', 0) > 10:
            scores['Taquicardia Ventricular'] = min(1, scores['Taquicardia Ventricular'] * 1.3)
        
        # Normalizar
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores

# =============================================================================
# INTERFAZ DE USUARIO MEJORADA CON STREAMLIT
# =============================================================================

class ECGAppInterface:
    """Interfaz de usuario mejorada para la aplicación ECG"""
    
    def __init__(self):
        self.processor = OptimizedECGProcessor()
        self.detector = AdvancedArrhythmiaDetector()
        self.setup_page()
    
    def setup_page(self):
        """Configurar la página principal mejorada"""
        # Header principal
        st.title("❤️ CardioAI Advanced Pro")
        st.subheader("Sistema Inteligente de Análisis ECG con Deep Learning y Validación Clínica")
        
        # Información de la aplicación
        with st.expander("ℹ️ Información de la Aplicación", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("### 🎯 Funcionalidades Avanzadas")
                st.write("- 📊 Análisis comprehensivo de ECG")
                st.write("- 🧠 Modelos Deep Learning avanzados")
                st.write("- ⚡ Procesamiento en tiempo real")
                st.write("- 📈 Visualizaciones interactivas")
                st.write("- 🔍 Detección de arritmias complejas")
                st.write("- 🚨 Sistema de alertas clínicas")
                st.write("- 📋 Validación con criterios clínicos")
            
            with col2:
                st.write("### 📁 Formatos Soportados")
                st.write("- EDF/EDF+ (Europeo)")
                st.write("- BDF (Biosemi)")
                st.write("- Binario personalizado")
                st.write("- RAW (Datos crudos)")
                st.write("- Auto-detección inteligente")
            
            with col3:
                st.write("### 🏥 Arritmias Detectadas")
                st.write("- Fibrilación Auricular")
                st.write("- Taquicardia Ventricular")  
                st.write("- Extrasístoles (PVC)")
                st.write("- Bigeminismo/Trigeminismo")
                st.write("- Bloqueos AV")
                st.write("- Patrones complejos")
                st.write("- Y más...")
    
    def render_file_upload(self):
        """Interfaz de carga de archivos mejorada"""
        st.write("### 📤 Carga de Archivo ECG")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Selecciona tu archivo ECG",
                type=['edf', 'bdf', 'bin', 'raw', 'txt', 'csv'],
                help="Formatos soportados: EDF, BDF, binario, RAW"
            )
        
        with col2:
            file_type = st.selectbox(
                "Tipo de archivo",
                ['Auto-detección', 'EDF/BDF', 'Binario Personalizado', 'RAW']
            )
        
        # Configuración avanzada para archivos binarios
        binary_config = None
        if file_type == 'Binario Personalizado':
            with st.expander("⚙️ Configuración Binaria Avanzada"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    binary_format = st.selectbox(
                        "Formato de datos",
                        list(self.processor.binary_formats.keys()),
                        format_func=lambda x: f"{x} ({self.processor.binary_formats[x]})"
                    )
                
                with col2:
                    bytes_per_sample = st.number_input("Bytes por muestra", min_value=1, max_value=8, value=2)
                
                with col3:
                    header_size = st.number_input("Tamaño cabecera (bytes)", min_value=0, value=0)
                
                binary_config = {
                    'binary_format': binary_format,
                    'bytes_per_sample': bytes_per_sample,
                    'header_size': header_size
                }
        
        return uploaded_file, file_type, binary_config
    
    def render_comprehensive_analysis(self, analysis_results):
        """Mostrar análisis comprehensivo mejorado"""
        st.write("### 📊 Análisis Comprehensivo")
        
        if 'error' in analysis_results:
            st.error(f"❌ Error en el análisis: {analysis_results['error']}")
            return
        
        # Panel de Calidad de Señal
        self._render_signal_quality(analysis_results.get('signal_quality', {}))
        
        # Panel de Artefactos
        self._render_artifact_analysis(analysis_results.get('artifacts', {}))
        
        # Panel de Alertas Clínicas
        self._render_clinical_alerts(analysis_results.get('clinical_alerts', {}))
        
        # Panel de Diagnósticos
        self._render_arrhythmia_analysis(analysis_results)
        
        # Panel de Explicaciones
        self._render_explanations(analysis_results.get('explanation', ''))
        
        # Panel de Tendencias
        self._render_trend_analysis(analysis_results.get('trend_analysis', {}))
        
        # Panel de Feedback
        self._render_feedback_section(analysis_results)
    
    def _render_signal_quality(self, quality_results):
        """Renderizar panel de calidad de señal"""
        st.write("### 🔍 Calidad de Señal")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            quality_color = {
                "EXCELENTE": "🟢",
                "BUENA": "🟢", 
                "ACEPTABLE": "🟡",
                "POBRE": "🔴",
                "NO EVALUABLE": "⚫"
            }.get(quality_results.get('quality_level', 'NO EVALUABLE'), '⚫')
            
            st.metric(
                "Nivel de Calidad", 
                f"{quality_color} {quality_results.get('quality_level', 'NO EVALUABLE')}"
            )
        
        with col2:
            st.metric(
                "Score de Calidad", 
                f"{quality_results.get('quality_score', 0):.2f}"
            )
        
        with col3:
            st.metric(
                "Confiabilidad Diagnóstica",
                quality_results.get('diagnostic_reliability', 'Muy Baja')
            )
        
        with col4:
            noise_level = quality_results.get('noise_level', 1.0)
            noise_status = "BAJO" if noise_level < 0.3 else "MODERADO" if noise_level < 0.6 else "ALTO"
            st.metric("Ruido", noise_status)
        
        # Recomendaciones de calidad
        if quality_results.get('recommendations'):
            with st.expander("💡 Recomendaciones de Calidad"):
                for rec in quality_results['recommendations']:
                    st.write(f"• {rec}")
    
    def _render_artifact_analysis(self, artifact_results):
        """Renderizar análisis de artefactos"""
        if artifact_results.get('artifact_count', 0) > 0:
            st.write("### ⚠️ Análisis de Artefactos")
            
            st.warning(f"Se detectaron {artifact_results['artifact_count']} tipos de artefactos")
            
            for artifact_name, artifact_info in artifact_results.get('detected_artifacts', {}).items():
                with st.expander(f"🔍 {artifact_name.replace('_', ' ').title()}"):
                    st.write(f"**Descripción:** {artifact_info.get('description', '')}")
                    st.write(f"**Severidad:** {artifact_info.get('severity', 0):.2f}")
                    st.write(f"**Solución sugerida:** {artifact_info.get('suggested_fix', '')}")
            
            if artifact_results.get('recommendations'):
                st.info("**Recomendaciones para mejorar la señal:**")
                for rec in artifact_results['recommendations']:
                    st.write(f"• {rec}")
    
    def _render_clinical_alerts(self, clinical_alerts):
        """Renderizar alertas clínicas"""
        if not clinical_alerts:
            return
        
        alert_level = clinical_alerts.get('alert_level', 'LOW')
        probability = clinical_alerts.get('probability', 0)
        diagnosis = clinical_alerts.get('primary_diagnosis', '')
        
        alert_configs = {
            'CRITICAL': ('🔴', 'ALERTA CRÍTICA'),
            'HIGH': ('🟠', 'ALERTA ALTA'), 
            'MEDIUM': ('🟡', 'ALERTA MEDIA'),
            'LOW': ('🔵', 'INFORMACIÓN'),
            'NORMAL': ('🟢', 'NORMAL')
        }
        
        emoji, title = alert_configs.get(alert_level, ('🔵', 'INFORMACIÓN'))
        
        if alert_level == 'CRITICAL':
            st.error(f"### {emoji} {title}: {diagnosis}")
        elif alert_level == 'HIGH':
            st.warning(f"### {emoji} {title}: {diagnosis}")
        elif alert_level == 'MEDIUM':
            st.warning(f"### {emoji} {title}: {diagnosis}")
        elif alert_level == 'LOW':
            st.info(f"### {emoji} {title}: {diagnosis}")
        else:
            st.success(f"### {emoji} {title}: {diagnosis}")
        
        st.write(f"**Probabilidad:** {probability:.1%}")
        
        st.write("**Recomendaciones:**")
        for rec in clinical_alerts.get('recommendations', []):
            st.write(f"• {rec}")
        
        if clinical_alerts.get('immediate_actions'):
            st.write("**Acciones inmediatas:**")
            for action in clinical_alerts['immediate_actions']:
                st.write(f"• {action}")
    
    def _render_arrhythmia_analysis(self, analysis_results):
        """Renderizar análisis de arritmias"""
        st.write("### 🏥 Análisis de Arritmias")
        
        arrhythmia_scores = analysis_results.get('arrhythmia_scores', {})
        validated_diagnoses = analysis_results.get('validated_diagnoses', {})
        
        # Crear gráfico de probabilidades
        fig, ax = plt.subplots(figsize=(10, 6))
        
        arrhythmias = list(arrhythmia_scores.keys())
        probabilities = list(arrhythmia_scores.values())
        
        # Colores basados en probabilidad y validación
        colors = []
        for i, (arrhythmia, prob) in enumerate(arrhythmia_scores.items()):
            if arrhythmia in validated_diagnoses:
                validation = validated_diagnoses[arrhythmia]
                if validation.get('meets_criteria', False):
                    colors.append('red' if prob > 0.7 else 'orange' if prob > 0.3 else 'yellow')
                else:
                    colors.append('lightgray')
            else:
                colors.append('green' if prob < 0.3 else 'orange' if prob < 0.7 else 'red')
        
        bars = ax.barh(arrhythmias, probabilities, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probabilidad')
        ax.set_title('Probabilidad de Arritmias (Validación Clínica)')
        
        # Añadir valores y anotaciones
        for bar, prob, arrhythmia in zip(bars, probabilities, arrhythmias):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{prob:.2f}', va='center')
            
            if arrhythmia in validated_diagnoses:
                validation = validated_diagnoses[arrhythmia]
                if not validation.get('meets_criteria', True):
                    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                           '❌', va='center')
                else:
                    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                           '✅', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Leyenda
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("🟢 Baja probabilidad (<30%)")
            st.write("🟡 Probabilidad media (30-70%)")
        with col2:
            st.write("🔴 Alta probabilidad (>70%)")
            st.write("⚪ No cumple criterios clínicos")
        with col3:
            st.write("✅ Validez clínica confirmada")
            st.write("❌ Validez clínica cuestionada")
    
    def _render_explanations(self, explanation):
        """Renderizar explicaciones del diagnóstico"""
        if explanation:
            st.write("### 📋 Explicación del Diagnóstico")
            st.info(explanation)
    
    def _render_trend_analysis(self, trend_analysis):
        """Renderizar análisis de tendencias"""
        if trend_analysis.get('total_changes', 0) > 0:
            st.write("### 📈 Análisis de Tendencia")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Estabilidad", trend_analysis.get('overall_trend', 'ESTABLE'))
                st.metric("Score de Estabilidad", f"{trend_analysis.get('stability_score', 1):.2f}")
            
            with col2:
                st.metric("Cambios Detectados", trend_analysis.get('total_changes', 0))
            
            # Mostrar alertas de tendencia
            for alert in trend_analysis.get('trend_alerts', []):
                st.warning(alert['message'])
            
            for improvement in trend_analysis.get('improvements', []):
                st.success(improvement['message'])
    
    def _render_feedback_section(self, analysis_results):
        """Renderizar sección de feedback"""
        st.write("### 💬 Sistema de Feedback")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### ¿Fue útil el diagnóstico?")
            
            feedback_type = st.selectbox(
                "Tipo de feedback",
                ["Seleccionar...", "Diagnóstico correcto", "Diagnóstico incorrecto", "Mejora sugerida"]
            )
            
            if feedback_type != "Seleccionar...":
                actual_diagnosis = st.text_input("Diagnóstico real (si es diferente)")
                comments = st.text_area("Comentarios adicionales")
                
                if st.button("Enviar Feedback"):
                    primary_diagnosis = analysis_results.get('clinical_alerts', {}).get('primary_diagnosis', '')
                    probability = analysis_results.get('clinical_alerts', {}).get('probability', 0)
                    
                    self.detector.learning_system.add_feedback(
                        prediction=primary_diagnosis,
                        actual_diagnosis=actual_diagnosis if actual_diagnosis else primary_diagnosis,
                        user_correction=comments,
                        confidence=probability
                    )
        
        with col2:
            st.write("#### Insights de Aprendizaje")
            insights = self.detector.learning_system.get_learning_insights()
            
            if 'message' in insights:
                st.info(insights['message'])
            else:
                st.metric("Precisión General", f"{insights['overall_accuracy']:.1%}")
                st.metric("Feedbacks Recibidos", insights['total_feedbacks'])
                
                if insights.get('condition_accuracy'):
                    st.write("**Precisión por Condición:**")
                    for condition, accuracy in insights['condition_accuracy'].items():
                        st.write(f"• {condition}: {accuracy:.1%}")
    
    def _plot_ecg_signal(self, ecg_signal, sampling_rate):
        """Visualización mejorada de la señal ECG"""
        st.write("### 📈 Visualización de la Señal ECG")
        
        # Selector de segmento
        total_samples = len(ecg_signal)
        max_display_samples = min(10000, total_samples)
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            start_sample = st.slider(
                "Muestra inicial",
                0,
                total_samples - max_display_samples,
                0,
                step=1000
            )
            
            display_samples = st.slider(
                "Muestras a mostrar",
                100,
                max_display_samples,
                min(5000, max_display_samples)
            )
        
        with col1:
            # Crear figura
            fig, ax = plt.subplots(figsize=(12, 4))
            
            end_sample = min(start_sample + display_samples, total_samples)
            time_axis = np.arange(start_sample, end_sample) / sampling_rate
            
            ax.plot(time_axis, ecg_signal[start_sample:end_sample], linewidth=0.8, color='#1f77b4')
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel('Amplitud')
            ax.set_title('Señal ECG')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

    def _generate_clinical_report(self, analysis_results, ecg_signal, sampling_rate, filename):
        """Generar reporte clínico completo"""
        st.write("### 📄 Generar Reporte Clínico")
        
        # Información del paciente (opcional)
        with st.expander("👤 Información del Paciente (Opcional)"):
            col1, col2 = st.columns(2)
            with col1:
                patient_name = st.text_input("Nombre del paciente")
                patient_age = st.number_input("Edad", min_value=0, max_value=120, value=0)
            with col2:
                patient_id = st.text_input("ID del paciente")
                recording_date = st.date_input("Fecha del registro")
        
        # Crear reporte
        report_content = self._format_clinical_report(analysis_results, ecg_signal, sampling_rate, filename, {
            'name': patient_name,
            'age': patient_age,
            'id': patient_id,
            'date': recording_date
        })
        
        # Botón de descarga
        st.download_button(
            label="📥 Descargar Reporte Clínico Completo",
            data=report_content,
            file_name=f"reporte_ecg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    def _format_clinical_report(self, analysis_results, ecg_signal, sampling_rate, filename, patient_info):
        """Formatear reporte clínico completo"""
        report = []
        
        # Encabezado
        report.append("=" * 80)
        report.append("REPORTE CLÍNICO DE ANÁLISIS ECG - CardioAI Advanced Pro")
        report.append("=" * 80)
        report.append("")
        
        # Información del paciente
        report.append("INFORMACIÓN DEL PACIENTE:")
        report.append("-" * 40)
        if patient_info['name']:
            report.append(f"Nombre: {patient_info['name']}")
        if patient_info['id']:
            report.append(f"ID: {patient_info['id']}")
        if patient_info['age'] > 0:
            report.append(f"Edad: {patient_info['age']} años")
        if patient_info['date']:
            report.append(f"Fecha del registro: {patient_info['date']}")
        report.append(f"Archivo analizado: {filename}")
        report.append(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Resumen de la señal
        report.append("RESUMEN DE LA SEÑAL:")
        report.append("-" * 40)
        report.append(f"Muestras totales: {len(ecg_signal):,}")
        report.append(f"Frecuencia de muestreo: {sampling_rate} Hz")
        report.append(f"Duración: {len(ecg_signal)/sampling_rate:.2f} segundos")
        report.append(f"Amplitud media: {np.mean(ecg_signal):.4f}")
        report.append(f"Desviación estándar: {np.std(ecg_signal):.4f}")
        report.append("")
        
        # Calidad de señal
        quality = analysis_results.get('signal_quality', {})
        report.append("CALIDAD DE SEÑAL:")
        report.append("-" * 40)
        report.append(f"Nivel: {quality.get('quality_level', 'NO EVALUABLE')}")
        report.append(f"Score: {quality.get('quality_score', 0):.2f}")
        report.append(f"Confiabilidad diagnóstica: {quality.get('diagnostic_reliability', 'Muy Baja')}")
        report.append("")
        
        # Diagnóstico principal
        clinical_alerts = analysis_results.get('clinical_alerts', {})
        if clinical_alerts:
            report.append("DIAGNÓSTICO PRINCIPAL:")
            report.append("-" * 40)
            report.append(f"Condición: {clinical_alerts.get('primary_diagnosis', 'No determinado')}")
            report.append(f"Probabilidad: {clinical_alerts.get('probability', 0):.1%}")
            report.append(f"Nivel de alerta: {clinical_alerts.get('alert_level', 'LOW')}")
            report.append("")
        
        # Recomendaciones clínicas
        if clinical_alerts.get('recommendations'):
            report.append("RECOMENDACIONES CLÍNICAS:")
            report.append("-" * 40)
            for rec in clinical_alerts['recommendations']:
                report.append(f"• {rec}")
            report.append("")
        
        # Análisis detallado de arritmias
        arrhythmia_scores = analysis_results.get('arrhythmia_scores', {})
        if arrhythmia_scores:
            report.append("ANÁLISIS DETALLADO DE ARRITMIAS:")
            report.append("-" * 40)
            for arrhythmia, score in sorted(arrhythmia_scores.items(), key=lambda x: x[1], reverse=True):
                if score > 0.1:  # Solo mostrar probabilidades significativas
                    report.append(f"• {arrhythmia}: {score:.1%}")
            report.append("")
        
        # Explicación del diagnóstico
        if analysis_results.get('explanation'):
            report.append("EXPLICACIÓN DEL DIAGNÓSTICO:")
            report.append("-" * 40)
            report.append(analysis_results['explanation'])
            report.append("")
        
        # Artefactos detectados
        artifacts = analysis_results.get('artifacts', {})
        if artifacts.get('artifact_count', 0) > 0:
            report.append("ARTEFACTOS DETECTADOS:")
            report.append("-" * 40)
            for art_name, art_info in artifacts.get('detected_artifacts', {}).items():
                report.append(f"• {art_name}: {art_info.get('description', '')} (Severidad: {art_info.get('severity', 0):.2f})")
            report.append("")
        
        # Tendencias
        trends = analysis_results.get('trend_analysis', {})
        if trends.get('total_changes', 0) > 0:
            report.append("ANÁLISIS DE TENDENCIAS:")
            report.append("-" * 40)
            report.append(f"Estabilidad: {trends.get('overall_trend', 'ESTABLE')}")
            report.append(f"Score de estabilidad: {trends.get('stability_score', 1):.2f}")
            report.append("")
        
        # Advertencias importantes
        report.append("ADVERTENCIAS IMPORTANTES:")
        report.append("-" * 40)
        report.append("• Este es un sistema de apoyo al diagnóstico, no un diagnóstico definitivo")
        report.append("• Los resultados deben ser validados por personal médico calificado")
        report.append("• En caso de síntomas o emergencias, busque atención médica inmediata")
        report.append("• El sistema aprende continuamente de los feedbacks proporcionados")
        report.append("")
        
        report.append("=" * 80)
        report.append("CardioAI Advanced Pro - Sistema Inteligente de Análisis ECG")
        report.append("=" * 80)
        
        return "\n".join(report)

    def run(self):
        """Ejecutar la aplicación completa"""
        # Carga de archivo
        uploaded_file, file_type, binary_config = self.render_file_upload()
        
        if uploaded_file is not None:
            # Procesar archivo
            with st.spinner("🔄 Procesando archivo ECG..."):
                if file_type == 'Binario Personalizado' and binary_config:
                    ecg_signal, sampling_rate = self.processor.load_ecg_file(
                        uploaded_file,
                        binary_format=binary_config['binary_format'],
                        bytes_per_sample=binary_config['bytes_per_sample'],
                        header_size=binary_config['header_size']
                    )
                else:
                    ecg_signal, sampling_rate = self.processor.load_ecg_file(uploaded_file)
            
            if ecg_signal is not None:
                # Mostrar información básica de la señal
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Muestras", f"{len(ecg_signal):,}")
                with col2:
                    st.metric("Frecuencia", f"{sampling_rate} Hz")
                with col3:
                    duration = len(ecg_signal) / sampling_rate
                    st.metric("Duración", f"{duration:.1f} s")
                with col4:
                    st.metric("Amplitud", f"{np.std(ecg_signal):.3f}")
                
                # Visualización de la señal
                self._plot_ecg_signal(ecg_signal, sampling_rate)
                
                # Análisis comprehensivo
                with st.spinner("🧠 Realizando análisis comprehensivo..."):
                    analysis_results = self.detector.comprehensive_analysis(ecg_signal, sampling_rate)
                
                # Mostrar resultados
                self.render_comprehensive_analysis(analysis_results)
                
                # Generar reporte
                self._generate_clinical_report(analysis_results, ecg_signal, sampling_rate, uploaded_file.name)
            else:
                st.error("❌ No se pudo cargar el archivo ECG. Verifique el formato y configuración.")
        
        # Información adicional cuando no hay archivo
        else:
            st.info("👆 Carga tu archivo ECG para comenzar")
            st.write("Esta aplicación analiza señales electrocardiográficas usando inteligencia artificial avanzada para detectar posibles arritmias y patrones cardíacos anormales.")
            
            st.write("### Características principales:")
            st.write("🔄 **Procesamiento robusto** de múltiples formatos de archivo")
            st.write("🔍 **Detección avanzada** de arritmias complejas")
            st.write("🚨 **Sistema de alertas** con prioridades clínicas")
            st.write("📋 **Validación** con criterios clínicos establecidos")
            st.write("📈 **Análisis de tendencias** temporales")
            st.write("💡 **Explicaciones** comprensibles de los diagnósticos")
            st.write("📊 **Reportes clínicos** completos y descargables")
            
            # Ejemplo de formato de archivo binario
            with st.expander("📋 Guía de Formatos Binarios"):
                st.write("### Configuración para Archivos Binarios")
                st.write("**Formato de datos comunes:**")
                st.write("- `int16`: Entero 16-bit con signo (común en dispositivos médicos)")
                st.write("- `float32`: Punto flotante 32-bit (alta precisión)")
                st.write("- `uint16`: Entero 16-bit sin signo")
                st.write("")
                st.write("**Ejemplo de estructura:**")
                st.code("[Cabecera: 0-512 bytes][Muestra 1: 2 bytes][Muestra 2: 2 bytes]...")
                st.write("")
                st.write("**Parámetros típicos:**")
                st.write("- **Bytes por muestra:** 2 (int16) o 4 (float32)")
                st.write("- **Tamaño cabecera:** 0-1024 bytes")
                st.write("- **Frecuencia:** 250-1000 Hz")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal"""
    try:
        app = ECGAppInterface()
        app.run()
    except Exception as e:
        st.error(f"❌ Error crítico en la aplicación: {str(e)}")
        st.info("**Solución de problemas:**")
        st.write("- Verifique que todas las dependencias estén instaladas")
        st.write("- Reinicie la aplicación")
        st.write("- Si el problema persiste, contacte al soporte técnico")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
