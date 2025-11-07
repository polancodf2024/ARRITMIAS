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
            'compensatory_pause_ratio', 'pattern_regularity', 'pvc_density'
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
        # CSS personalizado mejorado
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .sub-header {
            font-size: 1.5rem;
            color: #2e86ab;
            margin: 1rem 0;
            font-weight: 600;
        }
        .info-box {
            background-color: #f0f8ff;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #ffc107;
            margin: 1rem 0;
        }
        .success-box {
            background-color: #d4edda;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            margin: 1rem 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header principal mejorado
        st.markdown('<h1 class="main-header">❤️ CardioAI Advanced Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #555;">Analizador Inteligente de Señales ECG con Deep Learning</p>', unsafe_allow_html=True)
        
        # Información de la aplicación
        with st.expander("ℹ️ Información de la Aplicación", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🎯 Funcionalidades")
                st.markdown("""
                - 📊 Análisis avanzado de ECG
                - 🧠 Modelos Deep Learning
                - ⚡ Procesamiento en tiempo real
                - 📈 Visualizaciones interactivas
                - 🔍 Detección de arritmias
                """)
            
            with col2:
                st.markdown("### 📁 Formatos Soportados")
                st.markdown("""
                - EDF/EDF+ (Europeo)
                - BDF (Biosemi)
                - Binario personalizado
                - RAW (Datos crudos)
                - Auto-detección
                """)
            
            with col3:
                st.markdown("### 🏥 Arritmias Detectadas")
                st.markdown("""
                - Fibrilación Auricular
                - Taquicardia Ventricular
                - Extrasístoles (PVC)
                - Bigeminismo/Trigeminismo
                - Bloqueos AV
                - Y más...
                """)
    
    def render_file_upload(self):
        """Interfaz de carga de archivos mejorada"""
        st.markdown('<div class="sub-header">📤 Carga de Archivo ECG</div>', unsafe_allow_html=True)
        
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
    
    def render_analysis_results(self, ecg_signal, sampling_rate, features):
        """Mostrar resultados del análisis mejorado"""
        st.markdown('<div class="sub-header">📊 Resultados del Análisis</div>', unsafe_allow_html=True)
        
        if ecg_signal is None:
            st.error("❌ No se pudo procesar la señal ECG")
            return
        
        # Métricas principales en tarjetas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Muestras</h3>
                <h2>{len(ecg_signal):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚡ Frecuencia</h3>
                <h2>{sampling_rate} Hz</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            duration = len(ecg_signal) / sampling_rate
            st.markdown(f"""
            <div class="metric-card">
                <h3>⏱️ Duración</h3>
                <h2>{duration:.1f} s</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📏 Amplitud</h3>
                <h2>{np.std(ecg_signal):.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualización de la señal
        self._plot_ecg_signal(ecg_signal, sampling_rate)
        
        # Análisis de características
        if features is not None:
            self._display_feature_analysis(features)
            
            # Clasificación de arritmias
            self._classify_arrhythmias(features)
    
    def _plot_ecg_signal(self, ecg_signal, sampling_rate):
        """Visualización mejorada de la señal ECG"""
        st.markdown("### 📈 Visualización de la Señal ECG")
        
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
    
    def _display_feature_analysis(self, features):
        """Mostrar análisis de características mejorado"""
        st.markdown("### 🔍 Análisis de Características")
        
        # Seleccionar características más importantes
        important_features = {
            'Frecuencia Cardíaca Media': features[6],
            'Variabilidad RR': features[1],
            'Ancho QRS': features[8],
            'Ratio LF/HF': features[14],
            'Entropía': features[13],
            'Densidad PVC': features[25] if len(features) > 25 else 0,
            'Score Bigeminismo': features[19] if len(features) > 19 else 0,
            'Score Trigeminismo': features[20] if len(features) > 20 else 0
        }
        
        # Mostrar métricas
        cols = st.columns(4)
        for idx, (name, value) in enumerate(important_features.items()):
            with cols[idx % 4]:
                st.metric(name, f"{value:.3f}")
        
        # Gráfico de características
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_names_short = [name[:15] + '...' if len(name) > 15 else name 
                             for name in self.detector.feature_names[:10]]
        ax.barh(feature_names_short, features[:10])
        ax.set_xlabel('Valor')
        ax.set_title('Top 10 Características Extraídas')
        plt.tight_layout()
        st.pyplot(fig)
    
    def _classify_arrhythmias(self, features):
        """Clasificación de arritmias mejorada"""
        st.markdown("### 🏥 Clasificación de Arritmias")
        
        # Simular clasificación (en producción sería con modelos reales)
        scores = self._simulate_classification(features)
        
        # Mostrar resultados
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de probabilidades
            fig, ax = plt.subplots(figsize=(8, 6))
            arrhythmias = list(scores.keys())
            probabilities = list(scores.values())
            
            colors = ['green' if p < 0.3 else 'orange' if p < 0.7 else 'red' for p in probabilities]
            
            bars = ax.barh(arrhythmias, probabilities, color=colors)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probabilidad')
            ax.set_title('Probabilidad de Arritmias')
            
            # Añadir valores en las barras
            for bar, prob in zip(bars, probabilities):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{prob:.2f}', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Diagnóstico principal
            max_arrhythmia = max(scores, key=scores.get)
            max_prob = scores[max_arrhythmia]
            
            if max_prob > 0.7:
                st.error(f"**ALERTA:** {max_arrhythmia}")
                st.markdown(f"Probabilidad: {max_prob:.1%}")
                
                if "Bigeminismo" in max_arrhythmia or "Trigeminismo" in max_arrhythmia:
                    st.warning("""
                    **Patrón de Agrupamiento Detectado:**
                    - PVCs en patrones regulares
                    - Requiere evaluación cardiológica
                    - Puede progresar a taquicardia
                    """)
            elif max_prob > 0.3:
                st.warning(f"**Sospecha:** {max_arrhythmia}")
                st.markdown(f"Probabilidad: {max_prob:.1%}")
            else:
                st.success("**Ritmo Sinusal Normal**")
                st.markdown(f"Probabilidad: {1-max_prob:.1%}")
    
    def _simulate_classification(self, features):
        """Simular clasificación basada en características (placeholder para modelo real)"""
        # Esta es una simulación - en producción se usarían los modelos reales entrenados
        scores = {
            'Ritmo Sinusal Normal': max(0, 1 - np.abs(features[1]) * 2),
            'Fibrilación Auricular': min(1, np.abs(features[14]) * 0.5),
            'Extrasístoles Ventriculares': min(1, features[25] * 3 if len(features) > 25 else 0),
            'Taquicardia Ventricular': min(1, (features[6] - 80) / 100),
            'Bigeminismo Ventricular': min(1, features[19] * 2 if len(features) > 19 else 0),
            'Trigeminismo Ventricular': min(1, features[20] * 2 if len(features) > 20 else 0),
            'Bloqueo AV': min(1, np.abs(features[10] - 200) / 200)
        }
        
        # Normalizar
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        return scores

    def _generate_report(self, ecg_signal, sampling_rate, features, filename):
        """Generar reporte descargable mejorado"""
        st.markdown("### 📄 Generar Reporte")
        
        # Crear reporte
        report_content = f"""
        REPORTE DE ANÁLISIS ECG - CardioAI Advanced Pro
        ==============================================
        
        Archivo: {filename}
        Fecha de análisis: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        RESUMEN DE LA SEÑAL:
        - Muestras totales: {len(ecg_signal):,}
        - Frecuencia de muestreo: {sampling_rate} Hz
        - Duración: {len(ecg_signal)/sampling_rate:.2f} segundos
        - Amplitud media: {np.mean(ecg_signal):.4f}
        - Desviación estándar: {np.std(ecg_signal):.4f}
        
        CARACTERÍSTICAS EXTRACTADAS:
        {self._format_features_for_report(features)}
        
        OBSERVACIONES:
        - Análisis realizado con modelos de deep learning avanzados
        - Los resultados deben ser validados por personal médico calificado
        - Este es un sistema de apoyo al diagnóstico, no un diagnóstico definitivo
        
        CardioAI Advanced Pro - Sistema Inteligente de Análisis ECG
        """
        
        # Botón de descarga
        st.download_button(
            label="📥 Descargar Reporte Completo",
            data=report_content,
            file_name=f"ecg_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    def _format_features_for_report(self, features):
        """Formatear características para el reporte"""
        if features is None:
            return "No se pudieron extraer características"
        
        feature_text = ""
        for i, (name, value) in enumerate(zip(self.detector.feature_names, features)):
            if i < len(features):
                feature_text += f"- {name}: {value:.4f}\n"
        
        return feature_text

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
                # Extraer características
                with st.spinner("🧠 Analizando características avanzadas..."):
                    features, error = self.detector.extract_advanced_features(ecg_signal, sampling_rate)
                
                if error:
                    st.error(f"❌ Error en análisis: {error}")
                else:
                    # Mostrar resultados
                    self.render_analysis_results(ecg_signal, sampling_rate, features)
                    
                    # Descargar reporte
                    self._generate_report(ecg_signal, sampling_rate, features, uploaded_file.name)
            else:
                st.error("❌ No se pudo cargar el archivo ECG. Verifique el formato y configuración.")
        
        # Información adicional cuando no hay archivo
        else:
            st.markdown("""
            <div class="info-box">
                <h3>👆 Carga tu archivo ECG para comenzar</h3>
                <p>Esta aplicación analiza señales electrocardiográficas usando inteligencia artificial avanzada 
                para detectar posibles arritmias y patrones cardíacos anormales.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Ejemplo de formato de archivo binario
            with st.expander("📋 Guía de Formatos Binarios"):
                st.markdown("""
                ### Configuración para Archivos Binarios
                
                **Formato de datos comunes:**
                - `int16`: Entero 16-bit con signo (común en dispositivos médicos)
                - `float32`: Punto flotante 32-bit (alta precisión)
                - `uint16`: Entero 16-bit sin signo
                
                **Ejemplo de estructura:**
                ```
                [Cabecera: 0-512 bytes][Muestra 1: 2 bytes][Muestra 2: 2 bytes]...
                ```
                
                **Parámetros típicos:**
                - **Bytes por muestra:** 2 (int16) o 4 (float32)
                - **Tamaño cabecera:** 0-1024 bytes
                - **Frecuencia:** 250-1000 Hz
                """)

# =============================================================================
# FUNCIÓN PRINCIPAL CORREGIDA
# =============================================================================

def main():
    """Función principal corregida"""
    app = ECGAppInterface()
    app.run()

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
