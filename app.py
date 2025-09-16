import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from PIL import Image
import io
import rasterio
from rasterio.plot import show
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Procesador de Imágenes Multiespectrales",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stImage > img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def load_image(uploaded_file):
    """Carga una imagen desde el archivo subido"""
    try:
        if uploaded_file.name.lower().endswith(('.tiff', '.tif')):
            # Para archivos TIFF/GeoTIFF
            with rasterio.open(uploaded_file) as src:
                image = src.read()
                if len(image.shape) == 3:
                    image = np.transpose(image, (1, 2, 0))
                return image, src.meta
        else:
            # Para otros formatos
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            return img_array, None
    except Exception as e:
        st.error(f"Error al cargar la imagen: {str(e)}")
        return None, None

def process_image(image, band, binarize, bins_count):
    """Procesa la imagen según los parámetros seleccionados"""
    if len(image.shape) == 3:
        if band <= image.shape[2]:
            img_band = image[:, :, band-1]
        else:
            img_band = image[:, :, 0]  # Fallback a la primera banda
    else:
        img_band = image
    
    # Normalizar si es necesario
    if img_band.dtype != np.uint8:
        img_band = cv2.normalize(img_band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    processed_img = img_band.copy()
    
    if binarize:
        thresh = threshold_otsu(img_band)
        processed_img = (img_band > thresh).astype(np.uint8) * 255
        
    return img_band, processed_img, thresh if binarize else None

def create_histogram(image, bins_count, title="Histograma"):
    """Crea un histograma de la imagen"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(image.flatten(), bins=bins_count, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Valor de píxel')
    ax.set_ylabel('Frecuencia')
    ax.grid(True, alpha=0.3)
    
    # Añadir estadísticas al gráfico
    mean_val = np.mean(image)
    std_val = np.std(image)
    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Media: {mean_val:.2f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_val + std_val:.2f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_val - std_val:.2f}')
    ax.legend()
    
    return fig

def calculate_statistics(image):
    """Calcula estadísticas básicas de la imagen"""
    return {
        'Media': np.mean(image),
        'Mediana': np.median(image),
        'Desviación Estándar': np.std(image),
        'Varianza': np.var(image),
        'Mínimo': np.min(image),
        'Máximo': np.max(image),
        'Rango': np.max(image) - np.min(image),
        'Píxeles totales': image.size
    }

# INTERFAZ PRINCIPAL
def main():
    # Título principal
    st.markdown('<h1 class="main-header">🛰️ Procesador de Imágenes Multiespectrales</h1>', unsafe_allow_html=True)
    
    # Sidebar con controles
    with st.sidebar:
        st.header("🎛️ Panel de Control")
        
        # Upload de archivo
        uploaded_file = st.file_uploader(
            "📁 Subir Imagen",
            type=['jpg', 'jpeg', 'png', 'tiff', 'tif'],
            help="Formatos soportados: JPG, PNG, TIFF"
        )
        
        st.divider()
        
        if uploaded_file is not None:
            # Cargar imagen
            image, metadata = load_image(uploaded_file)
            
            if image is not None:
                # Información de la imagen
                st.subheader("📋 Información de la Imagen")
                st.write(f"**Archivo:** {uploaded_file.name}")
                st.write(f"**Dimensiones:** {image.shape}")
                st.write(f"**Tipo:** {image.dtype}")
                
                if len(image.shape) == 3:
                    st.write(f"**Bandas disponibles:** {image.shape[2]}")
                    max_band = image.shape[2]
                else:
                    st.write("**Tipo:** Imagen en escala de grises")
                    max_band = 1
                
                st.divider()
                
                # Controles de procesamiento
                st.subheader("⚙️ Parámetros de Procesamiento")
                
                band = st.selectbox(
                    "🎨 Banda a visualizar:",
                    options=list(range(1, max_band + 1)),
                    index=0,
                    help="Selecciona qué banda espectral mostrar"
                )
                
                binarize = st.checkbox(
                    "🔲 Binarizar (Otsu)",
                    value=False,
                    help="Aplica umbralización automática usando el método de Otsu"
                )
                
                bins_count = st.slider(
                    "📊 Bins del histograma:",
                    min_value=32,
                    max_value=512,
                    value=256,
                    step=32,
                    help="Número de contenedores para el histograma"
                )
                
                st.divider()
                
                # Opciones adicionales
                st.subheader("🔧 Opciones Avanzadas")
                
                show_statistics = st.checkbox("📈 Mostrar estadísticas", value=True)
                show_comparison = st.checkbox("🔍 Comparación lado a lado", value=True)
                color_map = st.selectbox(
                    "🎨 Mapa de colores:",
                    options=['gray', 'viridis', 'plasma', 'inferno', 'magma', 'jet'],
                    index=0
                )

    # Área principal
    if uploaded_file is not None and image is not None:
        
        # Procesar imagen
        original_band, processed_img, threshold_value = process_image(image, band, binarize, bins_count)
        
        # Layout principal
        if show_comparison:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"🖼️ Original - Banda {band}")
                st.image(original_band, caption=f"Banda {band} original", use_column_width=True, clamp=True)
                
            with col2:
                st.subheader("⚡ Procesada")
                if binarize and threshold_value is not None:
                    st.image(processed_img, caption=f"Binarizada (umbral: {threshold_value:.2f})", use_column_width=True, clamp=True)
                    st.info(f"🎯 Umbral automático (Otsu): **{threshold_value:.2f}**")
                else:
                    st.image(processed_img, caption="Sin procesamiento adicional", use_column_width=True, clamp=True)
        else:
            st.subheader(f"🖼️ Imagen - Banda {band}")
            if binarize:
                st.image(processed_img, caption=f"Banda {band} binarizada", use_column_width=True, clamp=True)
                if threshold_value is not None:
                    st.info(f"🎯 Umbral automático (Otsu): **{threshold_value:.2f}**")
            else:
                st.image(original_band, caption=f"Banda {band}", use_column_width=True, clamp=True)
        
        st.divider()
        
        # Histograma
        st.subheader("📊 Análisis del Histograma")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = create_histogram(original_band, bins_count, f"Histograma - Banda {band}")
            st.pyplot(fig)
            plt.close(fig)  # Liberar memoria
        
        with col2:
            if show_statistics:
                stats = calculate_statistics(original_band)
                st.subheader("📈 Estadísticas")
                
                for key, value in stats.items():
                    if isinstance(value, float):
                        st.metric(key, f"{value:.2f}")
                    else:
                        st.metric(key, f"{value:,}")
        
        # Tabla de estadísticas completa
        if show_statistics:
            st.subheader("📋 Tabla de Estadísticas Detalladas")
            stats_df = pd.DataFrame(list(calculate_statistics(original_band).items()), 
                                  columns=['Estadística', 'Valor'])
            stats_df['Valor'] = stats_df['Valor'].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else f"{x:,}")
            st.dataframe(stats_df, use_container_width=True)
        
        # Información adicional para archivos GeoTIFF
        if metadata is not None:
            with st.expander("🗺️ Metadatos Geoespaciales"):
                st.json(dict(metadata))
    
    else:
        # Pantalla de bienvenida
        st.info("👆 Sube una imagen usando el panel lateral para comenzar el análisis")
        
        st.markdown("""
        ## 🚀 Características principales:
        
        - 📁 **Carga múltiples formatos**: JPG, PNG, TIFF/GeoTIFF
        - 🎨 **Análisis multiespectral**: Visualiza diferentes bandas espectrales
        - 🔲 **Binarización automática**: Usa el método de Otsu para umbralización
        - 📊 **Histogramas interactivos**: Análisis estadístico completo
        - 📈 **Estadísticas detalladas**: Media, desviación estándar, rango, etc.
        - 🗺️ **Soporte GeoTIFF**: Información geoespacial incluida
        - 🔍 **Comparación visual**: Vista lado a lado de imágenes
        
        ## 📋 Instrucciones de uso:
        
        1. **Sube tu imagen** usando el botón "Subir Imagen" en el panel lateral
        2. **Selecciona la banda** espectral que deseas analizar
        3. **Activa la binarización** si necesitas segmentación automática
        4. **Ajusta los parámetros** del histograma según tus necesidades
        5. **Analiza los resultados** usando las estadísticas y visualizaciones
        """)

if __name__ == "__main__":
    main()
