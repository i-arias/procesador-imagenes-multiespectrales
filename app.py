import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Configuración de la página
st.set_page_config(
    page_title="Vision - Procesamiento de Imágenes",
    page_icon="👁️",
    layout="wide"
)

st.title("🔍 Vision - Procesamiento de Imágenes")
st.markdown("---")

# Función para cargar imagen
def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

# Función para procesar imagen
def process_image(img, band, binarize_otsu, global_bins):
    # Convertir a escala de grises si es necesario
    if len(img.shape) == 3:
        if band == 1:
            processed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            processed_img = img[:, :, band-1]  # Seleccionar banda específica
    else:
        processed_img = img
    
    # Aplicar binarización Otsu si está activada
    if binarize_otsu:
        _, processed_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return processed_img

# Función para generar histograma
def generate_histogram(img, bins):
    hist, bin_edges = np.histogram(img.flatten(), bins=bins, range=(0, 256))
    return hist, bin_edges

# Sidebar para controles
st.sidebar.header("⚙️ Controles")

# Upload de imagen
uploaded_file = st.sidebar.file_uploader(
    "Selecciona una imagen",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help="Formatos soportados: JPG, PNG, BMP"
)

if uploaded_file is not None:
    # Cargar imagen
    original_img = load_image(uploaded_file)
    
    # Controles interactivos
    st.sidebar.markdown("### 🎛️ Parámetros de procesamiento")
    
    # Slider para selección de banda
    max_bands = 1 if len(original_img.shape) == 2 else original_img.shape[2]
    band = st.sidebar.slider(
        "Banda",
        min_value=1,
        max_value=max(max_bands, 1),
        value=1,
        help="Selecciona la banda a procesar"
    )
    
    # Checkbox para binarización Otsu
    binarize_otsu = st.sidebar.checkbox(
        "Binarizar (Otsu)",
        value=False,
        help="Aplica binarización automática usando el método de Otsu"
    )
    
    # Slider para bins del histograma
    global_bins = st.sidebar.slider(
        "Bins global",
        min_value=32,
        max_value=1024,
        value=256,
        step=32,
        help="Número de bins para el histograma"
    )
    
    # Procesar imagen
    processed_img = process_image(original_img, band, binarize_otsu, global_bins)
    
    # Layout de columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Imagen Original")
        st.image(original_img, caption=f"Archivo: {uploaded_file.name}", use_container_width=True)
        
        # Información de la imagen
        st.info(f"""
        **Información de la imagen:**
        - Dimensiones: {original_img.shape[:2]}
        - Tipo: {original_img.dtype}
        - Bandas: {'3 (RGB)' if len(original_img.shape) == 3 else '1 (Escala de grises)'}
        - Banda actual: {band}
        """)
    
    with col2:
        st.subheader("🔄 Imagen Procesada")
        
        # Mostrar imagen procesada
        if binarize_otsu:
            st.image(processed_img, caption="Imagen binarizada (Otsu)", use_container_width=True, clamp=True)
        else:
            st.image(processed_img, caption=f"Banda {band}", use_container_width=True, clamp=True)
    
    # Histograma
    st.markdown("---")
    st.subheader("📊 Análisis del Histograma")
    
    # Generar histograma
    hist, bin_edges = generate_histogram(processed_img, global_bins)
    
    # Crear gráfico del histograma
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Intensidad de píxel')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Histograma - {global_bins} bins')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Estadísticas
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        st.metric("📈 Valor máximo", f"{np.max(processed_img)}")
    with col4:
        st.metric("📉 Valor mínimo", f"{np.min(processed_img)}")
    with col5:
        st.metric("📊 Media", f"{np.mean(processed_img):.2f}")
    with col6:
        st.metric("📏 Desv. Estándar", f"{np.std(processed_img):.2f}")
    
    # Descargar imagen procesada
    st.markdown("---")
    st.subheader("💾 Descarga")
    
    if st.button("Descargar imagen procesada"):
        # Convertir a PIL Image para descarga
        if processed_img.dtype != np.uint8:
            download_img = (processed_img * 255).astype(np.uint8)
        else:
            download_img = processed_img
            
        pil_img = Image.fromarray(download_img)
        
        # Crear buffer
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        st.download_button(
            label="📥 Descargar PNG",
            data=img_buffer,
            file_name=f"processed_{uploaded_file.name.split('.')[0]}.png",
            mime="image/png"
        )

else:
    # Mensaje cuando no hay imagen
    st.info("👆 Sube una imagen usando el panel lateral para comenzar el análisis.")
    
    # Ejemplo de imágenes que se pueden procesar
    st.markdown("### 🖼️ Tipos de procesamiento disponibles:")
    st.markdown("""
    - **Selección de bandas**: Para imágenes RGB o multicanal
    - **Binarización Otsu**: Segmentación automática
    - **Análisis de histograma**: Con bins configurables
    - **Estadísticas**: Valores máx/mín, media, desviación estándar
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Desarrollado con ❤️ usando Streamlit | Vision App v1.0
    </div>
    """, 
    unsafe_allow_html=True
)
