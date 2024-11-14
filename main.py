# main.py

# Primero las importaciones de streamlit
import streamlit as st

# Configuración de la página DEBE SER LA PRIMERA LLAMADA A STREAMLIT
st.set_page_config(layout="wide", page_title="Dashboard Redes Sociales")

# Resto de importaciones
import pandas as pd
from PIL import Image
from datetime import datetime
import sys

# Import visualization modules
from visualizations.facebook_viz import show_facebook_metrics
from visualizations.instagram_viz import show_instagram_metrics
#from visualizations.youtube_viz import show_youtube_metrics
#from visualizations.twitter_viz import show_twitter_metrics
from visualizations.linkedin_viz import *
from visualizations.linkedin_viz_2 import *
sys.path.append('.')

def main():
    # Header
    col1, col2 = st.columns([1, 5])
    
    #with col1:
    #    image = Image.open("logo.png")
    #    st.image(image, width=150)
    
    with col2:
        st.title("Dashboard de Análisis de Redes Sociales")
        st.markdown("### Análisis de Métricas por Red Social")

    # Sidebar para selección de red social
    option = st.sidebar.selectbox(
        'Seleccione una Red Social',
        ["Facebook", "Instagram", "YouTube", "X (Twitter)", "LinkedIn","LinkedIn II"]
    )

    # Mostrar el dashboard correspondiente
    if option == "Facebook":
        show_facebook_metrics()
    elif option == "Instagram":
        show_instagram_metrics()
    elif option == "YouTube":
        show_youtube_metrics()
    elif option == "X (Twitter)":
        show_twitter_metrics()
    elif option == "LinkedIn":
        show_linkedin_metrics()
    elif option == "LinkedIn II":
        show_linkedin_metrics_2()

    # Footer
    st.markdown("---")
    st.markdown("**Desarrollado por:** Andres Fernando Gomez Rojas")

if __name__ == "__main__":
    main()