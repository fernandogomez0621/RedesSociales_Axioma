import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Diccionario de nombres de páginas
PAGE_NAMES = {
    '10240616': 'IAlimentos',
    '11830997': 'Manufactura-Latam.com',
    '11831952': 'El Empaque + Conversión',
    '11831955': 'Tecnología del Plástico',
    '10201761': 'La Barra', 
    '10201763': 'Fierros',
    '10224894': 'En-Obra.com',
    '11830168': 'ElHospital.com'
}

def load_linkedin_data():
    """Cargar y combinar datos de LinkedIn desde ambos archivos Excel"""
    try:
        # Definir los tipos de datos para la carga
        dtype_dict = {
            'PageID': str
        }
        
        # Cargar primer archivo
        st.write("Cargando Parte I...")
        df1 = pd.read_excel(
            "redes_sociales/linkedin/Share_Statistics_By_Date_Part_I.xlsx",
            dtype=dtype_dict
        )
        st.write(f"Parte I cargada. Dimensiones: {df1.shape}")
        
        # Cargar segundo archivo
        st.write("Cargando Parte II...")
        df2 = pd.read_excel(
            "redes_sociales/linkedin/Share_Statistics_By_Date_Part_II.xlsx",
            dtype=dtype_dict
        )
        st.write(f"Parte II cargada. Dimensiones: {df2.shape}")
        
        # Concatenar
        df = pd.concat([df1, df2], ignore_index=True)
        
        # Convertir la columna de fecha
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Agregar el nombre de la página
        df['Page Name'] = df['PageID'].astype(str).map(PAGE_NAMES)
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

def calculate_monthly_metrics(df, start_date=None, end_date=None):
    """Calcular métricas mensuales y sus variaciones"""
    # Filtrar por fecha si se especifica
    if start_date and end_date:
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Excluir métricas que contienen 'Unique'
    non_unique_columns = [col for col in df.columns if 'Unique' not in col]
    metrics_to_analyze = [col for col in non_unique_columns if col not in ['PageID', 'Date', 'Page Name']]
    
    # Agrupar por página y mes
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_metrics = df.groupby(['Page Name', 'Month'])[metrics_to_analyze].sum().reset_index()
    
    return monthly_metrics

def show_metric_comparison(monthly_metrics, metric_name):
    """Mostrar comparación mensual para una métrica específica"""
    st.subheader(f"Análisis mensual - {metric_name}")
    
    # Pivotear los datos para mostrar meses en columnas
    pivot_table = monthly_metrics.pivot(index='Page Name', columns='Month', values=metric_name)
    
    # Agregar columna de totales
    pivot_table['Total'] = pivot_table.sum(axis=1)
    
    # Calcular variaciones porcentuales
    variations_pct = pd.DataFrame(index=pivot_table.index)
    # Calcular diferencias absolutas
    variations_abs = pd.DataFrame(index=pivot_table.index)
    
    for i in range(1, len(pivot_table.columns) - 1):  # -1 para excluir la columna Total
        prev_month = pivot_table.columns[i-1]
        curr_month = pivot_table.columns[i]
        # Variación porcentual
        pct_change = (pivot_table[curr_month] - pivot_table[prev_month]) / pivot_table[prev_month] * 100
        variations_pct[f'Var % {curr_month}'] = pct_change.round(2)
        # Diferencia absoluta
        abs_change = pivot_table[curr_month] - pivot_table[prev_month]
        variations_abs[f'Dif {curr_month}'] = abs_change
    
    # Mostrar tablas con formato
    st.write("Valores mensuales y totales:")
    
    # Reemplazar NaN con 0 para evitar errores en la aplicación del gradiente
    pivot_table.fillna(0, inplace=True)
    
    # Función para aplicar el gradiente por fila
    def background_gradient_by_row(s, m, M, cmap='RdYlGn'):
        import numpy as np
        rng = M - m
        norm = (s - m) / (rng if rng != 0 else 1)
        normed = norm.clip(0, 1)  # Asegurar que los valores estén entre 0 y 1
        colors = px.colors.sample_colorscale(cmap, normed)
        return [f'background-color: {color}' for color in colors]
    
    # Aplicar estilo solo a las columnas numéricas que no sean Total
    numeric_cols = pivot_table.columns[:-1]  # Excluir la columna Total
    
    # Aplicar el estilo fila por fila
    styled_table = pivot_table.style.apply(
        lambda x: background_gradient_by_row(
            x[numeric_cols],
            x[numeric_cols].min(),
            x[numeric_cols].max()
        ),
        subset=numeric_cols,
        axis=1
    )
    
    # Formatear la columna Total en negrita y los números con separador de miles
    styled_table = styled_table.format('{:,.0f}').applymap(
        lambda x: 'font-weight: bold',
        subset=['Total']
    )
    
    st.dataframe(styled_table)

    st.write("Variaciones porcentuales (%):")
    if not variations_pct.empty:
        st.dataframe(
            variations_pct.style.background_gradient(
                cmap='RdYlGn',
                vmin=-100,
                vmax=100
            ).format("{:.2f}%")  # Formatear con dos decimales y símbolo %
        )
    
    st.write("Diferencias absolutas respecto al mes anterior:")
    if not variations_abs.empty:
        st.dataframe(
            variations_abs.style.background_gradient(
                cmap='RdYlGn',
                vmin=variations_abs.min().min(),
                vmax=variations_abs.max().max()
            ).format("{:,.0f}")  # Formatear sin decimales y con separador de miles
        )

def create_comparison_charts(monthly_metrics, metric_name):
    """Crear gráficos comparativos para una métrica"""
    # Convertir Month a string para plotly
    monthly_metrics['Month'] = monthly_metrics['Month'].astype(str)
    
    # Calcular totales por página
    totals = monthly_metrics.groupby('Page Name')[metric_name].sum().reset_index()
    
    # Gráfico individual por empresa
    fig_individual = make_subplots(
        rows=len(PAGE_NAMES)//2 + len(PAGE_NAMES)%2,
        cols=2,
        subplot_titles=[f"{page} (Total: {totals[totals['Page Name'] == page][metric_name].values[0]:,.0f})" 
                       for page in PAGE_NAMES.values()]
    )
    
    row = 1
    col = 1
    for page_name in PAGE_NAMES.values():
        page_data = monthly_metrics[monthly_metrics['Page Name'] == page_name]
        
        fig_individual.add_trace(
            go.Bar(
                x=page_data['Month'],
                y=page_data[metric_name],
                name=page_name,
                text=page_data[metric_name].round(0),  # Agregar etiquetas
                textposition='auto',
            ),
            row=row,
            col=col
        )
        
        if col == 2:
            row += 1
            col = 1
        else:
            col += 1
    
    fig_individual.update_layout(
        height=300*len(PAGE_NAMES)//2,
        showlegend=False,
        title_text=f"Métricas por Página - {metric_name}"
    )
    
    # Gráfico comparativo de todas las empresas
    fig_combined = px.bar(
        monthly_metrics,
        x='Month',
        y=metric_name,
        color='Page Name',
        barmode='group',
        title=f'Comparación de {metric_name} por Empresa',
        text=metric_name  # Agregar etiquetas
    )
    
    fig_combined.update_traces(texttemplate='%{text:.0f}', textposition='auto')
    
    return fig_individual, fig_combined

def show_linkedin_metrics():
    """Función principal para mostrar el dashboard de LinkedIn"""
    st.header("Análisis de Métricas de LinkedIn")
    
    # Cargar datos
    df = load_linkedin_data()
    if df is None:
        return
    
    # Filtros de fecha
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Fecha inicial",
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date(),
            value=df['Date'].min().date()
        )
    with col2:
        end_date = st.date_input(
            "Fecha final",
            min_value=df['Date'].min().date(),
            max_value=df['Date'].max().date(),
            value=df['Date'].max().date()
        )
    
    # Calcular métricas mensuales
    monthly_metrics = calculate_monthly_metrics(
        df,
        pd.to_datetime(start_date),
        pd.to_datetime(end_date)
    )
    
    # Selector de métrica
    metric_options = [col for col in df.columns if 'Unique' not in col and col not in ['PageID', 'Date', 'Month', 'Page Name']]
    selected_metric = st.selectbox("Seleccione una métrica para analizar:", metric_options)
    
    # Mostrar análisis de la métrica seleccionada
    show_metric_comparison(monthly_metrics, selected_metric)
    
    # Mostrar gráficos
    st.subheader("Visualización de Métricas")
    
    fig_individual, fig_combined = create_comparison_charts(monthly_metrics, selected_metric)
    
    st.plotly_chart(fig_individual, use_container_width=True)
    st.plotly_chart(fig_combined, use_container_width=True)
