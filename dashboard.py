import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import re  # Add re import for regex
import os
import joblib
import json
from datetime import datetime, timedelta
import numpy as np
from catboost import Pool
import io

# Import model functions
from modelo import prepare_final_dataset_improved

# Configuración estilo gráfico general (sin líneas de fondo)
sns.set_style("white")
sns.set_context("talk", font_scale=0.9)
plt.rcParams.update({
    "axes.edgecolor": "#E0E0E0",
    "axes.linewidth": 0.8,
    "axes.titlesize": 14,
    "axes.titleweight": 'bold',
    "axes.labelcolor": "#333333",
    "axes.labelsize": 12,
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "sans-serif"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.autolayout": True,
    "figure.constrained_layout.use": True
})

# Paletas de colores personalizadas
COLOR_GRADIENT = ["#e6f3ff", "#cce7ff", "#99cfff", "#66b8ff", "#33a0ff", "#0088ff", "#006acc", "#004d99", "#003366"]
TEMPORADA_COLORS = ["#e6f3ff", "#99ccff", "#4d94ff", "#0066cc", "#004d99", "#003366", "#001a33", "#000d1a", "#000000"]
COLOR_GRADIENT_WARM = ["#fff5e6", "#ffebcc", "#ffd699", "#ffc266", "#ffad33", "#ff9900", "#cc7a00", "#995c00", "#663d00"]
COLOR_GRADIENT_GREEN = ["#e6ffe6", "#ccffcc", "#99ff99", "#66ff66", "#33ff33", "#00ff00", "#00cc00", "#009900", "#006600"]

TIENDAS_EXTRANJERAS = [
    "I301COINBERGAMO(TRUCCO)", "I302COINVARESE(TRUCCO)", "I303COINBARICASAMASSIMA(TRUCCO)",
    "I304COINMILANO5GIORNATE(TRUCCO)", "I305COINROMACINECITTA(TRUCCO)", "I306COINGENOVA(TRUCCO)",
    "I309COINSASSARI(TRUCCO)", "I314COINCATANIA(TRUCCO)", "I315COINCAGLIARI(TRUCCO)",
    "I316COINLECCE(TRUCCO)", "I317COINMILANOCANTORE(TRUCCO)", "I318COINMESTRE(TRUCCO)",
    "I319COINPADOVA(TRUCCO)", "I320COINFIRENZE(TRUCCO)", "I321COINROMASANGIOVANNI(TRUCCO)",
    "TRUCCOONLINEB2C"
]

COL_ONLINE = '#2ca02c'   # verde fuerte
COL_OTRAS = '#ff7f0e'    # naranja

def custom_sort_key(talla):
    """
    Clave de ordenación personalizada para tallas.
    Prioriza: 1. Tallas numéricas, 2. Tallas de letra estándar, 3. Tallas únicas, 4. Resto.
    """
    talla_str = str(talla).upper()
    
    # Prioridad 1: Tallas numéricas (e.g., '36', '38')
    if talla_str.isdigit():
        return (0, int(talla_str))
    
    # Prioridad 2: Tallas de letra estándar
    size_order = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
    if talla_str in size_order:
        return (1, size_order.index(talla_str))
        
    # Prioridad 3: Tallas únicas
    if talla_str in ['U', 'ÚNICA', 'UNICA', 'TU']:
        return (2, talla_str)
        
    # Prioridad 4: Resto, ordenado alfabéticamente
    return (3, talla_str)

def setup_streamlit_styles():
    """Configurar estilos de Streamlit"""
    st.markdown("""
    <style>
    .dashboard-container {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .kpi-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        width: 100%;
        margin-top: 0;
        padding-top: 0;
    }
    .kpi-row {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        flex-wrap: nowrap;
        width: 100%;
    }
    .kpi-group {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        margin-top: 0;
        background-color: white;
        width: 100%;
    }
    .kpi-group-title {
        color: #666666;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 10px;
        margin-top: 0;
        padding-bottom: 5px;
        border-bottom: 1px solid #e5e7eb;
    }
    .kpi-item {
        flex: 1;
        text-align: center;
        padding: 15px;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background-color: white;
        min-width: 150px;
    }
    .small-font {
        color: #666666;
        font-size: 14px;
        margin-bottom: 5px;
        margin-top: 0;
    }
    .metric-value {
        color: #111827;
        font-size: 24px;
        font-weight: bold;
        margin: 0;
    }
    .section-title {
        color: #111827;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 20px;
        margin-top: 0;
        line-height: 1.2;
    }
    .viz-title {
        color: #111827;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 15px;
        margin-top: 0;
        line-height: 1.2;
    }
    .viz-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        margin-top: 20px;
    }
    .viz-container {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        background-color: white;
        height: 100%;
    }
    div.block-container {
        padding-top: 0;
        margin-top: 0;
    }
    div.stMarkdown {
        margin-top: 0;
        padding-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)

def viz_title(text):
    """Función unificada para títulos de visualizaciones"""
    st.markdown(f'<h3 class="viz-title">{text}</h3>', unsafe_allow_html=True)

def titulo(text):
    st.markdown(f"<h4 style='text-align:left;color:#666666;margin:0;padding:0;font-size:20px;font-weight:bold;'>{text}</h4>", unsafe_allow_html=True)

def subtitulo(text):
    st.markdown(f"<h5 style='text-align:left;color:#666666;margin:0;padding:0;font-size:22px;font-weight:bold;'>{text}</h5>", unsafe_allow_html=True)

def aplicar_filtros(df_ventas, df_traspasos=None):
    if not pd.api.types.is_datetime64_any_dtype(df_ventas['Fecha Documento']):
        df_ventas['Fecha Documento'] = pd.to_datetime(df_ventas['Fecha Documento'], format='%d/%m/%Y', errors='coerce')
    fecha_min, fecha_max = df_ventas['Fecha Documento'].min(), df_ventas['Fecha Documento'].max()

    fecha_inicio, fecha_fin = st.sidebar.date_input(
        "Rango de fechas",
        [fecha_min, fecha_max],
        min_value=fecha_min,
        max_value=fecha_max
    )

    if fecha_inicio > fecha_fin:
        st.sidebar.error("La fecha de inicio debe ser anterior a la fecha de fin.")
        if df_traspasos is not None:
            return df_ventas.iloc[0:0], df_traspasos.iloc[0:0], False, []
        return df_ventas.iloc[0:0], False, []

    df_ventas_filtrado = df_ventas[(df_ventas['Fecha Documento'] >= pd.to_datetime(fecha_inicio)) &
                     (df_ventas['Fecha Documento'] <= pd.to_datetime(fecha_fin))]
    tiendas = sorted(df_ventas_filtrado['NombreTPV'].dropna().unique())
    modo_tienda = st.sidebar.selectbox(
        "Modo selección tiendas",
        ["Todas las tiendas", "Seleccionar tiendas específicas"]
    )
    if modo_tienda == "Todas las tiendas":
        tienda_seleccionada = tiendas
        tiendas_especificas = False
    else:
        tienda_seleccionada = st.sidebar.multiselect(
            "Selecciona tienda(s)",
            options=tiendas
        )
        if not tienda_seleccionada:
            st.sidebar.warning("Selecciona al menos una tienda para mostrar datos.")
            if df_traspasos is not None:
                return df_ventas.iloc[0:0], df_traspasos.iloc[0:0], False, []
            return df_ventas.iloc[0:0], False, []
        tiendas_especificas = True
    
    df_ventas_filtrado = df_ventas_filtrado[df_ventas_filtrado['NombreTPV'].isin(tienda_seleccionada)]
    
    # Aplicar filtro de tienda a traspasos si se proporciona
    if df_traspasos is not None:
        df_traspasos_filtrado = df_traspasos.copy()
        # Asegurar que la columna Tienda existe en traspasos
        if 'Tienda' in df_traspasos_filtrado.columns:
            df_traspasos_filtrado = df_traspasos_filtrado[df_traspasos_filtrado['Tienda'].isin(tienda_seleccionada)]
        return df_ventas_filtrado, df_traspasos_filtrado, tiendas_especificas, tienda_seleccionada
    
    return df_ventas_filtrado, tiendas_especificas, tienda_seleccionada



def create_resizable_chart(chart_key, chart_function):
    """
    Crea un contenedor para el gráfico con funcionalidad de redimensionamiento
    """
    col1, col2 = st.columns([4, 1])
    with col1:
        size = st.select_slider(
            f'Ajustar tamaño del gráfico {chart_key}',
            options=['Pequeño', 'Mediano', 'Grande', 'Extra Grande'],
            value='Mediano',
            key=f'size_{chart_key}'
        )
    
    sizes = {
        'Pequeño': 300,
        'Mediano': 500,
        'Grande': 700,
        'Extra Grande': 900
    }
    
    height = sizes[size]
    
    st.markdown(f'<div class="chart-container" style="height: {height}px;">', unsafe_allow_html=True)
    chart_function(height)
    st.markdown('</div>', unsafe_allow_html=True)

def plot_bar(df, x, y, title, palette='Greens', rotate_x=30, color=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    if color:
        sns.barplot(x=x, y=y, data=df, color=color, ax=ax)
    else:
        # Normalizar los valores para el degradado
        norm_values = (df[y] - df[y].min()) / (df[y].max() - df[y].min())
        colors = [COLOR_GRADIENT[int(v * (len(COLOR_GRADIENT)-1))] if not pd.isna(v) else COLOR_GRADIENT[0] for v in norm_values]
        
        sns.barplot(x=x, y=y, data=df, palette=colors, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', color="#111827", loc="left", pad=0)
    ax.set_xlabel(x, fontsize=13)
    ax.set_ylabel(y, fontsize=13)
    plt.xticks(rotation=rotate_x, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    ax.grid(False)
    ax.set_axisbelow(True)
    sns.despine()
    
    # Ajustar valores sobre las barras
    for bar in ax.patches:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.01 * value),
            f'{int(value)}',
            ha='center',
            va='bottom',
            fontsize=10,
            color='#333'
        )
    
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)

def viz_container(title, render_function):
    """Contenedor para visualizaciones"""
    st.markdown('<div class="viz-container">', unsafe_allow_html=True)
    viz_title(title)
    render_function()
    st.markdown('</div>', unsafe_allow_html=True)

def mostrar_dashboard(df_productos, df_traspasos, df_ventas, seccion):
    setup_streamlit_styles()
    
    # Use cached preprocessing for better performance
    df_ventas = preprocess_ventas_data(df_ventas)
    df_productos = df_productos.copy()
    
    # Asegurar columna 'Talla' en traspasos si no existe
    if "Talla" not in df_traspasos.columns:
        for col in ["Talla", "Size"]:
            if col in df_traspasos.columns:
                df_traspasos["Talla"] = df_traspasos[col]
                break

    # Calcular ranking completo de todas las tiendas ANTES de aplicar filtros
    ventas_por_tienda_completo = calculate_store_rankings(df_ventas)
    
    # Aplicar filtros
    df_ventas, df_traspasos_filtrado, tiendas_especificas, tienda_seleccionada = aplicar_filtros(df_ventas, df_traspasos)
    if df_ventas.empty:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        return

    if seccion == "Resumen General":
        try:
            # Calcular KPIs
            total_ventas_dinero = df_ventas['Ventas Dinero'].sum()
            total_familias = df_ventas['Familia'].nunique()
            
            # Calcular Total Devoluciones (monetary amount of negative quantities)
            devoluciones = df_ventas[df_ventas['Cantidad'] < 0].copy()
            total_devoluciones_dinero = abs(devoluciones['Ventas Dinero'].sum())  # Use abs() to show positive value
            
            # Separar tiendas físicas y online
            ventas_fisicas = df_ventas[~df_ventas['Es_Online']]
            ventas_online = df_ventas[df_ventas['Es_Online']]
            
            # Calcular KPIs por tipo de tienda
            ventas_fisicas_dinero = ventas_fisicas['Ventas Dinero'].sum()
            ventas_online_dinero = ventas_online['Ventas Dinero'].sum()
            tiendas_fisicas = ventas_fisicas['NombreTPV'].nunique()
            tiendas_online = ventas_online['NombreTPV'].nunique()

            # KPIs Generales en una sola fila
            st.markdown("""
                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                    <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                        KPIs Generales
                    </div>
                    <div style="display: flex; justify-content: space-between; gap: 15px;">
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Total Ventas Netas</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:,.2f}€</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Total Devoluciones</p>
                            <p style="color: #dc2626; font-size: 24px; font-weight: bold; margin: 0;">{:,.2f}€</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Número de Familias</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{}</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Total Tiendas</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{}</p>
                        </div>
                    </div>
                </div>
            """.format(total_ventas_dinero, total_devoluciones_dinero, total_familias, tiendas_fisicas + tiendas_online), unsafe_allow_html=True)
            
            # KPIs por Tipo de Tienda en una sola fila
            st.markdown("""
                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                    <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                        KPIs por Tipo de Tienda
                    </div>
                    <div style="display: flex; justify-content: space-between; gap: 15px;">
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tiendas Físicas</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{}</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Ventas Físicas</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:,.2f}€</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tiendas Online</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{}</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Ventas Online</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:,.2f}€</p>
                        </div>
                    </div>
                </div>
            """.format(tiendas_fisicas, ventas_fisicas_dinero, tiendas_online, ventas_online_dinero), unsafe_allow_html=True)
            
            # ===== KPIs de Rotación de Stock =====
            st.markdown("### 📊 **KPIs de Rotación de Stock (V2025 e I2025)**")
            
            # Filtrar ventas solo para V2025 e I2025
            ventas_rotacion = df_ventas[df_ventas['Temporada'].isin(['V2025', 'I2025'])].copy()
            
            # Preparar datos de entrada en almacén y traspasos para el cálculo de rotación
            if not df_productos.empty and 'Fecha REAL entrada en almacén' in df_productos.columns:
                # Crear ACT_14 en df_productos para matching
                df_productos_rotacion = df_productos.copy()
                df_productos_rotacion['ACT_14'] = df_productos_rotacion['ACT'].astype(str).str[:14]
                df_productos_rotacion['Fecha REAL entrada en almacén'] = pd.to_datetime(
                    df_productos_rotacion['Fecha REAL entrada en almacén'], errors='coerce')
                
                # Preparar traspasos
                df_traspasos_rotacion = df_traspasos.copy()
                df_traspasos_rotacion['ACT_14'] = df_traspasos_rotacion['ACT'].astype(str).str[:14]
                df_traspasos_rotacion['Fecha Enviado'] = pd.to_datetime(
                    df_traspasos_rotacion['Fecha Enviado'], errors='coerce')
                
                # Preparar ventas
                ventas_rotacion['ACT_14'] = ventas_rotacion['ACT'].astype(str).str[:14]
                ventas_rotacion['Fecha Documento'] = pd.to_datetime(
                    ventas_rotacion['Fecha Documento'], errors='coerce')
                
                # OPTIMIZACIÓN: Usar merge en lugar de loops anidados
                # 1. Merge ventas con entrada en almacén
                ventas_con_entrada = ventas_rotacion.merge(
                    df_productos_rotacion[['ACT_14', 'Talla', 'Fecha REAL entrada en almacén']],
                    on=['ACT_14', 'Talla'],
                    how='inner'
                )
                
                # 2. Merge con traspasos
                rotacion_completa = ventas_con_entrada.merge(
                    df_traspasos_rotacion[['ACT_14', 'Talla', 'Tienda', 'Fecha Enviado']],
                    left_on=['ACT_14', 'Talla', 'NombreTPV'],
                    right_on=['ACT_14', 'Talla', 'Tienda'],
                    how='inner'
                )
                
                # 3. Calcular días de rotación
                rotacion_completa['Dias_Rotacion'] = (
                    rotacion_completa['Fecha Documento'] - rotacion_completa['Fecha REAL entrada en almacén']
                ).dt.days
                
                # Filtrar solo días positivos
                rotacion_completa = rotacion_completa[rotacion_completa['Dias_Rotacion'] >= 0]
                
                if not rotacion_completa.empty:
                    # Calcular rotación por tienda
                    rotacion_por_tienda = rotacion_completa.groupby('NombreTPV').agg({
                        'Dias_Rotacion': ['mean', 'count']
                    }).reset_index()
                    rotacion_por_tienda.columns = ['Tienda', 'Dias_Promedio', 'Productos_Con_Rotacion']
                    
                    # Calcular rotación por producto
                    rotacion_por_producto = rotacion_completa.groupby(['ACT_14', 'Descripción Familia']).agg({
                        'Dias_Rotacion': ['mean', 'count']
                    }).reset_index()
                    rotacion_por_producto.columns = ['ACT', 'Producto', 'Dias_Promedio', 'Ventas_Con_Rotacion']
                    
                    # Calcular KPIs
                    tienda_mayor_rotacion = "Sin datos"
                    tienda_mayor_rotacion_dias = 0
                    tienda_menor_rotacion = "Sin datos"
                    tienda_menor_rotacion_dias = 0
                    producto_mayor_rotacion = "Sin datos"
                    producto_mayor_rotacion_dias = 0
                    producto_menor_rotacion = "Sin datos"
                    producto_menor_rotacion_dias = 0
                    
                    if not rotacion_por_tienda.empty:
                        # Tienda con mayor rotación (menos días)
                        idx_mayor = rotacion_por_tienda['Dias_Promedio'].idxmin()
                        tienda_mayor_rotacion = rotacion_por_tienda.loc[idx_mayor, 'Tienda']
                        tienda_mayor_rotacion_dias = rotacion_por_tienda.loc[idx_mayor, 'Dias_Promedio']
                        
                        # Tienda con menor rotación (más días)
                        idx_menor = rotacion_por_tienda['Dias_Promedio'].idxmax()
                        tienda_menor_rotacion = rotacion_por_tienda.loc[idx_menor, 'Tienda']
                        tienda_menor_rotacion_dias = rotacion_por_tienda.loc[idx_menor, 'Dias_Promedio']
                    
                    if not rotacion_por_producto.empty:
                        # Producto con mayor rotación (menos días)
                        idx_mayor = rotacion_por_producto['Dias_Promedio'].idxmin()
                        producto_mayor_rotacion = rotacion_por_producto.loc[idx_mayor, 'Producto']
                        producto_mayor_rotacion_dias = rotacion_por_producto.loc[idx_mayor, 'Dias_Promedio']
                        
                        # Producto con menor rotación (más días)
                        idx_menor = rotacion_por_producto['Dias_Promedio'].idxmax()
                        producto_menor_rotacion = rotacion_por_producto.loc[idx_menor, 'Producto']
                        producto_menor_rotacion_dias = rotacion_por_producto.loc[idx_menor, 'Dias_Promedio']
                    
                    # Mostrar KPIs de rotación
                    st.markdown("""
                        <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                            <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                                KPIs de Rotación de Stock
                            </div>
                            <div style="display: flex; justify-content: space-between; gap: 15px;">
                                <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                                    <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tienda Mayor Rotación</p>
                                    <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                                    <p style="color: #059669; font-size: 12px; margin: 0;">{:.1f} días promedio</p>
                                </div>
                                <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                                    <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tienda Menor Rotación</p>
                                    <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                                    <p style="color: #dc2626; font-size: 12px; margin: 0;">{:.1f} días promedio</p>
                                </div>
                                <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                                    <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Producto Mayor Rotación</p>
                                    <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                                    <p style="color: #059669; font-size: 12px; margin: 0;">{:.1f} días promedio</p>
                                </div>
                                <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                                    <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Producto Menor Rotación</p>
                                    <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                                    <p style="color: #dc2626; font-size: 12px; margin: 0;">{:.1f} días promedio</p>
                                </div>
                            </div>
                        </div>
                    """.format(
                        tienda_mayor_rotacion, tienda_mayor_rotacion_dias,
                        tienda_menor_rotacion, tienda_menor_rotacion_dias,
                        producto_mayor_rotacion, producto_mayor_rotacion_dias,
                        producto_menor_rotacion, producto_menor_rotacion_dias
                    ), unsafe_allow_html=True)
                    
                    # Mostrar estadísticas adicionales
                    st.info(f"📊 Análisis basado en {len(rotacion_completa)} productos con rotación calculada")
                else:
                    st.info("No se encontraron productos con datos completos de entrada, traspaso y venta para calcular rotación.")
            else:
                st.info("No hay datos de entrada en almacén disponibles para calcular rotación de stock.")

            # Col 1: Ventas por mes (centered)
            col1a, col1b, col1c = st.columns([1, 2, 1])
            
            with col1b:
                viz_title("Ventas Mensuales por Tipo de Tienda")
                ventas_mes_tipo = df_ventas.groupby(['Mes', 'Es_Online']).agg({
                    'Cantidad': 'sum',
                    'Ventas Dinero': 'sum'
                }).reset_index()
                
                ventas_mes_tipo['Tipo'] = ventas_mes_tipo['Es_Online'].map({True: 'Online', False: 'Física'})
                
                # Calculate dynamic width based on number of months
                num_months = len(ventas_mes_tipo['Mes'].unique())
                dynamic_width = max(800, num_months * 300)  # Minimum 800px, 300px per month for much wider graph
                
                fig = px.bar(ventas_mes_tipo, 
                            x='Mes', 
                            y='Cantidad', 
                            color='Tipo',
                            color_discrete_map={'Física': '#1e3a8a', 'Online': '#60a5fa'},
                            barmode='stack',
                            text='Cantidad',
                            height=400,
                            width=dynamic_width)
                
                fig.update_layout(
                    xaxis_title="Mes",
                    yaxis_title="Cantidad",
                    showlegend=True,
                    xaxis_tickangle=45,
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                fig.update_traces(
                    texttemplate='%{text:,.0f}', 
                    textposition='outside',
                    hovertemplate="Mes: %{x}<br>Cantidad: %{text:,.0f}<br>Ventas: %{customdata:,.2f}€<extra></extra>",
                    customdata=ventas_mes_tipo['Ventas Dinero'],
                    opacity=0.8
                )
                
                # Use HTML container with dynamic width
                st.markdown(f"""
                    <div style="width: {dynamic_width}px; max-width: 100%; overflow-x: auto;">
                        <div style="width: 100%;">
                """, unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=False)
                st.markdown("</div></div>", unsafe_allow_html=True)

            # Col 2,3: Ranking de tiendas
            col2, col3 = st.columns(2)
            
            if tiendas_especificas:
                # Mostrar tabla de ranking para las tiendas seleccionadas (full width)
                viz_title("Ranking de Tiendas Seleccionadas")
                
                # Filtrar solo las tiendas seleccionadas del ranking completo
                tiendas_ranking = ventas_por_tienda_completo[ventas_por_tienda_completo['Tienda'].isin(tienda_seleccionada)].copy()
                
                # Calcular la familia más vendida para cada tienda (cached)
                familias_por_tienda = calculate_family_rankings(df_ventas)
                
                # Obtener la familia top para cada tienda seleccionada
                familias_top = []
                for tienda in tiendas_ranking['Tienda']:
                    familias_tienda = familias_por_tienda[familias_por_tienda['NombreTPV'] == tienda]
                    if not familias_tienda.empty:
                        familia_top = familias_tienda.iloc[0]['Familia']
                        familias_top.append(familia_top)
                    else:
                        familias_top.append('Sin datos')
                
                tiendas_ranking['Familia Top'] = familias_top
                
                # Reordenar columnas
                tiendas_ranking = tiendas_ranking[['Tienda', 'Ranking', 'Unidades Vendidas', 'Ventas (€)', 'Familia Top']]
                
                # Mostrar tabla
                st.dataframe(
                    tiendas_ranking.style.format({
                        'Unidades Vendidas': '{:,.0f}',
                        'Ventas (€)': '{:,.2f}€'
                    }),
                    use_container_width=True
                )
            else:
                # Mostrar top 20 y bottom 20 como antes
                with col2:
                    # Top 20 tiendas con más ventas por ventas (€)
                    viz_title("Top 20 tiendas con más ventas")
                    top_20_tiendas = ventas_por_tienda_completo.head(20)
                    
                    fig = px.bar(
                        top_20_tiendas,
                        x='Tienda',
                        y='Ventas (€)',
                        color='Ventas (€)',
                        color_continuous_scale=COLOR_GRADIENT,
                        height=400,
                        labels={'Tienda': 'Tienda', 'Ventas (€)': 'Ventas (€)', 'Unidades Vendidas': 'Unidades'}
                    )
                    fig.update_layout(
                        xaxis_tickangle=45,
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    fig.update_traces(
                        texttemplate='%{y:,.2f}€',
                        textposition='outside',
                        hovertemplate="Tienda: %{x}<br>Ventas: %{y:,.2f}€<br>Unidades: %{customdata:,}<extra></extra>",
                        customdata=top_20_tiendas['Unidades Vendidas'],
                        opacity=0.8
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col3:
                    # Top 20 tiendas con menos ventas por ventas (€)
                    viz_title("Top 20 tiendas con menos ventas")
                    bottom_20_tiendas = ventas_por_tienda_completo.tail(20)
                    
                    fig = px.bar(
                        bottom_20_tiendas,
                        x='Tienda',
                        y='Ventas (€)',
                        color='Ventas (€)',
                        color_continuous_scale=COLOR_GRADIENT,
                        height=400,
                        labels={'Tienda': 'Tienda', 'Ventas (€)': 'Ventas (€)', 'Unidades Vendidas': 'Unidades'}
                    )
                    fig.update_layout(
                        xaxis_tickangle=45,
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    fig.update_traces(
                        texttemplate='%{y:,.2f}€',
                        textposition='outside',
                        hovertemplate="Tienda: %{x}<br>Ventas: %{y:,.2f}€<br>Unidades: %{customdata:,}<extra></extra>",
                        customdata=bottom_20_tiendas['Unidades Vendidas'],
                        opacity=0.8
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Col 4: Unidades Vendidas por Talla (centered)
            col4a, col4b, col4c = st.columns([1, 2, 1])
            
            with col4b:
                viz_title("Unidades Vendidas por Talla")
                
                familias = sorted(df_ventas['Familia'].unique())
                familia_seleccionada = st.selectbox("Selecciona una familia:", familias)
                
                # Filtrar df_ventas por familia seleccionada
                df_familia = df_ventas[df_ventas['Familia'] == familia_seleccionada].copy()

                if df_familia.empty:
                    st.warning("No hay datos de ventas para la familia seleccionada.")
                else:
                    # Agrupamos por Talla y Temporada
                    tallas_sumadas = (
                        df_familia.groupby(['Talla', 'Temporada'])['Cantidad']
                        .sum()
                        .reset_index()
                    )
                    
                    # Orden personalizado de tallas
                    tallas_presentes = df_familia['Talla'].dropna().unique()
                    tallas_orden = sorted(tallas_presentes, key=custom_sort_key)

                    # Gráfico de barras apiladas por Temporada
                    temporada_colors = get_temporada_colors(df_ventas)
                    fig = px.bar(
                        tallas_sumadas,
                        x='Talla',
                        y='Cantidad',
                        color='Temporada',
                        text='Cantidad',
                        category_orders={'Talla': tallas_orden},
                        color_discrete_map=temporada_colors,
                        height=450
                    )
                    
                    fig.update_layout(
                        xaxis_title="Talla",
                        yaxis_title="Unidades Vendidas",
                        barmode="stack",
                        margin=dict(t=30, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    
                    fig.update_traces(texttemplate='%{text:.0f}', textposition='inside', opacity=0.9)
                    st.plotly_chart(fig, use_container_width=True)



            # Col 5,6: Tablas por Temporada con layout dinámico
            # Preparar datos de entrada en almacén para las tablas por temporada
            # Agregar Descripción Familia a df_productos usando ACT codes de df_ventas
            df_productos_temp = df_productos.copy()
            

            
            # Crear ACT_14 (primeros 14 caracteres) en df_productos
            df_productos_temp['ACT_14'] = df_productos_temp['ACT'].astype(str).str[:14]
            
            # Crear mapeo de ACT a Descripción Familia desde df_ventas
            act_to_familia = df_ventas[['ACT', 'Descripción Familia']].drop_duplicates().set_index('ACT')['Descripción Familia'].to_dict()
            
            # Agregar Descripción Familia a df_productos usando ACT_14
            df_productos_temp['Descripción Familia'] = df_productos_temp['ACT_14'].map(act_to_familia)
            
            # Filtrar por la familia seleccionada
            df_almacen_fam = df_productos_temp[
                df_productos_temp['Descripción Familia'] == familia_seleccionada
            ].copy()
            
            # Identificar productos sin familia asignada
            df_sin_familia = df_productos_temp[
                df_productos_temp['Descripción Familia'].isna()
            ].copy()
            
            # Inicializar df_pendientes como DataFrame vacío
            df_pendientes = pd.DataFrame()
            

            
            if not df_almacen_fam.empty and 'Fecha REAL entrada en almacén' in df_almacen_fam.columns:
                # Convertir fecha de entrada en almacén
                df_almacen_fam['Fecha REAL entrada en almacén'] = pd.to_datetime(
                    df_almacen_fam['Fecha REAL entrada en almacén'], errors='coerce'
                )
                
                # Separar filas con fecha válida y sin fecha
                df_almacen_fam_con_fecha = df_almacen_fam.dropna(subset=['Fecha REAL entrada en almacén'])
                df_almacen_fam_sin_fecha = df_almacen_fam[df_almacen_fam['Fecha REAL entrada en almacén'].isna()].copy()
                
                # Agregar mes de entrada para filas con fecha válida
                df_almacen_fam_con_fecha['Mes Entrada'] = df_almacen_fam_con_fecha['Fecha REAL entrada en almacén'].dt.to_period('M').astype(str)
                
                # Separar filas pendientes de entrega (sin fecha válida)
                if not df_almacen_fam_sin_fecha.empty:
                    # Crear DataFrame separado para pendientes de entrega
                    df_pendientes = df_almacen_fam_sin_fecha.copy()
                    df_pendientes['Estado'] = 'Pendiente de entrega'
                else:
                    df_pendientes = pd.DataFrame()
                
                # Usar solo las filas con fecha válida para el análisis de almacén
                df_almacen_fam = df_almacen_fam_con_fecha
                
                # Obtener el último mes de df_ventas para filtrar los datos
                ultimo_mes_ventas = df_ventas['Mes'].max()
                
                # Preparar datos para la tabla por Temporada
                # Buscar la columna correcta para cantidad de entrada en almacén
                cantidad_col = None
                for col in df_almacen_fam.columns:
                    if 'cantidad' in col.lower() and 'pedida' not in col.lower():
                        cantidad_col = col
                        break
                
                if cantidad_col is None:
                    # Si no encontramos una columna específica, usar 'Cantidad' o la primera columna numérica
                    numeric_cols = df_almacen_fam.select_dtypes(include=[np.number]).columns
                    if 'Cantidad' in df_almacen_fam.columns:
                        cantidad_col = 'Cantidad'
                    elif len(numeric_cols) > 0:
                        cantidad_col = numeric_cols[0]
                    else:
                        st.error("No se encontró una columna de cantidad válida")
                        cantidad_col = 'Cantidad'  # Fallback
                

                
                datos_tabla = (
                    df_almacen_fam.groupby(['Mes Entrada', 'Talla'])[cantidad_col]
                    .sum()
                    .reset_index()
                    .rename(columns={cantidad_col: 'Cantidad Entrada Almacén'})
                    .sort_values(['Mes Entrada', 'Talla'])
                )
                
                # Filtrar datos hasta el último mes de ventas
                datos_tabla = datos_tabla[datos_tabla['Mes Entrada'] <= ultimo_mes_ventas]
                
                if not datos_tabla.empty:
                    # Crear Tema_6 (primeros 6 caracteres) en df_almacen_fam
                    df_almacen_fam['Tema_6'] = df_almacen_fam['Tema'].astype(str).str[:6]
                    
                    # Obtener todos los temas únicos de df_productos (no solo los vendidos)
                    temas_productos = sorted(df_almacen_fam['Tema_6'].unique())
                    
                    # Calcular temas y num_temas SIEMPRE
                    temas = temas_productos
                    num_temas = len(temas)

                    if num_temas > 0:
                        # --- Sección: Entradas almacén y traspasos ---
                        st.markdown('<hr style="margin: 1em 0; border-top: 2px solid #bbb;">', unsafe_allow_html=True)
                        st.markdown('<h4 style="color:#333;font-weight:bold;">Entradas almacén y traspasos</h4>', unsafe_allow_html=True)
                        
                        # Si se han seleccionado tiendas específicas, mostrar tabla de análisis temporal
                        if tiendas_especificas:
                            st.subheader("Análisis Temporal: Entrada Almacén → Envío → Primera Venta")
                            # Preparar datos para el análisis temporal
                            timeline_data = []
                            df_almacen_fam_timeline = df_almacen_fam.copy()
                            df_almacen_fam_timeline['Fecha REAL entrada en almacén'] = pd.to_datetime(
                                df_almacen_fam_timeline['Fecha REAL entrada en almacén'], errors='coerce')
                            df_traspasos_timeline = df_traspasos_filtrado.copy()
                            df_traspasos_timeline['Fecha Enviado'] = pd.to_datetime(
                                df_traspasos_timeline['Fecha Enviado'], errors='coerce')
                            # Create ACT_14 in traspasos data to match warehouse data
                            df_traspasos_timeline['ACT_14'] = df_traspasos_timeline['ACT'].astype(str).str[:14]
                            df_ventas_timeline = df_ventas.copy()
                            
                            df_ventas_timeline['Fecha Documento'] = pd.to_datetime(
                                df_ventas_timeline['Fecha Documento'], errors='coerce')
                            
                            merged = pd.merge(
                                df_almacen_fam_timeline,
                                df_traspasos_timeline,
                                left_on=['ACT_14', 'Talla'],
                                right_on=['ACT_14', 'Talla'],
                                suffixes=('_almacen', '_traspaso')
                            )
                            merged = merged[merged['Fecha Enviado'] >= merged['Fecha REAL entrada en almacén']]
                            
                            for _, row in merged.iterrows():
                                fecha_entrada = row['Fecha REAL entrada en almacén']
                                fecha_envio = row['Fecha Enviado']
                                act = row['ACT_almacen'].strip()  # Remove trailing spaces
                                talla = row['Talla'].strip()  # Remove trailing spaces
                                tienda_envio = row['Tienda']
                                tema = row['Tema']
                                
                                ventas_producto = df_ventas_timeline[
                                    (df_ventas_timeline['ACT'].str.strip() == act) &
                                    (df_ventas_timeline['Talla'].str.strip() == talla) &
                                    (df_ventas_timeline['NombreTPV'].str.strip() == tienda_envio.strip()) &
                                    (df_ventas_timeline['Fecha Documento'] >= fecha_entrada) &
                                    (df_ventas_timeline['Cantidad'] > 0)
                                ]
                                
                                if not ventas_producto.empty:
                                    primera_venta = ventas_producto.loc[ventas_producto['Fecha Documento'].idxmin()]
                                    fecha_primera_venta = primera_venta['Fecha Documento']
                                    dias_entrada_venta = (fecha_primera_venta - fecha_entrada).days
                                else:
                                    fecha_primera_venta = None
                                    dias_entrada_venta = -1  # Use -1 instead of None for "Sin ventas"
                                dias_entrada_envio = (fecha_envio - fecha_entrada).days
                                timeline_data.append({
                                    'ACT': act,
                                    'Tema': tema,
                                    'Talla': talla,
                                    'Tienda Envío': tienda_envio,
                                    'Fecha Entrada Almacén': fecha_entrada.strftime('%d/%m/%Y'),
                                    'Fecha Enviado': fecha_envio.strftime('%d/%m/%Y'),
                                    'Fecha Primera Venta': fecha_primera_venta.strftime('%d/%m/%Y') if fecha_primera_venta else "Sin ventas",
                                    'Días Entrada-Envío': dias_entrada_envio,
                                    'Días Entrada-Primera Venta': dias_entrada_venta if dias_entrada_venta != -1 else -1
                                })
                            
                            if timeline_data:
                                df_timeline = pd.DataFrame(timeline_data)
                                df_timeline['Fecha Entrada Almacén'] = pd.to_datetime(df_timeline['Fecha Entrada Almacén'], format='%d/%m/%Y')
                                df_timeline = df_timeline.sort_values('Fecha Entrada Almacén', ascending=False)
                                df_timeline['Fecha Entrada Almacén'] = df_timeline['Fecha Entrada Almacén'].dt.strftime('%d/%m/%Y')
                                st.dataframe(
                                    df_timeline,
                                    use_container_width=True,
                                    hide_index=True
                                )
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    avg_dias_envio = pd.to_numeric(df_timeline['Días Entrada-Envío'], errors='coerce').mean()
                                    st.metric("Promedio días Entrada→Envío", f"{avg_dias_envio:.1f} días")
                                with col2:
                                    avg_dias_venta = pd.to_numeric(df_timeline['Días Entrada-Primera Venta'].replace('Sin ventas', pd.NA), errors='coerce').mean()
                                    st.metric("Promedio días Entrada→Primera Venta", f"{avg_dias_venta:.1f} días" if not pd.isna(avg_dias_venta) else "N/A")
                                with col3:
                                    total_productos = len(df_timeline)
                                    st.metric("Total productos analizados", f"{total_productos}")
                            else:
                                st.info("No se encontraron datos de envíos para los productos de entrada en almacén de la familia seleccionada.")
                        else:
                            if num_temas == 1:
                                # Un tema: centrado
                                col5a, col5b, col5c = st.columns([1, 2, 1])
                                with col5b:
                                    tema = temas[0]
                                    st.subheader(f"Entrada Almacén - {tema}")
                                    
                                    # Crear gráfico de comparación enviado vs ventas
                                    if tema == 'T_OI25':
                                        temporada_comparacion = 'I2025'
                                    elif tema == 'T_PV25':
                                        temporada_comparacion = 'V2025'
                                    else:
                                        temporada_comparacion = None
                                    
                                    if temporada_comparacion:
                                        ventas_temporada = df_ventas[df_ventas['Temporada'] == temporada_comparacion]
                                        if not ventas_temporada.empty:
                                            act_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]['ACT_14'].unique()
                                            ventas_tema = ventas_temporada[ventas_temporada['ACT'].isin(act_tema)]
                                            if not ventas_tema.empty:
                                                ventas_por_talla = ventas_tema.groupby('Talla')['Cantidad'].sum().reset_index()
                                                enviado_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                                enviado_por_talla = enviado_tema.groupby('Talla')[cantidad_col].sum().reset_index()
                                                datos_comparacion = pd.merge(
                                                    enviado_por_talla, 
                                                    ventas_por_talla, 
                                                    on='Talla', 
                                                    how='outer'
                                                ).fillna(0)
                                                # Ordenar tallas
                                                datos_comparacion = datos_comparacion.sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                x = np.arange(len(datos_comparacion))
                                                width = 0.35
                                                ax.bar(x - width/2, datos_comparacion[cantidad_col], width, label='Enviado Almacén', color='purple', alpha=0.8)
                                                ax.bar(x + width/2, datos_comparacion['Cantidad'], width, label='Ventas', color='darkblue', alpha=0.8)
                                                ax.set_xlabel('Talla')
                                                ax.set_ylabel('Cantidad')
                                                ax.set_title(f'Enviado vs Ventas - {tema} ({temporada_comparacion})')
                                                ax.set_xticks(x)
                                                ax.set_xticklabels(datos_comparacion['Talla'])
                                                ax.legend()
                                                ax.grid(True, alpha=0.3)
                                                st.pyplot(fig)
                                                plt.close()
                                    
                                    # Filtrar datos para este tema específico
                                    datos_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                    datos_tabla_tema = (
                                        datos_tema.groupby(['Mes Entrada', 'Talla'])[cantidad_col]
                                        .sum()
                                        .reset_index()
                                        .rename(columns={cantidad_col: 'Cantidad Entrada Almacén'})
                                        .sort_values(['Mes Entrada', 'Talla'])
                                    )
                                    
                                    if not datos_tabla_tema.empty:
                                        # Crear tabla pivot para mejor visualización
                                        tabla_pivot = datos_tabla_tema.pivot_table(
                                            index='Mes Entrada',
                                            columns='Talla',
                                            values='Cantidad Entrada Almacén',
                                            fill_value=0
                                        ).round(0)
                                        tallas_orden = sorted(tabla_pivot.columns, key=custom_sort_key)
                                        tabla_pivot = tabla_pivot[tallas_orden]
                                        st.dataframe(
                                            tabla_pivot.style.format("{:,.0f}"),
                                            use_container_width=True,
                                            hide_index=False
                                        )
                                        total_temp = tabla_pivot.sum().sum()
                                        st.write(f"**Total Entrada Almacén:** {total_temp:,.0f}")
                                    else:
                                        st.info(f"No hay datos para el tema {tema}")
                            elif num_temas == 2:
                                col5, col6 = st.columns(2)
                                for i, tema in enumerate(temas):
                                    with locals()[f'col{5+i}']:
                                        st.subheader(f"Entrada Almacén - {tema}")
                                        
                                        # Crear gráfico de comparación enviado vs ventas
                                        if tema == 'T_OI25':
                                            temporada_comparacion = 'I2025'
                                        elif tema == 'T_PV25':
                                            temporada_comparacion = 'V2025'
                                        else:
                                            temporada_comparacion = None
                                        
                                        if temporada_comparacion:
                                            ventas_temporada = df_ventas[df_ventas['Temporada'] == temporada_comparacion]
                                            if not ventas_temporada.empty:
                                                act_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]['ACT_14'].unique()
                                                ventas_tema = ventas_temporada[ventas_temporada['ACT'].isin(act_tema)]
                                                if not ventas_tema.empty:
                                                    ventas_por_talla = ventas_tema.groupby('Talla')['Cantidad'].sum().reset_index()
                                                    enviado_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                                    enviado_por_talla = enviado_tema.groupby('Talla')[cantidad_col].sum().reset_index()
                                                    datos_comparacion = pd.merge(
                                                        enviado_por_talla, 
                                                        ventas_por_talla, 
                                                        on='Talla', 
                                                        how='outer'
                                                    ).fillna(0)
                                                    # Ordenar tallas
                                                    datos_comparacion = datos_comparacion.sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                                                    fig, ax = plt.subplots(figsize=(8, 5))
                                                    x = np.arange(len(datos_comparacion))
                                                    width = 0.35
                                                    ax.bar(x - width/2, datos_comparacion[cantidad_col], width, label='Enviado Almacén', color='purple', alpha=0.8)
                                                    ax.bar(x + width/2, datos_comparacion['Cantidad'], width, label='Ventas', color='darkblue', alpha=0.8)
                                                    ax.set_xlabel('Talla')
                                                    ax.set_ylabel('Cantidad')
                                                    ax.set_title(f'Enviado vs Ventas - {tema} ({temporada_comparacion})')
                                                    ax.set_xticks(x)
                                                    ax.set_xticklabels(datos_comparacion['Talla'])
                                                    ax.legend()
                                                    ax.grid(True, alpha=0.3)
                                                    st.pyplot(fig)
                                                    plt.close()
                                        
                                        # Filtrar datos para este tema específico
                                        datos_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                        datos_tabla_tema = (
                                            datos_tema.groupby(['Mes Entrada', 'Talla'])[cantidad_col]
                                            .sum()
                                            .reset_index()
                                            .rename(columns={cantidad_col: 'Cantidad Entrada Almacén'})
                                            .sort_values(['Mes Entrada', 'Talla'])
                                        )
                                        
                                        if not datos_tabla_tema.empty:
                                            # Crear tabla pivot para mejor visualización
                                            tabla_pivot = datos_tabla_tema.pivot_table(
                                                index='Mes Entrada',
                                                columns='Talla',
                                                values='Cantidad Entrada Almacén',
                                                fill_value=0
                                            ).round(0)
                                            tallas_orden = sorted(tabla_pivot.columns, key=custom_sort_key)
                                            tabla_pivot = tabla_pivot[tallas_orden]
                                            st.dataframe(
                                                tabla_pivot.style.format("{:,.0f}"),
                                                use_container_width=True,
                                                hide_index=False
                                            )
                                            total_temp = tabla_pivot.sum().sum()
                                            st.write(f"**Total Entrada Almacén:** {total_temp:,.0f}")
                                        else:
                                            st.info(f"No hay datos para el tema {tema}")
                            else:
                                col5, col6 = st.columns(2)
                                mitad = (num_temas + 1) // 2
                                temas_col5 = temas[:mitad]
                                temas_col6 = temas[mitad:]
                                with col5:
                                    for tema in temas_col5:
                                        st.subheader(f"Entrada Almacén - {tema}")
                                        
                                        # Crear gráfico de comparación enviado vs ventas
                                        if tema == 'T_OI25':
                                            temporada_comparacion = 'I2025'
                                        elif tema == 'T_PV25':
                                            temporada_comparacion = 'V2025'
                                        else:
                                            temporada_comparacion = None
                                        
                                        if temporada_comparacion:
                                            # Obtener datos de ventas para la temporada
                                            ventas_temporada = df_ventas[df_ventas['Temporada'] == temporada_comparacion]
                                            if not ventas_temporada.empty:
                                                # Obtener ACTs del tema actual
                                                act_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]['ACT_14'].unique()
                                                
                                                # Filtrar ventas por ACTs del tema
                                                ventas_tema = ventas_temporada[ventas_temporada['ACT'].isin(act_tema)]
                                                
                                                if not ventas_tema.empty:
                                                    # Agrupar ventas por talla
                                                    ventas_por_talla = ventas_tema.groupby('Talla')['Cantidad'].sum().reset_index()
                                                    
                                                    # Obtener datos de enviado del tema
                                                    enviado_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                                    enviado_por_talla = enviado_tema.groupby('Talla')[cantidad_col].sum().reset_index()
                                                    
                                                    # Combinar datos
                                                    datos_comparacion = pd.merge(
                                                        enviado_por_talla, 
                                                        ventas_por_talla, 
                                                        on='Talla', 
                                                        how='outer'
                                                    ).fillna(0)
                                                    
                                                    # Ordenar tallas
                                                    datos_comparacion = datos_comparacion.sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                                                    
                                                    # Crear gráfico
                                                    fig, ax = plt.subplots(figsize=(8, 5))
                                                    
                                                    x = np.arange(len(datos_comparacion))
                                                    width = 0.35
                                                    
                                                    ax.bar(x - width/2, datos_comparacion[cantidad_col], width, label='Enviado Almacén', color='purple', alpha=0.8)
                                                    ax.bar(x + width/2, datos_comparacion['Cantidad'], width, label='Ventas', color='darkblue', alpha=0.8)
                                                    
                                                    ax.set_xlabel('Talla')
                                                    ax.set_ylabel('Cantidad')
                                                    ax.set_title(f'Enviado vs Ventas - {tema} ({temporada_comparacion})')
                                                    ax.set_xticks(x)
                                                    ax.set_xticklabels(datos_comparacion['Talla'])
                                                    ax.legend()
                                                    ax.grid(True, alpha=0.3)
                                                    st.pyplot(fig)
                                                    plt.close()
                                        
                                        # Filtrar datos para este tema específico
                                        datos_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                        datos_tabla_tema = (
                                            datos_tema.groupby(['Mes Entrada', 'Talla'])[cantidad_col]
                                            .sum()
                                            .reset_index()
                                            .rename(columns={cantidad_col: 'Cantidad Entrada Almacén'})
                                            .sort_values(['Mes Entrada', 'Talla'])
                                        )
                                        
                                        if not datos_tabla_tema.empty:
                                            # Crear tabla pivot para mejor visualización
                                            tabla_pivot = datos_tabla_tema.pivot_table(
                                                index='Mes Entrada',
                                                columns='Talla',
                                                values='Cantidad Entrada Almacén',
                                                fill_value=0
                                            ).round(0)
                                            tallas_orden = sorted(tabla_pivot.columns, key=custom_sort_key)
                                            tabla_pivot = tabla_pivot[tallas_orden]
                                            st.dataframe(
                                                tabla_pivot.style.format("{:,.0f}"),
                                                use_container_width=True,
                                                hide_index=False
                                            )
                                            total_temp = tabla_pivot.sum().sum()
                                            st.write(f"**Total Entrada Almacén:** {total_temp:,.0f}")
                                        else:
                                            st.info(f"No hay datos para el tema {tema}")
                                with col6:
                                    for tema in temas_col6:
                                        st.subheader(f"Entrada Almacén - {tema}")
                                        
                                        # Crear gráfico de comparación enviado vs ventas
                                        if tema == 'T_OI25':
                                            temporada_comparacion = 'I2025'
                                        elif tema == 'T_PV25':
                                            temporada_comparacion = 'V2025'
                                        else:
                                            temporada_comparacion = None
                                        
                                        if temporada_comparacion:
                                            # Obtener datos de ventas para la temporada
                                            ventas_temporada = df_ventas[df_ventas['Temporada'] == temporada_comparacion]
                                            if not ventas_temporada.empty:
                                                # Obtener ACTs del tema actual
                                                act_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]['ACT_14'].unique()
                                                
                                                # Filtrar ventas por ACTs del tema
                                                ventas_tema = ventas_temporada[ventas_temporada['ACT'].isin(act_tema)]
                                                
                                                if not ventas_tema.empty:
                                                    # Agrupar ventas por talla
                                                    ventas_por_talla = ventas_tema.groupby('Talla')['Cantidad'].sum().reset_index()
                                                    
                                                    # Obtener datos de enviado del tema
                                                    enviado_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                                    enviado_por_talla = enviado_tema.groupby('Talla')[cantidad_col].sum().reset_index()
                                                    
                                                    # Combinar datos
                                                    datos_comparacion = pd.merge(
                                                        enviado_por_talla, 
                                                        ventas_por_talla, 
                                                        on='Talla', 
                                                        how='outer'
                                                    ).fillna(0)
                                                    
                                                    # Ordenar tallas
                                                    datos_comparacion = datos_comparacion.sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                                                    
                                                    # Crear gráfico
                                                    fig, ax = plt.subplots(figsize=(8, 5))
                                                    
                                                    x = np.arange(len(datos_comparacion))
                                                    width = 0.35
                                                    
                                                    ax.bar(x - width/2, datos_comparacion[cantidad_col], width, label='Enviado Almacén', color='purple', alpha=0.8)
                                                    ax.bar(x + width/2, datos_comparacion['Cantidad'], width, label='Ventas', color='darkblue', alpha=0.8)
                                                    
                                                    ax.set_xlabel('Talla')
                                                    ax.set_ylabel('Cantidad')
                                                    ax.set_title(f'Enviado vs Ventas - {tema} ({temporada_comparacion})')
                                                    ax.set_xticks(x)
                                                    ax.set_xticklabels(datos_comparacion['Talla'])
                                                    ax.legend()
                                                    ax.grid(True, alpha=0.3)
                                                    st.pyplot(fig)
                                                    plt.close()
                                        
                                        # Filtrar datos para este tema específico
                                        datos_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                        datos_tabla_tema = (
                                            datos_tema.groupby(['Mes Entrada', 'Talla'])[cantidad_col]
                                            .sum()
                                            .reset_index()
                                            .rename(columns={cantidad_col: 'Cantidad Entrada Almacén'})
                                            .sort_values(['Mes Entrada', 'Talla'])
                                        )
                                        
                                        if not datos_tabla_tema.empty:
                                            # Crear tabla pivot para mejor visualización
                                            tabla_pivot = datos_tabla_tema.pivot_table(
                                                index='Mes Entrada',
                                                columns='Talla',
                                                values='Cantidad Entrada Almacén',
                                                fill_value=0
                                            ).round(0)
                                            tallas_orden = sorted(tabla_pivot.columns, key=custom_sort_key)
                                            tabla_pivot = tabla_pivot[tallas_orden]
                                            st.dataframe(
                                                tabla_pivot.style.format("{:,.0f}"),
                                                use_container_width=True,
                                                hide_index=False
                                            )
                                            total_temp = tabla_pivot.sum().sum()
                                            st.write(f"**Total Entrada Almacén:** {total_temp:,.0f}")
                                        else:
                                            st.info(f"No hay datos para el tema {tema}")
                else:
                    st.info("No hay datos de entrada en almacén disponibles para la familia seleccionada.")

            # --- Tabla de Pendientes de Entrega ---
            if not df_pendientes.empty:
                st.markdown("---")
                viz_title("Pendientes de Entrega")
                
                # Buscar la columna correcta para cantidad
                cantidad_col_pendientes = None
                for col in df_pendientes.columns:
                    if 'cantidad' in col.lower() and 'pedida' not in col.lower():
                        cantidad_col_pendientes = col
                        break
                
                if cantidad_col_pendientes is None:
                    # Si no encontramos una columna específica, usar 'Cantidad' o la primera columna numérica
                    numeric_cols = df_pendientes.select_dtypes(include=[np.number]).columns
                    if 'Cantidad' in df_pendientes.columns:
                        cantidad_col_pendientes = 'Cantidad'
                    elif len(numeric_cols) > 0:
                        cantidad_col_pendientes = numeric_cols[0]
                    else:
                        st.error("No se encontró una columna de cantidad válida para pendientes")
                        cantidad_col_pendientes = 'Cantidad'  # Fallback
                
                # Preparar datos de pendientes por talla
                datos_pendientes = (
                    df_pendientes.groupby(['Talla'])[cantidad_col_pendientes]
                    .sum()
                    .reset_index()
                    .rename(columns={cantidad_col_pendientes: 'Cantidad Pendiente'})
                    .sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                )
                
                if not datos_pendientes.empty:
                    # Mostrar tabla de pendientes
                    st.dataframe(
                        datos_pendientes.style.format({
                            'Cantidad Pendiente': '{:,.0f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Mostrar total
                    total_pendientes = datos_pendientes['Cantidad Pendiente'].sum()
                    st.write(f"**Total Pendientes de Entrega:** {total_pendientes:,.0f}")
                    
                    # Mostrar información adicional
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total productos pendientes", len(df_pendientes))
                    with col2:
                        st.metric("Tallas diferentes", len(datos_pendientes))
                    with col3:
                        st.metric("Promedio por talla", f"{total_pendientes/len(datos_pendientes):,.0f}")
                else:
                    st.info("No hay datos de pendientes de entrega para mostrar.")
            else:
                st.info("No hay productos pendientes de entrega.")

            # --- Tabla de Productos Sin Familia Asignada ---
            if not df_sin_familia.empty:
                st.markdown("---")
                viz_title("Productos Sin Familia Asignada")
                
                # Buscar la columna correcta para cantidad
                cantidad_col_sin_familia = None
                for col in df_sin_familia.columns:
                    if 'cantidad' in col.lower() and 'pedida' not in col.lower():
                        cantidad_col_sin_familia = col
                        break
                
                if cantidad_col_sin_familia is None:
                    # Si no encontramos una columna específica, usar 'Cantidad' o la primera columna numérica
                    numeric_cols = df_sin_familia.select_dtypes(include=[np.number]).columns
                    if 'Cantidad' in df_sin_familia.columns:
                        cantidad_col_sin_familia = 'Cantidad'
                    elif len(numeric_cols) > 0:
                        cantidad_col_sin_familia = numeric_cols[0]
                    else:
                        st.error("No se encontró una columna de cantidad válida para productos sin familia")
                        cantidad_col_sin_familia = 'Cantidad'  # Fallback
                
                # Verificar si existe la columna 'Modelo Artículo'
                if 'Modelo Artículo' in df_sin_familia.columns:
                    # Preparar datos de productos sin familia por Modelo Artículo
                    datos_sin_familia = (
                        df_sin_familia.groupby(['Modelo Artículo'])[cantidad_col_sin_familia]
                        .sum()
                        .reset_index()
                        .rename(columns={cantidad_col_sin_familia: 'Cantidad Total'})
                        .sort_values('Cantidad Total', ascending=False)
                    )
                    
                    if not datos_sin_familia.empty:
                        # Mostrar tabla de productos sin familia
                        st.dataframe(
                            datos_sin_familia.style.format({
                                'Cantidad Total': '{:,.0f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Mostrar total
                        total_sin_familia = datos_sin_familia['Cantidad Total'].sum()
                        st.write(f"**Total Productos Sin Familia:** {total_sin_familia:,.0f}")
                        
                        # Mostrar información adicional
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total productos sin familia", len(df_sin_familia))
                        with col2:
                            st.metric("Modelos diferentes", len(datos_sin_familia))
                        with col3:
                            st.metric("Promedio por modelo", f"{total_sin_familia/len(datos_sin_familia):,.0f}")
                    else:
                        st.info("No hay datos de productos sin familia para mostrar.")
                else:
                    st.warning("No se encontró la columna 'Modelo Artículo' en los datos de productos sin familia.")
            else:
                st.info("No hay productos sin familia asignada.")

            # --- Tabla de Cantidad Pedida por Mes y Talla ---
            # Solo mostrar esta tabla cuando NO se han seleccionado tiendas específicas
            if not tiendas_especificas:
                st.markdown("---")
                viz_title("Cantidad Pedida por Mes y Talla")
                
                if not df_almacen_fam.empty and 'Cantidad Pedida' in df_almacen_fam.columns:
                    # Preparar datos de cantidad pedida
                    datos_pedida = (
                        df_almacen_fam.groupby(['Mes Entrada', 'Talla'])['Cantidad Pedida']
                        .sum()
                        .reset_index()
                        .rename(columns={'Mes Entrada': 'Mes', 'Cantidad Pedida': 'Cantidad Pedida'})
                        .sort_values(['Mes', 'Talla'])
                    )
                    
                    # Filtrar datos hasta el último mes de ventas
                    datos_pedida = datos_pedida[datos_pedida['Mes'] <= ultimo_mes_ventas]
                    
                    if not datos_pedida.empty:
                        # Crear tabla pivot para mejor visualización
                        tabla_pedida_pivot = datos_pedida.pivot_table(
                            index='Mes',
                            columns='Talla',
                            values='Cantidad Pedida',
                            fill_value=0
                        ).round(0)
                        
                        # Ordenar tallas usando la función custom_sort_key
                        tallas_orden = sorted(tabla_pedida_pivot.columns, key=custom_sort_key)
                        tabla_pedida_pivot = tabla_pedida_pivot[tallas_orden]
                        
                        # Mostrar la tabla
                        st.dataframe(
                            tabla_pedida_pivot.style.format("{:,.0f}"),
                            use_container_width=True,
                            hide_index=False
                        )
                        
                        # Mostrar total
                        total_pedida = tabla_pedida_pivot.sum().sum()
                        st.write(f"**Total Cantidad Pedida:** {total_pedida:,.0f}")
                    else:
                        st.info("No hay datos de cantidad pedida para la familia seleccionada.")
                else:
                    st.info("No hay datos de cantidad pedida disponibles para la familia seleccionada.")

            # --- Ventas vs Traspasos por Tienda ---
            st.markdown("---")
            viz_title("Ventas vs Traspasos por Tienda")
            
            # Preparar datos de traspasos hasta la fecha máxima de ventas
            ultimo_mes_ventas = df_ventas['Mes'].max()
            df_traspasos_filtrado = df_traspasos_filtrado.copy()
            
            # Convertir fecha de traspasos y filtrar hasta el último mes de ventas
            df_traspasos_filtrado['Mes Enviado'] = pd.to_datetime(df_traspasos_filtrado['Fecha Enviado']).dt.to_period('M').astype(str)
            df_traspasos_filtrado = df_traspasos_filtrado[df_traspasos_filtrado['Mes Enviado'] <= ultimo_mes_ventas]
            
            # Agrupar ventas por tienda y temporada
            ventas_por_tienda_temp = df_ventas.groupby(['Tienda', 'Temporada'])['Cantidad'].sum().reset_index()
            ventas_por_tienda_temp['Tipo'] = 'Ventas'
            ventas_por_tienda_temp = ventas_por_tienda_temp.rename(columns={'Cantidad': 'Cantidad Total'})
            
            # Obtener ACTs que existen en ventas (limpiar espacios)
            act_en_ventas = df_ventas['ACT'].str.strip().unique()
            
            # Limpiar ACTs en traspasos también
            df_traspasos_filtrado['ACT_clean'] = df_traspasos_filtrado['ACT'].str.strip()
            
            # Filtrar traspasos para solo incluir ACTs que están en ventas
            df_traspasos_filtrado_act = df_traspasos_filtrado[df_traspasos_filtrado['ACT_clean'].isin(act_en_ventas)]
            
            # Agrupar traspasos por tienda y temporada
            if not df_traspasos_filtrado_act.empty:
                # Asegurar que la columna Temporada existe en traspasos
                if 'Temporada' not in df_traspasos_filtrado_act.columns:
                    temporada_columns = [col for col in df_traspasos_filtrado_act.columns if 'temporada' in col.lower() or 'season' in col.lower()]
                    if temporada_columns:
                        df_traspasos_filtrado_act['Temporada'] = df_traspasos_filtrado_act[temporada_columns[0]]
                    else:
                        df_traspasos_filtrado_act['Temporada'] = 'Sin Temporada'
                else:
                    df_traspasos_filtrado_act['Temporada'] = df_traspasos_filtrado_act['Temporada'].fillna('Sin Temporada')
                
                # Limpiar temporada en traspasos para que coincida con ventas
                df_traspasos_filtrado_act['Temporada'] = df_traspasos_filtrado_act['Temporada'].str.strip().str[:5]
                
                traspasos_por_tienda_temp = df_traspasos_filtrado_act.groupby(['Tienda', 'Temporada'])['Enviado'].sum().reset_index()
                traspasos_por_tienda_temp['Tipo'] = 'Traspasos'
                traspasos_por_tienda_temp = traspasos_por_tienda_temp.rename(columns={'Enviado': 'Cantidad Total'})
            else:
                # Si no hay traspasos filtrados, crear un DataFrame vacío con la estructura correcta
                traspasos_por_tienda_temp = pd.DataFrame(columns=['Tienda', 'Temporada', 'Cantidad Total', 'Tipo'])
            
            # Combinar datos
            datos_comparacion = pd.concat([ventas_por_tienda_temp, traspasos_por_tienda_temp], ignore_index=True)
            
            if not datos_comparacion.empty:
                # Obtener top 20 tiendas por ventas totales
                top_tiendas_ventas = df_ventas.groupby('Tienda')['Cantidad'].sum().nlargest(20).index.tolist()
                
                # Filtrar datos para top 20 tiendas
                datos_top_tiendas = datos_comparacion[datos_comparacion['Tienda'].isin(top_tiendas_ventas)]
                
                if not datos_top_tiendas.empty:
                    # Crear gráfico con exactamente 2 barras por tienda (Ventas y Traspasos)
                    # Preparar datos para el nuevo formato
                    ventas_data = datos_top_tiendas[datos_top_tiendas['Tipo'] == 'Ventas'].copy()
                    traspasos_data = datos_top_tiendas[datos_top_tiendas['Tipo'] == 'Traspasos'].copy()
                    
                    # Obtener colores de temporada
                    temporada_colors = get_temporada_colors(df_ventas)
                    
                    # Crear figura
                    fig = go.Figure()
                    
                    # Obtener todas las tiendas únicas
                    tiendas_unicas = sorted(datos_top_tiendas['Tienda'].unique())
                    temporadas = sorted(datos_top_tiendas['Temporada'].unique())
                    
                    # Definir diferentes tonos de amarillo para traspasos por temporada
                    yellow_colors = ['#ffff00', '#ffeb3b', '#ffc107', '#ff9800', '#ff5722', '#f57c00', '#ef6c00', '#e65100']
                    
                    # Crear datos para cada tienda con dos barras (Ventas y Traspasos)
                    for tienda in tiendas_unicas:
                        # Datos de ventas para esta tienda
                        ventas_tienda = ventas_data[ventas_data['Tienda'] == tienda]
                        traspasos_tienda = traspasos_data[traspasos_data['Tienda'] == tienda]
                        
                        # Agregar barra de VENTAS (dividida por temporada)
                        if not ventas_tienda.empty:
                            for i, temporada in enumerate(temporadas):
                                ventas_temp = ventas_tienda[ventas_tienda['Temporada'] == temporada]
                                if not ventas_temp.empty:
                                    fig.add_trace(go.Bar(
                                        name=f'Ventas - {temporada}',
                                        x=[f'{tienda} - Ventas'],
                                        y=ventas_temp['Cantidad Total'],
                                        marker_color=temporada_colors.get(temporada, '#1f77b4'),
                                        text=ventas_temp['Cantidad Total'],
                                        texttemplate='%{text:,.0f}',
                                        textposition='inside',
                                        hovertemplate=f"Tienda: {tienda}<br>Tipo: Ventas<br>Temporada: {temporada}<br>Cantidad: %{{y:,.0f}}<extra></extra>",
                                        opacity=0.8,
                                        showlegend=True if tienda == tiendas_unicas[0] else False,  # Solo mostrar legend para la primera tienda
                                        legendgroup=f'Ventas - {temporada}'
                                    ))
                        
                        # Agregar barra de TRASPASOS (dividida por temporada)
                        if not traspasos_tienda.empty:
                            for i, temporada in enumerate(temporadas):
                                traspasos_temp = traspasos_tienda[traspasos_tienda['Temporada'] == temporada]
                                if not traspasos_temp.empty:
                                    # Usar diferentes tonos de amarillo para cada temporada
                                    yellow_color = yellow_colors[i % len(yellow_colors)]
                                    fig.add_trace(go.Bar(
                                        name=f'Traspasos - {temporada}',
                                        x=[f'{tienda} - Traspasos'],
                                        y=traspasos_temp['Cantidad Total'],
                                        marker_color=yellow_color,  # Diferentes tonos de amarillo por temporada
                                        text=traspasos_temp['Cantidad Total'],
                                        texttemplate='%{text:,.0f}',
                                        textposition='inside',
                                        hovertemplate=f"Tienda: {tienda}<br>Tipo: Traspasos<br>Temporada: {temporada}<br>Cantidad: %{{y:,.0f}}<extra></extra>",
                                        opacity=0.8,
                                        showlegend=True if tienda == tiendas_unicas[0] else False,  # Solo mostrar legend para la primera tienda
                                        legendgroup=f'Traspasos - {temporada}'
                                    ))
                    
                    # Configurar layout
                    fig.update_layout(
                        title="Ventas vs Traspasos por Tienda",
                        xaxis_title="Tienda",
                        yaxis_title="Cantidad Total",
                        barmode='stack',  # Barras apiladas por temporada
                        xaxis_tickangle=45,
                        showlegend=True,
                        margin=dict(t=30, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar tabla resumen con breakdown por temporada
                    st.subheader("Resumen de Ventas vs Traspasos por Temporada")
                    
                    # Tabla con breakdown por temporada
                    resumen_temporada = datos_top_tiendas.groupby(['Tienda', 'Tipo', 'Temporada'])['Cantidad Total'].sum().reset_index()
                    resumen_pivot_temp = resumen_temporada.pivot_table(
                        index=['Tienda', 'Temporada'], 
                        columns='Tipo', 
                        values='Cantidad Total', 
                        fill_value=0
                    ).reset_index()
                    
                    # Calcular totales por tienda
                    resumen_totales = datos_top_tiendas.groupby(['Tienda', 'Tipo'])['Cantidad Total'].sum().reset_index()
                    resumen_pivot_totales = resumen_totales.pivot(index='Tienda', columns='Tipo', values='Cantidad Total').fillna(0)
                    resumen_pivot_totales['Diferencia'] = resumen_pivot_totales['Ventas'] - resumen_pivot_totales['Traspasos']
                    
                    
                    resumen_pivot_totales['Eficiencia %'] = (resumen_pivot_totales['Ventas'] / resumen_pivot_totales['Traspasos'] * 100).fillna(0)

                    # Calcular Devoluciones (cantidad negativa) por tienda
                    devoluciones_por_tienda = df_ventas[df_ventas['Cantidad'] < 0].groupby('Tienda')['Cantidad'].sum().abs()
                    resumen_pivot_totales['Devoluciones'] = devoluciones_por_tienda.reindex(resumen_pivot_totales.index).fillna(0)
                    
                    # Calcular Ratio de devolución (Devoluciones / Ventas * 100)
                    resumen_pivot_totales['Ratio de devolución %'] = (resumen_pivot_totales['Devoluciones'] / resumen_pivot_totales['Ventas'] * 100).fillna(0)
                    
                    resumen_pivot_totales = resumen_pivot_totales.round(2)
                    
                    # Mostrar tabla de totales
                    st.write("**Totales por Tienda:**")
                    st.dataframe(
                        resumen_pivot_totales.style.format({
                            'Ventas': '{:,.0f}',
                            'Traspasos': '{:,.0f}',
                            'Diferencia': '{:,.0f}',
                            'Devoluciones': '{:,.0f}',
                            'Eficiencia %': '{:.1f}%',
                            'Ratio de devolución %': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                    
                    # Mostrar tabla detallada por temporada
                    st.write("**Detalle por Temporada:**")
                    st.dataframe(
                        resumen_pivot_temp.style.format({
                            'Ventas': '{:,.0f}',
                            'Traspasos': '{:,.0f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No hay datos suficientes para mostrar la comparación.")
            else:
                st.info("No hay datos de traspasos disponibles para la comparación.")

            
            # --- REVISED: Top/Bottom 10 Complete Descriptions by Family ---
            st.markdown("---")
            viz_title("Análisis de Descripciones por Familia")

            desc_path = os.path.join('data', 'datos_descripciones.xlsx')
            if os.path.exists(desc_path):
                try:
                    df_desc = pd.read_excel(desc_path, engine='openpyxl')
                    
                    # NUEVA LISTA DE COLUMNAS DE DESCRIPCIÓN
                    desc_cols = ['MANGA', 'CUELLO', 'TEJIDO', 'DETALLE', 'ESTILO', 'CORTE']
                    
                    # Verificar que las columnas necesarias existen
                    required_cols = ['ACT', 'fashion_main_description_1'] + desc_cols
                    if all(col in df_desc.columns for col in required_cols):
                        # --- FILTROS ---
                        col_filter1, col_filter2 = st.columns(2)
                        
                        with col_filter1:
                            # Preparamos ventas para cruzar con descripciones
                            ventas_desc = df_ventas.copy()
                            ventas_desc['ACT_clean'] = ventas_desc['ACT'].astype(str).str[:-1]

                            ventas_con_desc_pre = ventas_desc.merge(
                                df_desc[required_cols],
                                left_on='ACT_clean',
                                right_on='ACT',
                                how='inner'
                            )

                            familias_disponibles = sorted(ventas_con_desc_pre['Familia'].dropna().unique())
                            familia_seleccionada = st.selectbox(
                                "Selecciona una Familia:", 
                                familias_disponibles, 
                                key="familia_desc_selector"
                            )

                        with col_filter2:
                            opciones_desc = ["Descripción Completa"] + desc_cols
                            tipo_descripcion = st.selectbox(
                                "Selecciona Tipo de Descripción:", 
                                opciones_desc, 
                                key="tipo_desc_selector"
                            )

                        # --- GENERACIÓN DE DESCRIPCIONES ---
                        if tipo_descripcion == "Descripción Completa":
                            df_desc['Descripción Analizada'] = df_desc['fashion_main_description_1'].fillna('N/A')
                        else:
                            df_desc['Descripción Analizada'] = df_desc[tipo_descripcion].fillna('N/A')
                        
                        df_desc_clean = df_desc[['ACT', 'Descripción Analizada']].copy().dropna()
                        
                        ventas_con_desc = ventas_desc.merge(
                            df_desc_clean,
                            left_on='ACT_clean',
                            right_on='ACT',
                            how='inner'
                        )
                        
                        # FILTRO POR FAMILIA
                        df_familia_desc = ventas_con_desc[ventas_con_desc['Familia'] == familia_seleccionada]
                        
                        desc_group = df_familia_desc.groupby('Descripción Analizada').agg({
                            'Ventas Dinero': 'sum',
                            'Cantidad': 'sum'
                        }).reset_index()
                        
                        # FILTRO DE DESCRIPCIONES VACÍAS
                        desc_group = desc_group[desc_group['Descripción Analizada'] != 'N/A']
                        desc_group = desc_group[desc_group['Descripción Analizada'].str.strip() != '']

                        if not desc_group.empty:
                            desc_group = desc_group.sort_values('Ventas Dinero', ascending=False)
                            top10 = desc_group.head(10)
                            bottom10 = desc_group.tail(10)

                            # --- Side-by-side Top/Bottom 10 Descriptions ---
                            # --- GRÁFICOS DE TOP Y BOTTOM UNO DEBAJO DEL OTRO CON ALTURA DINÁMICA ---

                            # Configuración para altura dinámica
                            altura_por_fila = 40   # Altura estimada por barra
                            altura_minima = 400    # Altura mínima para que no quede muy apretado
                            altura_maxima = 800    # Altura máxima para que no sea exagerado

                            # Calculamos altura según la cantidad de barras
                            altura_top = min(max(len(top10) * altura_por_fila, altura_minima), altura_maxima)
                            altura_bottom = min(max(len(bottom10) * altura_por_fila, altura_minima), altura_maxima)

                            # --- GRÁFICO TOP 10 ---
                            viz_title(f'Top 10 en {tipo_descripcion} - {familia_seleccionada}')
                            fig_top = px.bar(
                                top10, 
                                x='Ventas Dinero', 
                                y='Descripción Analizada', 
                                orientation='h', 
                                color='Ventas Dinero', 
                                color_continuous_scale=COLOR_GRADIENT,
                                text='Cantidad'
                            )
                            fig_top.update_layout(
                                showlegend=False, 
                                height=altura_top,
                                yaxis={'categoryorder':'total ascending', 'title': ''},
                                margin=dict(t=30, b=0, l=0, r=0),
                                paper_bgcolor="rgba(0,0,0,0)", 
                                plot_bgcolor="rgba(0,0,0,0)"
                            )
                            fig_top.update_traces(
                                texttemplate='%{text:,.0f} uds', 
                                textposition='outside', 
                                hovertemplate="Descripción: %{y}<br>Ventas: %{x:,.2f}€<br>Unidades: %{text:,.0f}<extra></extra>",
                                opacity=0.8
                            )
                            st.plotly_chart(fig_top, use_container_width=True, key=f"top10_{tipo_descripcion}_{familia_seleccionada}")

                            # --- GRÁFICO BOTTOM 10 ---
                            viz_title(f'Bottom 10 en {tipo_descripcion} - {familia_seleccionada}')
                            fig_bottom = px.bar(
                                bottom10, 
                                x='Ventas Dinero', 
                                y='Descripción Analizada', 
                                orientation='h', 
                                color='Ventas Dinero', 
                                color_continuous_scale=COLOR_GRADIENT,
                                text='Cantidad'
                            )
                            fig_bottom.update_layout(
                                showlegend=False, 
                                height=altura_bottom,
                                yaxis={'categoryorder':'total ascending', 'title': ''},
                                margin=dict(t=30, b=0, l=0, r=0),
                                paper_bgcolor="rgba(0,0,0,0)", 
                                plot_bgcolor="rgba(0,0,0,0)"
                            )
                            fig_bottom.update_traces(
                                texttemplate='%{text:,.0f} uds', 
                                textposition='outside', 
                                hovertemplate="Descripción: %{y}<br>Ventas: %{x:,.2f}€<br>Unidades: %{text:,.0f}<extra></extra>",
                                opacity=0.8
                            )
                            st.plotly_chart(fig_bottom, use_container_width=True, key=f"bottom10_{tipo_descripcion}_{familia_seleccionada}")
                        else:
                            st.info(f"No hay datos de '{tipo_descripcion}' para la familia '{familia_seleccionada}'.")
                    else:
                        st.warning(f"Una o más columnas de descripción no se encontraron. Se necesitan: {required_cols}")
                except Exception as e:
                    st.error(f"Error crítico al procesar las descripciones de productos: {e}")
            else:
                st.warning("Archivo `datos_descripciones.xlsx` no encontrado en la carpeta `data/`.")

            # --- END REVISED ---

        except Exception as e:
            st.error(f"Error al calcular KPIs: {e}")

    elif seccion == "Geográfico y Tiendas":
        # Preparar datos
        ventas_por_zona = df_ventas.groupby('Zona geográfica')['Cantidad'].sum().reset_index()
        ventas_por_tienda = df_ventas.groupby('NombreTPV')['Cantidad'].sum().reset_index()
        tiendas_por_zona = df_ventas[['NombreTPV', 'Zona geográfica']].drop_duplicates().groupby('Zona geográfica').count().reset_index()

        # 1. KPIs: Mejor y peor tienda por zona
        viz_title("KPIs por Zona - Mejor y Peor Tienda")
        
        try:
            # Calcular ventas por tienda y zona
            ventas_tienda_zona = df_ventas.groupby(['Zona geográfica', 'NombreTPV']).agg({
                'Cantidad': 'sum',
                'Ventas Dinero': 'sum'
            }).reset_index()
            
            # Asegurar que las columnas numéricas son del tipo correcto
            ventas_tienda_zona['Cantidad'] = pd.to_numeric(ventas_tienda_zona['Cantidad'], errors='coerce').fillna(0)
            ventas_tienda_zona['Ventas Dinero'] = pd.to_numeric(ventas_tienda_zona['Ventas Dinero'], errors='coerce').fillna(0)
            
            # Asegurar que Zona geográfica es string
            ventas_tienda_zona['Zona geográfica'] = ventas_tienda_zona['Zona geográfica'].astype(str)
            
            # Calcular media de ventas por zona
            media_por_zona = ventas_tienda_zona.groupby('Zona geográfica')['Cantidad'].mean().reset_index()
            media_por_zona = media_por_zona.rename(columns={'Cantidad': 'Media_Zona'})
            
            # Unir con ventas por tienda
            ventas_tienda_zona = ventas_tienda_zona.merge(media_por_zona, on='Zona geográfica')
            
            # Calcular porcentaje vs media con manejo de división por cero
            ventas_tienda_zona['%_vs_Media'] = 0.0  # Default value
            mask = ventas_tienda_zona['Media_Zona'] > 0
            ventas_tienda_zona.loc[mask, '%_vs_Media'] = (
                (ventas_tienda_zona.loc[mask, 'Cantidad'] - ventas_tienda_zona.loc[mask, 'Media_Zona']) / 
                ventas_tienda_zona.loc[mask, 'Media_Zona'] * 100
            ).round(1)
            
            # Encontrar mejor y peor tienda por zona con manejo de errores
            mejores_tiendas = []
            peores_tiendas = []
            
            for zona in ventas_tienda_zona['Zona geográfica'].unique():
                zona_data = ventas_tienda_zona[ventas_tienda_zona['Zona geográfica'] == zona].copy()
                if not zona_data.empty and len(zona_data) > 0:
                    # Encontrar mejor tienda (máxima cantidad)
                    try:
                        mejor_idx = zona_data['Cantidad'].idxmax()
                        if pd.notna(mejor_idx) and mejor_idx in zona_data.index:
                            mejores_tiendas.append(zona_data.loc[mejor_idx].to_dict())
                    except:
                        pass
                    
                    # Encontrar peor tienda (mínima cantidad)
                    try:
                        peor_idx = zona_data['Cantidad'].idxmin()
                        if pd.notna(peor_idx) and peor_idx in zona_data.index:
                            peores_tiendas.append(zona_data.loc[peor_idx].to_dict())
                    except:
                        pass
            
            mejores_tiendas = pd.DataFrame(mejores_tiendas) if mejores_tiendas else pd.DataFrame()
            peores_tiendas = pd.DataFrame(peores_tiendas) if peores_tiendas else pd.DataFrame()
            
            # Mostrar KPIs en formato de tarjetas
            zonas = sorted([str(z) for z in df_ventas['Zona geográfica'].unique() if pd.notna(z)])
            
            for zona in zonas:
                mejor = mejores_tiendas[mejores_tiendas['Zona geográfica'] == zona] if not mejores_tiendas.empty else pd.DataFrame()
                peor = peores_tiendas[peores_tiendas['Zona geográfica'] == zona] if not peores_tiendas.empty else pd.DataFrame()
                
                if not mejor.empty and not peor.empty:
                    try:
                        st.markdown(f"""
                        <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                            <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                                {zona}
                            </div>
                            <div style="display: flex; justify-content: space-between; gap: 15px;">
                                <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                                    <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Mejor Tienda</p>
                                    <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{mejor.iloc[0]['NombreTPV']}</p>
                                    <p style="color: #059669; font-size: 14px; margin: 5px 0 0 0;">{mejor.iloc[0]['Cantidad']:,.0f} uds</p>
                                    <p style="color: #059669; font-size: 14px; margin: 0;">{mejor.iloc[0]['Ventas Dinero']:,.2f}€</p>
                                </div>
                                <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                                    <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Peor Tienda</p>
                                    <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{peor.iloc[0]['NombreTPV']}</p>
                                    <p style="color: #dc2626; font-size: 14px; margin: 5px 0 0 0;">{peor.iloc[0]['Cantidad']:,.0f} uds</p>
                                    <p style="color: #dc2626; font-size: 14px; margin: 0;">{peor.iloc[0]['Ventas Dinero']:,.2f}€</p>
                                    <p style="color: #dc2626; font-size: 12px; margin: 5px 0 0 0;">{peor.iloc[0]['%_vs_Media']}% vs media</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error al mostrar KPIs para {zona}: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error al procesar KPIs por zona: {str(e)}")
            st.info("Mostrando información básica de zonas...")
            
            # Fallback: mostrar información básica
            zonas_basicas = df_ventas.groupby('Zona geográfica')['Cantidad'].sum().reset_index()
            st.dataframe(zonas_basicas, use_container_width=True)

        # 2. Row: Ventas por zona y Tiendas por zona
        col1, col2 = st.columns(2)
        
        with col1:
            viz_title("Ventas por Zona")
            fig = px.bar(ventas_por_zona, 
                        x='Zona geográfica', 
                        y='Cantidad',
                        color='Cantidad',
                        color_continuous_scale=COLOR_GRADIENT,
                        text='Cantidad')
            fig.update_layout(
                showlegend=False,
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            viz_title("Tiendas por Zona")
            fig = px.bar(tiendas_por_zona, 
                        x='Zona geográfica', 
                        y='NombreTPV',
                        color='NombreTPV',
                        color_continuous_scale=COLOR_GRADIENT,
                        text='NombreTPV')
            fig.update_layout(
                showlegend=False,
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        # 3. Row: Evolución mensual por zona
        viz_title("Evolución Mensual por Zona")
        zona_mes_evol = df_ventas.groupby(['Mes', 'Zona geográfica'])['Cantidad'].sum().reset_index()
        fig = px.line(zona_mes_evol, 
                     x='Mes', 
                     y='Cantidad',
                     color='Zona geográfica',
                     color_discrete_sequence=COLOR_GRADIENT)
        fig.update_layout(
            showlegend=True,
            legend_title_text='Zona Geográfica',
            xaxis_tickangle=45,
            margin=dict(t=30, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        fig.update_traces(opacity=0.8)
        st.plotly_chart(fig, use_container_width=True)

        # 4. Row: Mapa España y tabla
        col3, col4 = st.columns(2)
        
        with col3:
            viz_title("Mapa de Ventas - España")
            
            # Separar datos por país
            df_espana = df_ventas[~df_ventas['NombreTPV'].isin(TIENDAS_EXTRANJERAS)].copy()
            
            # Procesar datos de España usando Zona geográfica
            mapeo_zona_ciudad = {
                'Zona Madrid': 'MADRID',
                'Zona Andalucía': 'SEVILLA',
                'Zona Valencia': 'VALENCIA',
                'Zona Galicia': 'VIGO',
                'Zona Murcia': 'MURCIA',
                'Zona Castilla y León': 'SALAMANCA',
                'Zona País Vasco': 'BILBAO',
                'Zona Aragón': 'ZARAGOZA',
                'Zona Asturias': 'GIJON',
                'Zona Castilla-La Mancha': 'ALBACETE',
                'Zona Cataluña': 'BARCELONA',
                'Zona Cantabria': 'SANTANDER',
                'Zona Navarra': 'PAMPLONA',
                'Zona La Rioja': 'LOGROÑO',
                'Zona Extremadura': 'BADAJOZ',
                'Zona Canarias': 'LAS PALMAS',
                'Zona Baleares': 'PALMA'
            }
            
            # Asignar ciudad basada en zona geográfica
            df_espana['Ciudad'] = df_espana['Zona geográfica'].map(mapeo_zona_ciudad)
            
            # Para tiendas sin zona geográfica, intentar extraer del nombre
            df_espana['Ciudad'] = df_espana['Ciudad'].fillna(
                df_espana['NombreTPV'].str.extract(r'ET\d{1,2}-([\w\s\.\(\)]+)')[0]
                .str.upper()
                .str.replace(r'ECITRUCCO|ECI|XANADU|TRUCCO|CORT.*|\(.*\)', '', regex=True)
                .str.strip()
            )

            coordenadas_espana = {
                'MADRID': (40.4168, -3.7038),
                'SEVILLA': (37.3886, -5.9823),
                'MALAGA': (36.7213, -4.4214),
                'VALENCIA': (39.4699, -0.3763),
                'VIGO': (42.2406, -8.7207),
                'MURCIA': (37.9834, -1.1299),
                'SALAMANCA': (40.9701, -5.6635),
                'CORDOBA': (37.8882, -4.7794),
                'BILBAO': (43.2630, -2.9350),
                'ZARAGOZA': (41.6488, -0.8891),
                'JAEN': (37.7796, -3.7849),
                'GIJON': (43.5453, -5.6615),
                'ALBACETE': (38.9943, -1.8585),
                'GRANADA': (37.1773, -3.5986),
                'CARTAGENA': (37.6051, -0.9862),
                'TARRAGONA': (41.1189, 1.2445),
                'LEON': (42.5987, -5.5671),
                'SANTANDER': (43.4623, -3.8099),
                'PAMPLONA': (42.8125, -1.6458),
                'VITORIA': (42.8467, -2.6727),
                'CASTELLON': (39.9864, -0.0513),
                'CADIZ': (36.5271, -6.2886),
                'JEREZ': (36.6850, -6.1261),
                'AVILES': (43.5560, -5.9222),
                'BADAJOZ': (38.8794, -6.9707),
                'BARCELONA': (41.3851, 2.1734),
                'LOGROÑO': (42.4627, -2.4449),
                'LAS PALMAS': (28.1235, -15.4366),
                'PALMA': (39.5696, 2.6502)
            }

            # Procesar datos para España
            df_espana['lat'] = df_espana['Ciudad'].map(lambda c: coordenadas_espana.get(c, (None, None))[0])
            df_espana['lon'] = df_espana['Ciudad'].map(lambda c: coordenadas_espana.get(c, (None, None))[1])
            df_espana = df_espana.dropna(subset=['lat', 'lon'])

            # Agrupar por ciudad incluyendo tanto cantidad como ventas en euros
            ventas_ciudad_espana = df_espana.groupby(['Ciudad', 'lat', 'lon']).agg({
                'Cantidad': 'sum',
                'Ventas Dinero': 'sum'
            }).reset_index()
            
            # --- FIX: asegurar que 'Cantidad' no tenga valores negativos ni NaN para el mapa ---
            ventas_ciudad_espana['Cantidad'] = pd.to_numeric(ventas_ciudad_espana['Cantidad'], errors='coerce').fillna(0)
            ventas_ciudad_espana['Cantidad'] = ventas_ciudad_espana['Cantidad'].clip(lower=0)
            # --- FIN FIX ---
            
            if not ventas_ciudad_espana.empty:
                fig_espana = px.scatter_mapbox(
                    ventas_ciudad_espana,
                    lat='lat',
                    lon='lon',
                    size='Cantidad',
                    color='Cantidad',
                    hover_name='Ciudad',
                    hover_data={'Cantidad': True, 'Ventas Dinero': True},
                    color_continuous_scale='Viridis',
                    zoom=5,
                    height=400,
                    title="España - Ventas por Ciudad"
                )
                fig_espana.update_layout(
                    mapbox_style='open-street-map',
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_espana, use_container_width=True)
            else:
                st.info("No hay datos disponibles para España.")
        
        with col4:
            # Tabla de tiendas por ciudad - España
            if not ventas_ciudad_espana.empty:
                st.write("**Tiendas por Ciudad - España**")
                
                # Mostrar tabla resumida por ciudad con cantidad y euros
                resumen_espana = ventas_ciudad_espana.sort_values('Cantidad', ascending=False)
                st.dataframe(
                    resumen_espana[['Ciudad', 'Cantidad', 'Ventas Dinero']].style.format({
                        'Cantidad': '{:,.0f}',
                        'Ventas Dinero': '{:,.2f}€'
                    }),
                    use_container_width=True
                )
                
                # Mostrar información de debug
                st.write(f"**Tiendas españolas:** {len(df_espana['NombreTPV'].unique())}")
                st.write(f"**Ciudades mapeadas:** {len(ventas_ciudad_espana)}")
            else:
                st.info("No hay datos disponibles para España.")

        # 5. Row: Mapa Italia y tabla
        col5, col6 = st.columns(2)
        
        with col5:
            viz_title("Mapa de Ventas - Italia")
            
            # Separar datos por país
            df_italia = df_ventas[df_ventas['NombreTPV'].isin(TIENDAS_EXTRANJERAS)].copy()
            
            # Procesar datos de Italia
            df_italia['Ciudad'] = df_italia['NombreTPV'].str.extract(r'I\d{3}COIN([A-Z]+)')[0]
            df_italia['Ciudad'] = df_italia['Ciudad'].fillna('MILANO')

            coordenadas_italia = {
                'BERGAMO': (45.6983, 9.6773),
                'VARESE': (45.8206, 8.8256),
                'BARICASAMASSIMA': (40.9634, 16.7514),
                'MILANO5GIORNATE': (45.4642, 9.1900),
                'ROMACINECITTA': (41.9028, 12.4964),
                'GENOVA': (44.4056, 8.9463),
                'SASSARI': (40.7259, 8.5557),
                'CATANIA': (37.5079, 15.0830),
                'CAGLIARI': (39.2238, 9.1217),
                'LECCE': (40.3519, 18.1720),
                'MILANOCANTORE': (45.4642, 9.1900),
                'MESTRE': (45.4903, 12.2424),
                'PADOVA': (45.4064, 11.8768),
                'FIRENZE': (43.7696, 11.2558),
                'ROMASANGIOVANNI': (41.9028, 12.4964),
                'MILANO': (45.4642, 9.1900)
            }

            # Procesar datos para Italia
            df_italia['lat'] = df_italia['Ciudad'].map(lambda c: coordenadas_italia.get(c, (None, None))[0])
            df_italia['lon'] = df_italia['Ciudad'].map(lambda c: coordenadas_italia.get(c, (None, None))[1])
            df_italia = df_italia.dropna(subset=['lat', 'lon'])

            # Agrupar por ciudad incluyendo tanto cantidad como ventas en euros
            ventas_ciudad_italia = df_italia.groupby(['Ciudad', 'lat', 'lon']).agg({
                'Cantidad': 'sum',
                'Ventas Dinero': 'sum'
            }).reset_index()
            
            # --- FIX: asegurar que 'Cantidad' no tenga valores negativos ni NaN para el mapa de Italia ---
            ventas_ciudad_italia['Cantidad'] = pd.to_numeric(ventas_ciudad_italia['Cantidad'], errors='coerce').fillna(0)
            ventas_ciudad_italia['Cantidad'] = ventas_ciudad_italia['Cantidad'].clip(lower=0)
            # --- FIN FIX ---
            
            if not ventas_ciudad_italia.empty:
                fig_italia = px.scatter_mapbox(
                    ventas_ciudad_italia,
                    lat='lat',
                    lon='lon',
                    size='Cantidad',
                    color='Cantidad',
                    hover_name='Ciudad',
                    hover_data={'Cantidad': True, 'Ventas Dinero': True},
                    color_continuous_scale='Plasma',
                    zoom=5,
                    height=400,
                    title="Italia - Ventas por Ciudad"
                )
                fig_italia.update_layout(
                    mapbox_style='open-street-map',
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_italia, use_container_width=True)
            else:
                st.info("No hay datos disponibles para Italia.")
        
        with col6:
            # Tabla de tiendas por ciudad - Italia
            if not ventas_ciudad_italia.empty:
                st.write("**Tiendas por Ciudad - Italia**")
                
                # Mostrar tabla resumida por ciudad con cantidad y euros
                resumen_italia = ventas_ciudad_italia.sort_values('Cantidad', ascending=False)
                st.dataframe(
                    resumen_italia[['Ciudad', 'Cantidad', 'Ventas Dinero']].style.format({
                        'Cantidad': '{:,.0f}',
                        'Ventas Dinero': '{:,.2f}€'
                    }),
                    use_container_width=True
                )
                
                # Mostrar información de debug
                st.write(f"**Tiendas italianas:** {len(df_italia['NombreTPV'].unique())}")
                st.write(f"**Ciudades mapeadas:** {len(ventas_ciudad_italia)}")
            else:
                st.info("No hay datos disponibles para Italia.")


    elif seccion == "Producto, Campaña, Devoluciones y Rentabilidad":
        devoluciones = df_ventas[df_ventas['Cantidad'] < 0].copy()
        ventas = df_ventas[df_ventas['Cantidad'] > 0].copy()

        # Calcular descuento real basado en la diferencia entre PVP y precio real de venta
        if all(col in df_ventas.columns for col in ['P.V.P.', 'Subtotal', 'Cantidad']):
            df_ventas['Precio Real Unitario'] = df_ventas['Subtotal'] / df_ventas['Cantidad']
            df_ventas['Descuento Real %'] = ((df_ventas['P.V.P.'] - df_ventas['Precio Real Unitario']) / df_ventas['P.V.P.'] * 100).fillna(0)
            df_ventas['Descuento Real %'] = df_ventas['Descuento Real %'].clip(0, 100)  # Limitar entre 0 y 100%

        # ===== KPIs =====
        st.markdown("### 📊 **KPIs de Devoluciones, Rebajas y Margen**")
        
        # Calculate KPIs
        # Tienda con más devoluciones y ratio
        tienda_mas_devoluciones = "Sin datos"
        ratio_devolucion_valor = 0
        if not devoluciones.empty:
            devoluciones_por_tienda = devoluciones.groupby('NombreTPV').agg({'Cantidad': 'sum'}).reset_index()
            devoluciones_por_tienda['Cantidad'] = abs(devoluciones_por_tienda['Cantidad'])
            devoluciones_por_tienda = devoluciones_por_tienda.sort_values('Cantidad', ascending=False)
            
            # Calcular ratio de devolución por tienda
            ventas_por_tienda = ventas.groupby('NombreTPV')['Cantidad'].sum().reset_index()
            ratio_devolucion = ventas_por_tienda.merge(devoluciones_por_tienda, on='NombreTPV', how='left')
            ratio_devolucion['Cantidad_y'] = ratio_devolucion['Cantidad_y'].fillna(0)
            ratio_devolucion['Ratio Devolución %'] = (ratio_devolucion['Cantidad_y'] / ratio_devolucion['Cantidad_x'] * 100).round(2)
            
            top_tienda_devolucion = ratio_devolucion.loc[ratio_devolucion['Cantidad_y'].idxmax()]
            tienda_mas_devoluciones = top_tienda_devolucion['NombreTPV']
            ratio_devolucion_valor = top_tienda_devolucion['Ratio Devolución %']
        
        # Talla más devuelta
        talla_mas_devuelta = "Sin datos"
        talla_devuelta_unidades = 0
        if not devoluciones.empty and 'Talla' in devoluciones.columns:
            talla_mas_devuelta_data = devoluciones.groupby('Talla')['Cantidad'].sum().abs().sort_values(ascending=False).head(1)
            if not talla_mas_devuelta_data.empty:
                talla_mas_devuelta = talla_mas_devuelta_data.index[0]
                talla_devuelta_unidades = talla_mas_devuelta_data.iloc[0]
        
        # Familia más devuelta
        familia_mas_devuelta = "Sin datos"
        familia_devuelta_unidades = 0
        if not devoluciones.empty:
            familia_mas_devuelta_data = devoluciones.groupby('Familia')['Cantidad'].sum().abs().sort_values(ascending=False).head(1)
            if not familia_mas_devuelta_data.empty:
                familia_mas_devuelta = familia_mas_devuelta_data.index[0]
                familia_devuelta_unidades = familia_mas_devuelta_data.iloc[0]
        
        # Rebajas 1ª (Enero y Junio)
        ventas_rebajas_1 = 0
        porcentaje_rebajas_1 = 0
        if 'Fecha Documento' in df_ventas.columns:
            df_ventas_temp = df_ventas.copy()
            df_ventas_temp['Fecha Documento'] = pd.to_datetime(df_ventas_temp['Fecha Documento'], errors='coerce')
            df_ventas_temp['mes'] = df_ventas_temp['Fecha Documento'].dt.month
            df_ventas_temp = df_ventas_temp[df_ventas_temp['Subtotal'] > 0]
            rebajas_1 = df_ventas_temp[df_ventas_temp['mes'].isin([1, 6])]
            
            if 'precio_pvp' in rebajas_1.columns:
                rebajas_1['Precio_venta'] = rebajas_1['Subtotal'] / rebajas_1['Cantidad']
                rebajas_1_real = rebajas_1[rebajas_1['Precio_venta'] < rebajas_1['precio_pvp']]
                ventas_rebajas_1 = rebajas_1_real['Subtotal'].sum()
            else:
                ventas_rebajas_1 = rebajas_1['Subtotal'].sum()
            
            if df_ventas_temp['Subtotal'].sum() > 0:
                porcentaje_rebajas_1 = (ventas_rebajas_1 / df_ventas_temp['Subtotal'].sum() * 100)
        
        # Rebajas 2ª (Febrero y Julio)
        ventas_rebajas_2 = 0
        porcentaje_rebajas_2 = 0
        if 'Fecha Documento' in df_ventas.columns:
            df_ventas_temp = df_ventas.copy()
            df_ventas_temp['Fecha Documento'] = pd.to_datetime(df_ventas_temp['Fecha Documento'], errors='coerce')
            df_ventas_temp['mes'] = df_ventas_temp['Fecha Documento'].dt.month
            df_ventas_temp = df_ventas_temp[df_ventas_temp['Subtotal'] > 0]
            rebajas_2 = df_ventas_temp[df_ventas_temp['mes'].isin([2, 7])]
            
            if 'precio_pvp' in rebajas_2.columns:
                rebajas_2['Precio_venta'] = rebajas_2['Subtotal'] / rebajas_2['Cantidad']
                rebajas_2_real = rebajas_2[rebajas_2['Precio_venta'] < rebajas_2['precio_pvp']]
                ventas_rebajas_2 = rebajas_2_real['Subtotal'].sum()
            else:
                ventas_rebajas_2 = rebajas_2['Subtotal'].sum()
            
            if df_ventas_temp['Subtotal'].sum() > 0:
                porcentaje_rebajas_2 = (ventas_rebajas_2 / df_ventas_temp['Subtotal'].sum() * 100)
        
        # Margen bruto por unidad (promedio)
        margen_unitario_promedio = 0
        if all(col in df_ventas.columns for col in ['P.V.P.', 'Precio Coste']):
            df_ventas_temp = df_ventas.copy()
            df_ventas_temp['margen_unitario'] = df_ventas_temp['P.V.P.'] - df_ventas_temp['Precio Coste']
            margen_unitario_promedio = df_ventas_temp['margen_unitario'].mean()
        
        # Margen porcentual (promedio)
        margen_porcentual_promedio = 0
        if all(col in df_ventas.columns for col in ['P.V.P.', 'Precio Coste']):
            df_ventas_temp = df_ventas.copy()
            df_ventas_temp['margen_unitario'] = df_ventas_temp['P.V.P.'] - df_ventas_temp['Precio Coste']
            df_ventas_temp['margen_%'] = df_ventas_temp['margen_unitario'] / df_ventas_temp['P.V.P.']
            margen_porcentual_promedio = df_ventas_temp['margen_%'].mean() * 100
        
        # KPIs in HTML style like Resumen General
        st.markdown("""
            <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                    KPIs de Devoluciones y Rebajas
                </div>
                <div style="display: flex; justify-content: space-between; gap: 15px;">
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tienda con más devoluciones</p>
                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                        <p style="color: #dc2626; font-size: 12px; margin: 0;">Ratio: {:.1f}%</p>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Talla más devuelta</p>
                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                        <p style="color: #dc2626; font-size: 12px; margin: 0;">{:.0f} unidades</p>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Familia más devuelta</p>
                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                        <p style="color: #dc2626; font-size: 12px; margin: 0;">{:.0f} unidades</p>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Rebajas 1ª (Enero/Junio)</p>
                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{:,.0f}€</p>
                        <p style="color: #059669; font-size: 12px; margin: 0;">{:.1f}% del total</p>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Rebajas 2ª (Febrero/Julio)</p>
                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{:,.0f}€</p>
                        <p style="color: #059669; font-size: 12px; margin: 0;">{:.1f}% del total</p>
                    </div>
                </div>
            </div>
        """.format(tienda_mas_devoluciones, ratio_devolucion_valor, talla_mas_devuelta, talla_devuelta_unidades, 
                   familia_mas_devuelta, familia_devuelta_unidades, ventas_rebajas_1, porcentaje_rebajas_1, 
                   ventas_rebajas_2, porcentaje_rebajas_2), unsafe_allow_html=True)
        
        # Margen KPIs in separate row
        st.markdown("""
            <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                    KPIs de Margen
                </div>
                <div style="display: flex; justify-content: space-between; gap: 15px;">
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Margen Unitario Promedio</p>
                        <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:.2f}€</p>
                        <p style="color: #059669; font-size: 12px; margin: 0;">por unidad</p>
                    </div>
                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Margen % Promedio</p>
                        <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:.1f}%</p>
                        <p style="color: #059669; font-size: 12px; margin: 0;">del PVP</p>
                    </div>
                </div>
            </div>
        """.format(margen_unitario_promedio, margen_porcentual_promedio), unsafe_allow_html=True)

        # ===== GRÁFICOS =====
        st.markdown("### **Análisis de Devoluciones y Temporadas**")

        # Row 1: Ventas vs Devoluciones por Familia
        st.markdown("#### **Ventas vs Devoluciones por Familia**")
        
        if not devoluciones.empty:
            # Preparar datos para comparación
            ventas_por_familia = ventas.groupby('Familia')['Cantidad'].sum().reset_index()
            ventas_por_familia['Tipo'] = 'Ventas'
            
            devoluciones_por_familia = devoluciones.groupby('Familia')['Cantidad'].sum().reset_index()
            devoluciones_por_familia['Cantidad'] = abs(devoluciones_por_familia['Cantidad'])
            devoluciones_por_familia['Tipo'] = 'Devoluciones'
            
            # Combinar datos
            comparacion_familias = pd.concat([ventas_por_familia, devoluciones_por_familia], ignore_index=True)
            
            # Crear gráfico de barras agrupadas
            fig = px.bar(
                comparacion_familias,
                x='Familia',
                y='Cantidad',
                color='Tipo',
                color_discrete_map={'Ventas': '#0066cc', 'Devoluciones': '#ff4444'},
                barmode='group',
                title="Ventas vs Devoluciones por Familia"
            )
            
            fig.update_layout(
                xaxis_tickangle=45,
                height=500,
                showlegend=True,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de devoluciones disponibles para mostrar la comparación.")

        # Row 2: Talla más devuelta y menos devuelta por familia
        st.markdown("#### **Análisis de Tallas por Familia**")
        
        if not devoluciones.empty and 'Talla' in devoluciones.columns:
            col_talla1, col_talla2 = st.columns(2)
            
            with col_talla1:
                # Talla más devuelta por familia
                talla_mas_devuelta_familia = devoluciones.groupby(['Familia', 'Talla'])['Cantidad'].sum().abs().reset_index()
                talla_mas_devuelta_familia = talla_mas_devuelta_familia.loc[talla_mas_devuelta_familia.groupby('Familia')['Cantidad'].idxmax()]
                
                fig = px.bar(
                    talla_mas_devuelta_familia,
                    x='Familia',
                    y='Cantidad',
                    color='Talla',
                    title="Talla más devuelta por Familia",
                    color_discrete_sequence=px.colors.sequential.Reds
                )
                fig.update_layout(
                    xaxis_tickangle=45,
                    height=400,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_talla2:
                # Talla menos devuelta por familia
                talla_menos_devuelta_familia = devoluciones.groupby(['Familia', 'Talla'])['Cantidad'].sum().abs().reset_index()
                talla_menos_devuelta_familia = talla_menos_devuelta_familia.loc[talla_menos_devuelta_familia.groupby('Familia')['Cantidad'].idxmin()]
                
                fig = px.bar(
                    talla_menos_devuelta_familia,
                    x='Familia',
                    y='Cantidad',
                    color='Talla',
                    title="Talla menos devuelta por Familia",
                    color_discrete_sequence=px.colors.sequential.Reds
                )
                fig.update_layout(
                    xaxis_tickangle=45,
                    height=400,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos de tallas en las devoluciones disponibles.")

        # Row 3: Análisis de ventas en/ fuera de temporada
        st.markdown("#### ** Análisis de Ventas por Temporada**")
        
        if 'Temporada' in df_ventas.columns:
            # Función para determinar si un producto fue vendido fuera de su temporada
            def vendido_fuera_temporada(row):
                temporada = row['Temporada']
                fecha = row['Fecha Documento']

                # Si la temporada no está bien definida, marcamos como fuera de temporada
                if not isinstance(temporada, str) or len(temporada) < 5:
                    return 1

                tipo_temporada = temporada[0]   # 'I' o 'V'
                ano_temporada_str = temporada[1:]

                # Validar que ano_temporada_str es numérico y tipo_temporada es válido
                if tipo_temporada not in ['I', 'V'] or not ano_temporada_str.isdigit():
                    return 1

                ano_temporada = int(ano_temporada_str)

                if tipo_temporada == 'I':  # Invierno: sept (año-1) a feb (año)
                    inicio = pd.Timestamp(year=ano_temporada - 1, month=9, day=1)
                    fin = pd.Timestamp(year=ano_temporada, month=2, day=28)  # ignoramos bisiestos
                    if inicio <= fecha <= fin:
                        return 0  # Vendido dentro de su temporada (Invierno)
                    else:
                        return 1  # Vendido fuera de temporada

                elif tipo_temporada == 'V':  # Verano: marzo a agosto año
                    inicio = pd.Timestamp(year=ano_temporada, month=3, day=1)
                    fin = pd.Timestamp(year=ano_temporada, month=8, day=31)
                    if inicio <= fecha <= fin:
                        return 0  # Vendido dentro de su temporada (Verano)
                    else:
                        return 1  # Vendido fuera de temporada
                else:
                    return 1  # Temporada no reconocida

            # Aplicar la función al DataFrame
            df_ventas_temp = df_ventas.copy()
            df_ventas_temp['vendido_fuera_temporada'] = df_ventas_temp.apply(vendido_fuera_temporada, axis=1)
            
            # Agrupar por temporada y tipo de venta
            analisis_temporada = df_ventas_temp.groupby(['Temporada', 'vendido_fuera_temporada'])['Cantidad'].sum().reset_index()
            analisis_temporada['Tipo_Venta'] = analisis_temporada['vendido_fuera_temporada'].map({
                0: 'En Temporada',
                1: 'Fuera de Temporada'
            })
            
            # Crear gráfico
            fig = px.bar(
                analisis_temporada,
                x='Temporada',
                y='Cantidad',
                color='Tipo_Venta',
                color_discrete_map={'En Temporada': '#0066cc', 'Fuera de Temporada': '#ff4444'},
                barmode='stack',
                title="Ventas En vs Fuera de Temporada por Campaña"
            )
            
            fig.update_layout(
                xaxis_tickangle=45,
                height=500,
                showlegend=True,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar tabla resumen
            st.markdown("**Resumen de Ventas por Temporada:**")
            resumen_temporada = analisis_temporada.pivot_table(
                index='Temporada',
                columns='Tipo_Venta',
                values='Cantidad',
                fill_value=0
            ).reset_index()
            # Asegurar que ambas columnas existen
            for col in ['En Temporada', 'Fuera de Temporada']:
                if col not in resumen_temporada.columns:
                    resumen_temporada[col] = 0
            # Calcular porcentajes
            resumen_temporada['Total'] = resumen_temporada['En Temporada'] + resumen_temporada['Fuera de Temporada']
            resumen_temporada['% En Temporada'] = (resumen_temporada['En Temporada'] / resumen_temporada['Total'] * 100).round(1)
            resumen_temporada['% Fuera de Temporada'] = (resumen_temporada['Fuera de Temporada'] / resumen_temporada['Total'] * 100).round(1)
            # Ordenar columnas para visual consistencia
            resumen_temporada = resumen_temporada[['Temporada', 'En Temporada', 'Fuera de Temporada', 'Total', '% En Temporada', '% Fuera de Temporada']]
            st.dataframe(
                resumen_temporada,
                use_container_width=True,
                hide_index=True
            )
            
        else:
            st.info("No hay datos de temporada disponibles para el análisis.")

        # ===== TABLA DE PRODUCTOS CON BAJO MARGEN (al final) =====
        st.markdown("---")
        st.markdown("### **Productos con Bajo Margen**")
        
        # Buscar columnas que podrían ser Precio Coste
        columnas_disponibles = df_ventas.columns.tolist()
        coste_col = None
        
        # Buscar Precio Coste
        for col in ['precio_cost', 'precio_coste', 'Precio Coste', 'Precio Costo', 'Coste', 'Costo', 'coste', 'costo']:
            if col in columnas_disponibles:
                coste_col = col
                break
        
        if coste_col and 'Subtotal' in df_ventas.columns and 'Cantidad' in df_ventas.columns:
            # Slider para ajustar el umbral de margen
            umbral_margen = st.slider(
                "Umbral de margen % (productos por debajo de este valor):",
                min_value=0.0,
                max_value=1.0,
                value=0.36,
                step=0.01,
                format="%.2f"
            )
            
            # Calcular márgenes usando Ventas Dinero (Subtotal) como precio de venta
            # Excluir devoluciones (Cantidad < 0)
            df_ventas_temp = df_ventas[df_ventas['Cantidad'] > 0].copy()
            df_ventas_temp['Precio_venta'] = df_ventas_temp['Subtotal'] / df_ventas_temp['Cantidad']
            df_ventas_temp['margen_unitario'] = df_ventas_temp['Precio_venta'] - df_ventas_temp[coste_col]
            df_ventas_temp['margen_%'] = df_ventas_temp['margen_unitario'] / df_ventas_temp['Precio_venta']
            
            # Filtrar productos con margen bajo (incluyendo márgenes negativos)
            productos_bajo_margen = df_ventas_temp[df_ventas_temp['margen_%'] < umbral_margen].copy()
            
            if not productos_bajo_margen.empty:
                # Preparar tabla con las columnas solicitadas
                tabla_bajo_margen = productos_bajo_margen[[
                    'ACT', 'Descripción Familia', 'Temporada', 'Fecha Documento', 
                    'Precio_venta', coste_col, 'margen_%'
                ]].copy()
                
                # Formatear columnas
                tabla_bajo_margen['Fecha Documento'] = pd.to_datetime(tabla_bajo_margen['Fecha Documento']).dt.strftime('%d/%m/%Y')
                tabla_bajo_margen['Precio_venta'] = tabla_bajo_margen['Precio_venta'].round(2)
                tabla_bajo_margen[coste_col] = tabla_bajo_margen[coste_col].round(2)
                tabla_bajo_margen['margen_%'] = (tabla_bajo_margen['margen_%'] * 100).round(1)
                
                # Renombrar columnas para mejor visualización
                tabla_bajo_margen.columns = [
                    'ACT', 'Familia', 'Temporada', 'Fecha Venta', 
                    'Precio Venta (€)', f'{coste_col} (€)', 'Margen %'
                ]
                
                st.markdown(f"**Productos con margen inferior al {umbral_margen*100:.0f}% ({len(tabla_bajo_margen)} productos):**")
                st.dataframe(
                    tabla_bajo_margen,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Estadísticas adicionales con manejo de errores
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Total productos", len(tabla_bajo_margen))
                with col_stats2:
                    margen_promedio_bajo = tabla_bajo_margen['Margen %'].mean()
                    if pd.isna(margen_promedio_bajo) or margen_promedio_bajo == float('inf') or margen_promedio_bajo == float('-inf'):
                        st.metric("Margen promedio", "N/A")
                    else:
                        st.metric("Margen promedio", f"{margen_promedio_bajo:.1f}%")
                with col_stats3:
                    # Calcular pérdida estimada de manera más robusta
                    try:
                        # Solo considerar productos con margen negativo o muy bajo
                        productos_perdida = tabla_bajo_margen[tabla_bajo_margen['Margen %'] < 0].copy()
                        if not productos_perdida.empty:
                            # Usar los nombres de columnas originales para el cálculo
                            coste_col_name = f'{coste_col} (€)'
                            precio_venta_col = 'Precio Venta (€)'
                            perdida_total = ((productos_perdida[coste_col_name] - productos_perdida[precio_venta_col]) * 
                                           abs(productos_perdida['Margen %'] / 100)).sum()
                            if pd.isna(perdida_total) or perdida_total == float('inf') or perdida_total == float('-inf'):
                                st.metric("Pérdida estimada", "N/A")
                            else:
                                st.metric("Pérdida estimada", f"{perdida_total:.0f}€")
                        else:
                            st.metric("Pérdida estimada", "0€")
                    except Exception as e:
                        st.metric("Pérdida estimada", f"Error: {str(e)}")
            else:
                st.info(f"No hay productos con margen inferior al {umbral_margen*100:.0f}%")
        else:
            st.info("No hay datos de Precio Coste, Subtotal o Cantidad disponibles para el análisis de márgenes.")

# Cached function for calculating store rankings
@st.cache_data
def calculate_store_rankings(df_ventas):
    """Cache the store ranking calculations"""
    ventas_por_tienda = df_ventas.groupby('NombreTPV').agg({
        'Cantidad': 'sum',
        'Ventas Dinero': 'sum'
    }).reset_index()
    ventas_por_tienda.columns = ['Tienda', 'Unidades Vendidas', 'Ventas (€)']
    
    # Ordenar por ventas (€) para obtener el ranking
    ventas_por_tienda = ventas_por_tienda.sort_values('Ventas (€)', ascending=False).reset_index(drop=True)
    ventas_por_tienda['Ranking'] = ventas_por_tienda.index + 1
    return ventas_por_tienda

# Cached function for calculating family rankings per store
@st.cache_data
def calculate_family_rankings(df_ventas):
    """Cache the family ranking calculations per store"""
    familias_por_tienda = df_ventas.groupby(['NombreTPV', 'Familia'])['Cantidad'].sum().reset_index()
    familias_por_tienda = familias_por_tienda.sort_values('Cantidad', ascending=False)
    return familias_por_tienda

# Cached function for data preprocessing
@st.cache_data
def preprocess_ventas_data(df_ventas):
    """Cache the data preprocessing to avoid reprocessing on every interaction"""
    df_ventas = df_ventas.copy()
    
    # Asegurarnos que las columnas existen y están en el formato correcto
    df_ventas['Fecha Documento'] = pd.to_datetime(df_ventas['Fecha Documento'], format='%d/%m/%Y', errors='coerce')
    df_ventas = df_ventas.dropna(subset=['Fecha Documento'])

    df_ventas['Mes'] = df_ventas['Fecha Documento'].dt.to_period('M').astype(str)
    df_ventas['Tienda'] = df_ventas['NombreTPV'].astype(str)
    df_ventas['Producto'] = df_ventas['ACT']
    df_ventas['Familia'] = df_ventas['Descripción Familia'].fillna("Sin Familia")
    
    # Asegurar que todas las columnas numéricas están en el formato correcto
    df_ventas['Cantidad'] = pd.to_numeric(df_ventas['Cantidad'], errors='coerce').fillna(0)
    df_ventas['Subtotal'] = pd.to_numeric(df_ventas['Subtotal'], errors='coerce').fillna(0)
    
    # Handle precio_pvp column if it exists
    if 'precio_pvp' in df_ventas.columns:
        df_ventas['precio_pvp'] = pd.to_numeric(df_ventas['precio_pvp'], errors='coerce').fillna(0)
    
    # Calcular ventas en dinero usando Subtotal
    df_ventas['Ventas Dinero'] = df_ventas['Subtotal']
    
    df_ventas['Descripción Color'] = df_ventas.get('Descripción Color', 'Desconocido')
    
    # Asegurar que la columna Temporada existe y está mapeada correctamente
    if 'Temporada' not in df_ventas.columns:
        # Buscar columnas que puedan contener la temporada
        temporada_columns = [col for col in df_ventas.columns if 'temporada' in col.lower() or 'season' in col.lower()]
        if temporada_columns:
            df_ventas['Temporada'] = df_ventas[temporada_columns[0]]
        else:
            # Si no hay columna de temporada, crear una por defecto
            df_ventas['Temporada'] = 'Sin Temporada'
    else:
        # Asegurar que la columna Temporada no tenga valores nulos
        df_ventas['Temporada'] = df_ventas['Temporada'].fillna('Sin Temporada')
    
    # Identificar tiendas online y físicas
    df_ventas['Es_Online'] = df_ventas['NombreTPV'].str.contains('ONLINE', case=False, na=False)
    
    return df_ventas

# Cached function for consistent temporada colors
@st.cache_data
def get_temporada_colors(df_ventas):
    """Get consistent color mapping for temporadas across all charts"""
    temporadas = sorted(df_ventas['Temporada'].unique())
    color_mapping = {}
    for i, temp in enumerate(temporadas):
        color_mapping[temp] = TEMPORADA_COLORS[i % len(TEMPORADA_COLORS)]
    return color_mapping

@st.cache_resource
def load_trained_models():
    """Load the improved trained models and configurations"""
    try:
        # Try to load improved models first
        improved_model_path = 'modelos_mejorados/model_en_robust.pkl'
        if os.path.exists(improved_model_path):
            print("🔧 Loading improved models...")
            
            # Load improved models
            model_en = joblib.load('modelos_mejorados/model_en_robust.pkl')
            model_fuera = joblib.load('modelos_mejorados/model_fuera_robust.pkl')
            
            # Load improved configurations
            with open('modelos_mejorados/features_en_robust.json', 'r') as f:
                config_en = json.load(f)
            
            with open('modelos_mejorados/features_fuera_robust.json', 'r') as f:
                config_fuera = json.load(f)
            
            print("✅ Improved models loaded successfully")
            
            return {
                'model_en': model_en,
                'model_fuera': model_fuera,
                'config_en': config_en,
                'config_fuera': config_fuera,
                'is_improved': True
            }
        else:
            # Fallback to original models
            print("⚠️ Improved models not found, using original models...")
            
            model_en = joblib.load('modelos_finales/model_en_final.pkl')
            model_fuera = joblib.load('modelos_finales/model_fuera_final.pkl')
            
            with open('modelos_finales/features_en_final.json', 'r') as f:
                config_en = json.load(f)
            
            with open('modelos_finales/features_fuera_final.json', 'r') as f:
                config_fuera = json.load(f)
            
            return {
                'model_en': model_en,
                'model_fuera': model_fuera,
                'config_en': config_en,
                'config_fuera': config_fuera,
                'is_improved': False
            }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

@st.cache_resource
def load_training_data():
    """Load and cache the training data"""
    try:
        training_data_path = 'data/datos_modelo_catboost.xlsx'
        if not os.path.exists(training_data_path):
            return None
        
        df_training = pd.read_excel(training_data_path)
        df_training['Fecha Documento'] = pd.to_datetime(df_training['Fecha Documento'], format='%d/%m/%Y', errors='coerce', dayfirst=True)
        return df_training
    except Exception as e:
        st.error(f"Error loading training data: {str(e)}")
        return None

@st.cache_data
def prepare_all_predictions_data(timestamp=None):
    """Prepare all prediction data in advance"""
    try:
        df_training = load_training_data()
        if df_training is None:
            return None, None, None
        
        # Prepare dataset with future months (12 months ahead)
        future_months = []
        current_date = datetime.now()
        for i in range(1, 13):  # 12 months ahead
            future_date = current_date + timedelta(days=30*i)
            future_months.append(future_date.strftime('%Y-%m-01'))
        
        # Prepare dataset with future months
        en_temporada, fuera_temporada, monthly = prepare_final_dataset_improved(
            df_training, meses_futuros=future_months
        )
        
        return monthly, en_temporada, fuera_temporada
        
    except Exception as e:
        st.error(f"Error preparing prediction data: {str(e)}")
        return None, None, None

@st.cache_data
def make_all_predictions(monthly, _models_config, timestamp=None):
    """Make all predictions in advance and cache them"""
    try:
        # Get future data (where we don't have actual values)
        future_data = monthly[
            (monthly['Cantidad_en_temporada'].isna()) | 
            (monthly['Cantidad_fuera_temporada'].isna())
        ].copy()
        
        if future_data.empty:
            return monthly
        
        # Check if we're using improved models
        is_improved = _models_config.get('is_improved', False)
        
        if is_improved:
            # Use improved prediction method with constraints
            print("🔧 Using improved models with constraints...")
            
            # Prepare model info for EN TEMPORADA
            model_info_en = {
                'model': _models_config['model_en'],
                'features': _models_config['config_en']['features'],
                'categorical_features': _models_config['config_en']['cat_features'],
                'prediction_constraints': _models_config['config_en'].get('prediction_constraints', {'min_value': 0, 'max_multiplier': 3})
            }
            
            # Prepare model info for FUERA TEMPORADA
            model_info_fuera = {
                'model': _models_config['model_fuera'],
                'features': _models_config['config_fuera']['features'],
                'categorical_features': _models_config['config_fuera']['cat_features'],
                'prediction_constraints': _models_config['config_fuera'].get('prediction_constraints', {'min_value': 0, 'max_multiplier': 3})
            }
            
            # Make predictions with constraints
            pred_en = predict_with_constraints(model_info_en, future_data)
            pred_fuera = predict_with_constraints(model_info_fuera, future_data)
            
        else:
            # Use original prediction method
            print("⚠️ Using original models...")
            
            # Make EN TEMPORADA predictions
            X_en = future_data[_models_config['config_en']['features']].fillna(0)
            cat_indices_en = [X_en.columns.get_loc(c) for c in _models_config['config_en']['cat_features'] if c in X_en.columns]
            pool_en = Pool(X_en, cat_features=cat_indices_en)
            pred_en = _models_config['model_en'].predict(pool_en)
            
            # Make FUERA TEMPORADA predictions
            X_fuera = future_data[_models_config['config_fuera']['features']].fillna(0)
            cat_indices_fuera = [X_fuera.columns.get_loc(c) for c in _models_config['config_fuera']['cat_features'] if c in X_fuera.columns]
            pool_fuera = Pool(X_fuera, cat_features=cat_indices_fuera)
            pred_fuera = _models_config['model_fuera'].predict(pool_fuera)
        
        # Update the monthly dataframe with predictions
        monthly_pred = monthly.copy()
        for i, idx in enumerate(future_data.index):
            monthly_pred.loc[idx, 'Pred_Cantidad_en_temporada'] = pred_en[i]
            monthly_pred.loc[idx, 'Pred_Cantidad_fuera_temporada'] = pred_fuera[i]
            monthly_pred.loc[idx, 'Pred_Cantidad_Total'] = pred_en[i] + pred_fuera[i]
        
        return monthly_pred
        
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return monthly

def create_prediction_data(df_ventas, selected_stores, selected_families, selected_sizes, months_ahead):
    """Create prediction data for the selected parameters"""
    try:
        # Prepare the dataset with future months
        future_months = []
        current_date = datetime.now()
        for i in range(1, months_ahead + 1):
            future_date = current_date + timedelta(days=30*i)
            future_months.append(future_date.strftime('%Y-%m-01'))
        
        # Prepare dataset with future months
        en_temporada, fuera_temporada, monthly = prepare_final_dataset_improved(
            df_ventas, meses_futuros=future_months
        )
        
        # Filter for selected stores, families, and sizes
        if selected_stores:
            monthly = monthly[monthly['Tienda'].isin(selected_stores)]
        if selected_families:
            monthly = monthly[monthly['Descripción Familia'].isin(selected_families)]
        if selected_sizes:
            monthly = monthly[monthly['Talla'].isin(selected_sizes)]
        
        return monthly, en_temporada, fuera_temporada
        
    except Exception as e:
        st.error(f"Error creating prediction data: {str(e)}")
        return None, None, None

def show_prediction_interface(df_ventas):
    """Show the prediction interface"""
    st.markdown("---")
    st.markdown("## 🔮 **Predicciones de Ventas**")
    st.markdown("Utiliza los modelos entrenados para predecir ventas futuras")
    
    # Clear cache button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Regenerar"):
            st.session_state.models_loaded = False
            st.session_state.monthly_predictions = None
            st.session_state.df_training = None
            st.rerun()
    with col2:
        st.info("💡 Si no ves predicciones, haz clic en 'Regenerar' para limpiar el caché")
    
    # Check if models are already loaded in session state
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    
    if 'monthly_predictions' not in st.session_state:
        st.session_state.monthly_predictions = None
    
    # Show loading progress
    if not st.session_state.models_loaded:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Load models (20%)
        status_text.text("Cargando modelos...")
        models_config = load_trained_models()
        if models_config is None:
            st.error("No se pudieron cargar los modelos.")
            return
        
        # Show which models are being used
        if models_config.get('is_improved', False):
            st.success("🚀 **Modelos Mejorados Cargados** - Con validación cruzada y restricciones anti-overfitting")
        else:
            st.warning("⚠️ **Modelos Originales Cargados** - Considera ejecutar `python run_model_improved.py` para mejores predicciones")
        
        progress_bar.progress(20)
        
        # Step 2: Load training data (40%)
        status_text.text("Cargando datos de entrenamiento...")
        df_training = load_training_data()
        if df_training is None:
            st.error("No se pudieron cargar los datos de entrenamiento.")
            return
        progress_bar.progress(40)
        
        # Step 3: Prepare prediction data (70%)
        status_text.text("Preparando datos para predicciones...")
        monthly_all, en_temporada, fuera_temporada = prepare_all_predictions_data(timestamp=datetime.now())
        if monthly_all is None:
            st.error("No se pudieron preparar los datos de predicción.")
            return
        progress_bar.progress(70)
        
        # Step 4: Make predictions (100%)
        status_text.text("Generando predicciones...")
        monthly_pred_all = make_all_predictions(monthly_all, models_config, timestamp=datetime.now())
        if monthly_pred_all is None:
            st.error("No se pudieron generar las predicciones.")
            return
        progress_bar.progress(100)
        
        # Store in session state
        st.session_state.models_loaded = True
        st.session_state.monthly_predictions = monthly_pred_all
        st.session_state.df_training = df_training
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ Modelos y predicciones cargados exitosamente!")
    else:
        # Use cached data
        monthly_pred_all = st.session_state.monthly_predictions
        df_training = st.session_state.df_training
        st.success("✅ Datos cargados desde caché")
    
    # Get all available options
    all_stores = sorted(df_training['NombreTPV'].unique())
    
    # Simple interface - just select months to predict
    col1, col2 = st.columns(2)
    
    with col1:
        months_ahead = st.slider("Meses a predecir", 1, 12, 6)
    
    with col2:
        # Store grouping options
        store_grouping = st.selectbox(
            "Agrupar tiendas por:",
            ["Todas las tiendas", "Por rendimiento", "Por zona geográfica", "Tiendas principales"]
        )
    
    # Filter predictions based on selection
    if store_grouping == "Todas las tiendas":
        selected_stores = all_stores
    elif store_grouping == "Por rendimiento":
        # Get top performing stores
        store_performance = df_training.groupby('NombreTPV')['Cantidad'].sum().sort_values(ascending=False)
        selected_stores = store_performance.head(10).index.tolist()
    elif store_grouping == "Por zona geográfica":
        # Group by geographic zones (you can customize this)
        madrid_stores = [s for s in all_stores if 'MADRID' in s.upper()]
        barcelona_stores = [s for s in all_stores if 'BARCELONA' in s.upper()]
        other_stores = [s for s in all_stores if 'MADRID' not in s.upper() and 'BARCELONA' not in s.upper()]
        selected_stores = madrid_stores + barcelona_stores + other_stores[:5]
    else:  # Tiendas principales
        selected_stores = all_stores[:15]  # Top 15 stores
    
    # Filter the predictions
    future_pred = monthly_pred_all[
        (monthly_pred_all['Pred_Cantidad_Total'].notna()) &
        (monthly_pred_all['Tienda'].isin(selected_stores))
    ].copy()
    
    # Debug information
    st.info(f"📊 Total predictions available: {len(monthly_pred_all[monthly_pred_all['Pred_Cantidad_Total'].notna()])}")
    st.info(f"📊 Predictions after store filtering: {len(future_pred)}")
    st.info(f"📊 Selected stores: {len(selected_stores)}")
    
    # Filter by months ahead - use the actual future dates from the data
    future_dates_in_data = pd.to_datetime(future_pred['ds']).dt.strftime('%Y-%m-01').unique()
    future_dates_in_data = sorted(future_dates_in_data)
    
    # Take only the first N months as requested
    if len(future_dates_in_data) > months_ahead:
        selected_future_dates = future_dates_in_data[:months_ahead]
    else:
        selected_future_dates = future_dates_in_data
    
    future_pred = future_pred[
        pd.to_datetime(future_pred['ds']).dt.strftime('%Y-%m-01').isin(selected_future_dates)
    ]
    
    # Debug information for date filtering
    st.info(f"📅 Available future dates: {future_dates_in_data}")
    st.info(f"📅 Selected future dates: {selected_future_dates}")
    st.info(f"📊 Predictions after date filtering: {len(future_pred)}")
    
    if future_pred.empty:
        st.warning("No hay predicciones disponibles para los filtros seleccionados.")
        st.info("💡 Try selecting fewer months or different store grouping options.")
        return
    
    # Show results immediately
    show_prediction_results(future_pred, months_ahead)

def show_prediction_results(monthly_pred, months_ahead):
    """Show prediction results"""
    st.success("✅ Predicciones generadas exitosamente!")
    
    # Filter for future predictions
    future_pred = monthly_pred[
        monthly_pred['Pred_Cantidad_Total'].notna()
    ].copy()
    
    if future_pred.empty:
        st.warning("No hay predicciones futuras disponibles")
        return
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_pred_en = future_pred['Pred_Cantidad_en_temporada'].sum()
        st.metric("Total EN Temporada", f"{total_pred_en:,.0f}")
    
    with col2:
        total_pred_fuera = future_pred['Pred_Cantidad_fuera_temporada'].sum()
        st.metric("Total FUERA Temporada", f"{total_pred_fuera:,.0f}")
    
    with col3:
        total_pred = future_pred['Pred_Cantidad_Total'].sum()
        st.metric("Total Predicción", f"{total_pred:,.0f}")
    
    with col4:
        avg_pred = future_pred['Pred_Cantidad_Total'].mean()
        st.metric("Promedio por Mes", f"{avg_pred:,.0f}")
    
    # Prediction charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly predictions
        monthly_summary = future_pred.groupby('ds').agg({
            'Pred_Cantidad_en_temporada': 'sum',
            'Pred_Cantidad_fuera_temporada': 'sum',
            'Pred_Cantidad_Total': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_summary['ds'],
            y=monthly_summary['Pred_Cantidad_en_temporada'],
            name='EN Temporada',
            marker_color='#0066cc'
        ))
        fig.add_trace(go.Bar(
            x=monthly_summary['ds'],
            y=monthly_summary['Pred_Cantidad_fuera_temporada'],
            name='FUERA Temporada',
            marker_color='#ff9900'
        ))
        
        fig.update_layout(
            title="Predicciones Mensuales",
            xaxis_title="Mes",
            yaxis_title="Cantidad Predicha",
            barmode='stack',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Store predictions
        store_summary = future_pred.groupby('Tienda').agg({
            'Pred_Cantidad_Total': 'sum'
        }).reset_index().sort_values('Pred_Cantidad_Total', ascending=True)
        
        fig = px.bar(
            store_summary,
            x='Pred_Cantidad_Total',
            y='Tienda',
            orientation='h',
            title="Predicciones por Tienda",
            color='Pred_Cantidad_Total',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed predictions table
    st.markdown("### 📊 Detalle de Predicciones")
    
    # Format the table
    display_pred = future_pred[['Tienda', 'Descripción Familia', 'Talla', 'ds', 
                               'Pred_Cantidad_en_temporada', 'Pred_Cantidad_fuera_temporada', 
                               'Pred_Cantidad_Total']].copy()
    
    display_pred['ds'] = pd.to_datetime(display_pred['ds']).dt.strftime('%Y-%m')
    display_pred = display_pred.round(2)
    
    st.dataframe(
        display_pred,
        use_container_width=True,
        hide_index=True
    )
    
    # Download predictions
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        display_pred.to_excel(writer, sheet_name='Predicciones', index=False)
    excel_data = output.getvalue()
    
    st.download_button(
        label="📥 Descargar Predicciones (Excel)",
        data=excel_data,
        file_name=f"predicciones_ventas_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    # Comparison graph section
    st.markdown("---")
    st.markdown("### 📈 Comparación Real vs Predicción por Familia")
    
    # Get available families
    available_families = sorted(future_pred['Descripción Familia'].unique())
    
    if available_families:
        # Family selector
        selected_family = st.selectbox(
            "Selecciona una familia para comparar:",
            available_families,
            index=0
        )
        
        # Season selector
        selected_season = st.selectbox(
            "Selecciona la temporada:",
            ["Verano (Marzo-Agosto)", "Invierno (Septiembre-Marzo)"],
            index=0
        )
        
        # Create comparison graph
        create_comparison_graph(future_pred, selected_family, selected_season)
    else:
        st.warning("No hay familias disponibles para comparar.")


def create_comparison_graph(future_pred, selected_family, selected_season):
    """Create comparison graph between real data and predictions for a specific family and season"""
    try:
        # Get training data for historical comparison
        df_training = load_training_data()
        if df_training is None:
            st.error("No se pudieron cargar los datos de entrenamiento para la comparación.")
            return
        
        # Define season months
        if selected_season == "Verano (Marzo-Agosto)":
            season_months = [3, 4, 5, 6, 7, 8]  # Marzo a Agosto
            season_name = "Verano"
        else:  # Invierno
            season_months = [9, 10, 11, 12, 1, 2]  # Septiembre a Marzo
            season_name = "Invierno"
        
        # Prepare historical data (real data) - use correct column name
        df_training['Fecha Documento'] = pd.to_datetime(df_training['Fecha Documento'], format='%d/%m/%Y', errors='coerce', dayfirst=True)
        df_training['año'] = df_training['Fecha Documento'].dt.year
        df_training['mes'] = df_training['Fecha Documento'].dt.month
        
        # Filter historical data for the selected family and season
        historical_data = df_training[
            (df_training['Descripción Familia'] == selected_family) &
            (df_training['mes'].isin(season_months)) &
            (df_training['año'].isin([2023, 2024, 2025]))
        ].copy()
        
        if historical_data.empty:
            st.warning(f"No hay datos históricos disponibles para {selected_family} en {season_name}.")
            return
        
        # Group historical data by year and month
        historical_grouped = historical_data.groupby(['año', 'mes']).agg({
            'Cantidad': 'sum'
        }).reset_index()
        historical_grouped['Fuente'] = 'Real'
        historical_grouped.rename(columns={'Cantidad': 'Cantidad_en_temporada'}, inplace=True)
        
        # Prepare prediction data
        future_pred['Fecha'] = pd.to_datetime(future_pred['ds'])
        future_pred['año'] = future_pred['Fecha'].dt.year
        future_pred['mes'] = future_pred['Fecha'].dt.month
        
        # Filter prediction data for the selected family and season
        prediction_data = future_pred[
            (future_pred['Descripción Familia'] == selected_family) &
            (future_pred['mes'].isin(season_months))
        ].copy()
        
        if prediction_data.empty:
            st.warning(f"No hay predicciones disponibles para {selected_family} en {season_name}.")
            return
        
        # Group prediction data by year and month
        prediction_grouped = prediction_data.groupby(['año', 'mes']).agg({
            'Pred_Cantidad_en_temporada': 'sum'
        }).reset_index()
        prediction_grouped['Fuente'] = 'Predicho'
        prediction_grouped.rename(columns={'Pred_Cantidad_en_temporada': 'Cantidad_en_temporada'}, inplace=True)
        
        # Combine historical and prediction data
        df_total = pd.concat([historical_grouped, prediction_grouped], ignore_index=True)
        
        # Create year-source column for better legend
        df_total['Año_Fuente'] = df_total['año'].astype(str) + ' - ' + df_total['Fuente']
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot
        sns.barplot(
            data=df_total, 
            x='mes', 
            y='Cantidad_en_temporada', 
            hue='Año_Fuente', 
            palette='tab10',
            ax=ax
        )
        
        # Customize the plot
        ax.set_title(f'Comparación Real vs Predicción - {selected_family} ({season_name})', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Mes', fontsize=12)
        ax.set_ylabel('Cantidad en Temporada', fontsize=12)
        
        # Set month labels
        if selected_season == "Verano (Marzo-Agosto)":
            month_labels = ['Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago']
            month_ticks = [3, 4, 5, 6, 7, 8]
        else:
            month_labels = ['Sep', 'Oct', 'Nov', 'Dic', 'Ene', 'Feb']
            month_ticks = [9, 10, 11, 12, 1, 2]
        
        ax.set_xticks(range(len(month_ticks)))
        ax.set_xticklabels(month_labels)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Customize legend
        ax.legend(title='Año - Fuente', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Display the plot
        st.pyplot(fig)
        
        # Add summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_real = historical_grouped['Cantidad_en_temporada'].sum()
            st.metric("Total Real", f"{total_real:,.0f}")
        
        with col2:
            total_pred = prediction_grouped['Cantidad_en_temporada'].sum()
            st.metric("Total Predicción", f"{total_pred:,.0f}")
        
        with col3:
            if total_real > 0:
                growth_pct = ((total_pred - total_real) / total_real) * 100
                st.metric("Crecimiento %", f"{growth_pct:+.1f}%")
            else:
                st.metric("Crecimiento %", "N/A")
        
    except Exception as e:
        st.error(f"Error al crear el gráfico de comparación: {str(e)}")
        st.error("Detalles del error: " + str(e.__class__.__name__))

def predict_with_constraints(model_info, X_new):
    """
    Make predictions with proper constraints (for improved models)
    """
    model = model_info['model']
    features = model_info['features']
    cat_features = model_info['categorical_features']
    constraints = model_info.get('prediction_constraints', {'min_value': 0, 'max_multiplier': 3})
    
    # Prepare data
    X_prepared = X_new[features].fillna(0)
    cat_indices = [X_prepared.columns.get_loc(c) for c in cat_features if c in X_prepared.columns]
    pool = Pool(X_prepared, cat_features=cat_indices)
    
    # Make predictions
    predictions = model.predict(pool)
    
    # Apply constraints
    predictions = np.maximum(predictions, constraints['min_value'])
    
    # Cap extreme predictions
    if 'max_multiplier' in constraints:
        max_observed = X_new.get('Cantidad', 0).max() if 'Cantidad' in X_new.columns else 1000
        predictions = np.minimum(predictions, max_observed * constraints['max_multiplier'])
    
    return predictions


