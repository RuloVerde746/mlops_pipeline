# -*- coding: utf-8 -*-
"""
streamlit_app.py - Dashboard Interactivo de Monitoreo del Modelo

Este dashboard muestra los resultados del monitoreo de data drift
generados por model_monitoring.py de forma interactiva.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import warnings
import json
warnings.filterwarnings('ignore')

def format_metric(value, format_str=".4f"):
    """
    Función helper para formatear métricas de forma segura.
    """
    try:
        # Limpiar el formato por si viene con ':'
        fmt = format_str.replace(":", "").strip()
        
        # Manejar NaNs o valores nulos
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "N/A"
            
        if isinstance(value, (int, float, np.number)):
            return format(value, fmt)
        else:
            return str(value)
    except:
        return str(value)

def get_column_value(data, key, default=0):
    """
    Función helper para manejar diferentes tipos de datos en columnas del dashboard.
    Soporta int, list, str y otros tipos con manejo seguro de errores.
    """
    try:
        if key not in data:
            return default
            
        value = data[key]
        
        # Si es un entero, retornar directamente
        if isinstance(value, int):
            return value
            
        # Si es una lista, retornar su longitud
        elif isinstance(value, list):
            return len(value)
            
        # Si es un string, intentar convertir a entero
        elif isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                # Si no se puede convertir, retornar longitud del string
                return len(value)
                
        # Si es otro tipo (float, None, etc.)
        else:
            try:
                # Intentar convertir a entero
                return int(value)
            except (ValueError, TypeError):
                # Si falla, retornar default
                return default
                
    except Exception:
        return default

# Configuración de la página
st.set_page_config(
    page_title="📊 Dashboard Monitoreo MLOps",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Dashboard de Monitoreo del Modelo de Crédito")
st.markdown("---")

# Sidebar para información general
st.sidebar.header("🎯 Información General")

# Toggle de Depuración
DEBUG_MODE = st.sidebar.checkbox("🔍 Habilitar Modo Depuración", value=False)
if DEBUG_MODE:
    st.sidebar.info("Modo depuración activo")

# Cargar datos del monitoreo
@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_monitoring_data():
    """Carga los datos de monitoreo generados por model_monitoring.py"""
    try:
        path = 'assets/streamlit_dashboard_data.pkl'
        if os.path.exists(path):
            data = joblib.load(path)
            # LOG DE CONSOLA PERMANENTE
            print(f"\n[DASHBOARD LOG] Datos cargados exitosamente")
            print(f"[DASHBOARD LOG] Timestamp data: {data.get('timestamp')}")
            print(f"[DASHBOARD LOG] Variables en resultados: {len(data.get('monitoring_results', {}))}")
            
            if DEBUG_MODE:
                st.sidebar.markdown("---")
                st.sidebar.subheader("🛠️ Debug Information")
                st.sidebar.write(f"📂 Archivo cargado: {path}")
                st.sidebar.write(f"📊 Tipo de data: {type(data)}")
                st.sidebar.write(f"📅 Timestamp: {data.get('timestamp', 'N/A')}")
                
                # Resumen de tipos en summary
                if 'summary' in data:
                    st.sidebar.write("📈 Tipos en Summary:")
                    for k, v in data['summary'].items():
                        st.sidebar.write(f"- {k}: {type(v)}")
            return data
        else:
            st.error("❌ No se encontraron datos de monitoreo. Ejecuta model_monitoring.py primero.")
            return None
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {str(e)}")
        if DEBUG_MODE:
            st.exception(e)
        return None

# Cargar datos
dashboard_data = load_monitoring_data()

if dashboard_data:
    # Métricas principales en el sidebar
    st.sidebar.metric(
        "📅 Última Actualización", 
        datetime.fromisoformat(dashboard_data['timestamp']).strftime("%Y-%m-%d %H:%M")
    )
    st.sidebar.metric("📊 Variables Totales", get_column_value(dashboard_data['summary'], 'total_variables'))
    st.sidebar.metric("🚨 Alertas Críticas", get_column_value(dashboard_data['summary'], 'critical_alerts'))
    st.sidebar.metric("⚠️ Alertas Advertencia", get_column_value(dashboard_data['summary'], 'warning_alerts'))
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Resumen", "🚨 Alertas", "📊 Variables", "📋 Reporte"])
    
    with tab1:
        st.header("📈 Resumen General")
        
        # KPIs principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔥 Variables con Drift Crítico", 
                get_column_value(dashboard_data['summary'], 'critical_alerts'),
                delta=f"{get_column_value(dashboard_data['summary'], 'critical_alerts')} variables"
            )
        
        with col2:
            st.metric(
                "⚠️ Variables con Advertencia", 
                get_column_value(dashboard_data['summary'], 'warning_alerts'),
                delta=f"{get_column_value(dashboard_data['summary'], 'warning_alerts')} variables"
            )
        
        with col3:
            st.metric(
                "✅ Variables Normales", 
                get_column_value(dashboard_data['summary'], 'normal_alerts'),
                delta=f"{get_column_value(dashboard_data['summary'], 'normal_alerts')} variables"
            )
        
        with col4:
            health_score = get_column_value(dashboard_data, 'health_score', 100)
            st.metric(
                "🏥 Salud del Modelo", 
                f"{health_score}%",
                delta=f"{health_score}%" if health_score > 70 else None,
                delta_color="normal" if health_score > 70 else "inverse"
            )
        
        # Gráfico de distribución de alertas
        st.subheader("📊 Distribución de Alertas")
        
        alert_data = {
            'Críticas': get_column_value(dashboard_data['summary'], 'critical_alerts'),
            'Advertencia': get_column_value(dashboard_data['summary'], 'warning_alerts'),
            'Normales': get_column_value(dashboard_data['summary'], 'normal_alerts')
        }
        
        fig_alerts = px.pie(
            values=list(alert_data.values()),
            names=list(alert_data.keys()),
            title="Distribución de Variables por Estado",
            color_discrete_map={
                'Críticas': 'red',
                'Advertencia': 'orange', 
                'Normales': 'green'
            }
        )
        st.plotly_chart(fig_alerts, width='stretch')
    
    with tab2:
        st.header("🚨 Alertas Detalladas")
        
        # Validación de datos críticos en modo debug
        if DEBUG_MODE:
            st.info("🔍 Validación de llaves para Alertas:")
            vars_sample = list(dashboard_data['monitoring_results'].values())
            if vars_sample:
                keys = vars_sample[0].keys()
                st.write(f"Llaves disponibles en variable: {list(keys)}")
                if 'ks_stat' not in keys and 'ks_statistic' in keys:
                    st.warning("⚠️ Se detectó 'ks_statistic' en lugar de 'ks_stat'. El dashboard podría mostrar N/A.")
        
        # Alertas críticas
        critical_count = get_column_value(dashboard_data['summary'], 'critical_alerts')
        if critical_count > 0:
            st.subheader("🔥 Alertas Críticas - Requieren Atención Inmediata")
            
            # Extraer lista de variables críticas desde monitoring_results
            critical_vars = [v for v in dashboard_data['monitoring_results'].values() 
                           if v.get('alert_level') == 'CRITICAL']
            
            if critical_vars:
                critical_df = pd.DataFrame(critical_vars)
                # LOG DE CONSOLA PERMANENTE
                print(f"\n[DASHBOARD] Procesando {len(critical_df)} alertas críticas")
                
                for idx, row in critical_df.iterrows():
                    var_name = row.get('variable')
                    if var_name is None or str(var_name) in ['nan', 'None', 'N/A']:
                        var_name = f"Variable {idx}"
                    
                    # Log detallado por terminal
                    print(f"  🔥 Crítica det.: {var_name} | PSI: {row.get('psi')} | KS: {row.get('ks_stat')}")
                    
                    with st.expander(f"🔥 Variable: {var_name}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            ks_val = row.get('ks_stat') or row.get('ks_statistic') or 'N/A'
                            st.metric("Métrica PSI", format_metric(row.get('psi', 'N/A')))
                            st.metric("Métrica KS", format_metric(ks_val))
                        with col2:
                            st.metric("Umbral PSI", row.get('psi_threshold', 'N/A'))
                            st.metric("Umbral KS", row.get('ks_threshold', 'N/A'))
                        
                        # Mostrar imagen si existe
                        img_path = f"assets/images/drift_plot_{row.get('variable', 'N/A')}.png"
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"Drift Plot - {row.get('variable', 'N/A')}")
        else:
            st.success("✅ No hay alertas críticas")
        
        # Alertas de advertencia
        warning_count = get_column_value(dashboard_data['summary'], 'warning_alerts')
        if warning_count > 0:
            st.subheader("⚠️ Alertas de Advertencia - Monitorear")
            
            # Extraer lista de variables de advertencia desde monitoring_results
            warning_vars = [v for v in dashboard_data['monitoring_results'].values() 
                          if v.get('alert_level') == 'WARNING']
            
            if warning_vars:
                warning_df = pd.DataFrame(warning_vars)
                # LOG DE CONSOLA PERMANENTE
                print(f"[DASHBOARD] Procesando {len(warning_df)} alertas de advertencia")
                
                for idx, row in warning_df.iterrows():
                    var_name = row.get('variable')
                    if var_name is None or str(var_name) in ['nan', 'None', 'N/A']:
                        var_name = f"Variable {idx}"
                    
                    # Log detallado por terminal
                    print(f"  ⚠️ Advert. det.: {var_name} | PSI: {row.get('psi')}")
                        
                    with st.expander(f"⚠️ Variable: {var_name}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            ks_val = row.get('ks_stat') or row.get('ks_statistic') or 'N/A'
                            st.metric("Métrica PSI", format_metric(row.get('psi', 'N/A')))
                            st.metric("Métrica KS", format_metric(ks_val))
                        with col2:
                            st.metric("Umbral PSI", row.get('psi_threshold', 'N/A'))
                            st.metric("Umbral KS", row.get('ks_threshold', 'N/A'))
        else:
            st.info("ℹ️ No hay alertas de advertencia")
    
    with tab3:
        st.header("📊 Análisis por Variable")
        
        # Tabla de todas las variables
        all_variables = []
        
        # Obtener lista de variables con alertas críticas
        critical_vars = [v for v in dashboard_data['monitoring_results'].values() 
                       if v.get('alert_level') == 'CRITICAL']
        
        for var in critical_vars:
            all_variables.append({
                'Variable': var.get('variable', 'N/A'),
                'Estado': 'Crítico',
                'PSI': var.get('psi', 0),
                'KS': var.get('ks_stat', 0),
                'Recomendación': 'Reentrenar modelo'
            })
        
        # Extraer lista de variables con advertencia desde monitoring_results
        warning_vars = [v for v in dashboard_data['monitoring_results'].values() 
                      if v.get('alert_level') == 'WARNING']
        
        for var in warning_vars:
            all_variables.append({
                'Variable': var.get('variable', 'N/A'),
                'Estado': 'Advertencia',
                'PSI': var.get('psi', 0),
                'KS': var.get('ks_stat', 0),
                'Recomendación': 'Monitorear closely'
            })
        
        # Extraer lista de variables normales desde monitoring_results
        normal_vars = [v for v in dashboard_data['monitoring_results'].values() 
                     if v.get('alert_level') == 'NORMAL']
        
        for var in normal_vars:
            all_variables.append({
                'Variable': var.get('variable', 'N/A'),
                'Estado': 'Normal',
                'PSI': var.get('psi', 0),
                'KS': var.get('ks_stat', 0),
                'Recomendación': 'Continuar monitoreo'
            })
        
        if all_variables:
            variables_df = pd.DataFrame(all_variables)
            
            # Filtros
            col1, col2 = st.columns(2)
            with col1:
                estado_filter = st.selectbox("Filtrar por Estado", 
                                           ['Todos', 'Crítico', 'Advertencia', 'Normal'])
            with col2:
                sort_by = st.selectbox("Ordenar por", ['Variable', 'PSI', 'KS'])
            
            # Aplicar filtros
            if estado_filter != 'Todos':
                variables_df = variables_df[variables_df['Estado'] == estado_filter]
            
            variables_df = variables_df.sort_values(sort_by, ascending=False)
            
            # Mostrar tabla
            st.dataframe(variables_df, width='stretch')
            
            # Gráfico de PSI vs KS
            st.subheader("📈 PSI vs KS por Variable")
            fig_scatter = px.scatter(
                variables_df, 
                x='PSI', 
                y='KS', 
                color='Estado',
                hover_data=['Variable', 'Recomendación'],
                title="Métricas de Drift por Variable"
            )
            st.plotly_chart(fig_scatter, width='stretch')
    
    with tab4:
        st.header("📋 Reporte Completo")
        
        # Información del reporte
        st.subheader("📊 Información del Reporte")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 Fecha de Generación", 
                     datetime.fromisoformat(dashboard_data['timestamp']).strftime("%Y-%m-%d %H:%M:%S"))
        with col2:
            st.metric("📁 Archivo de Datos", "streamlit_dashboard_data.pkl")
        with col3:
            st.metric("📄 Reporte HTML", "assets/drift_report.html")
        
        # Botón para descargar reporte HTML
        if os.path.exists('assets/drift_report.html'):
            with open('assets/drift_report.html', 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.download_button(
                label="📥 Descargar Reporte HTML",
                data=html_content,
                file_name="drift_report.html",
                mime="text/html"
            )
        
        # Recomendaciones automáticas
        st.subheader("💡 Recomendaciones Automáticas")
        
        if get_column_value(dashboard_data['summary'], 'critical_alerts') > 0:
            st.error("🔥 **ACCIONES REQUERIDAS:**")
            st.write("• Reentrenar el modelo con datos más recientes")
            st.write("• Investigar las causas del drift en variables críticas")
            st.write("• Considerar recolección de datos adicional")
        
        if get_column_value(dashboard_data['summary'], 'warning_alerts') > 0:
            st.warning("⚠️ **MONITOREO CERCANO:**")
            st.write("• Incrementar frecuencia de monitoreo")
            st.write("• Preparar plan de reentrenamiento")
            st.write("• Investigar tendencias en variables con advertencia")
        
        if get_column_value(dashboard_data, 'health_score', 100) > 80:
            st.success("✅ **MODELO SALUDABLE:**")
            st.write("• Continuar monitoreo regular")
            st.write("• Mantener programa de monitoreo actual")
        
        # Botón para actualizar datos
        if st.button("🔄 Actualizar Datos de Monitoreo"):
            st.cache_data.clear()
            st.rerun()

else:
    st.error("❌ No se pueden cargar los datos de monitoreo")
    st.info("💡 **Solución:** Ejecuta el siguiente comando:")
    st.code("python src/model_monitoring.py", language="bash")
    
    st.info("📋 **Requisitos:**")
    st.write("• Asegúrate que el archivo `assets/streamlit_dashboard_data.pkl` exista")
    st.write("• Verifica que el monitoreo se haya ejecutado correctamente")
    st.write("• Revisa que los archivos de reporte estén en la carpeta `assets/`")

# Footer
st.markdown("---")
st.markdown("🤖 **Dashboard MLOps Pipeline** | Monitoreo Continuo de Modelos de Machine Learning")

# Sección de Inspección de Datos (Solo en DEBUG_MODE)
if DEBUG_MODE and dashboard_data:
    st.markdown("---")
    with st.expander("🛠️ INSPECCIÓN DE DATOS CRUDOS (DEBUG)", expanded=False):
        st.json(dashboard_data)
        
        st.subheader("🔍 Estructura de monitoring_results")
        for var, details in dashboard_data.get('monitoring_results', {}).items():
            st.write(f"**Variable:** {var}")
            st.write(f"  • Alert Level: {details.get('alert_level')} ({type(details.get('alert_level'))})")
            st.write(f"  • PSI: {details.get('psi')} ({type(details.get('psi'))})")
            ks_val = details.get('ks_stat') or details.get('ks_statistic')
            st.write(f"  • KS: {ks_val} ({type(ks_val)})")
            if ks_val is None:
                st.error(f"❌ Falta métrica KS para {var}")

# Auto-refresh
refresh_interval = st.sidebar.slider("🔄 Auto-refresh (segundos)", 30, 300, 60)
if st.sidebar.button("🔄 Forzar Actualización"):
    st.cache_data.clear()
    st.rerun()

# Nota sobre auto-refresh
st.sidebar.caption(f"El dashboard se actualizará automáticamente cada {refresh_interval} segundos")
