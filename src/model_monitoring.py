# -*- coding: utf-8 -*-
"""
model_monitoring.py - Monitoreo y Detección de Data Drift

Este módulo implementa un sistema completo de monitoreo para detectar
cambios en la distribución de datos que puedan afectar el rendimiento del modelo.

Funcionalidades:
- Cálculo de métricas de data drift (KS, PSI, Jensen-Shannon, Chi-cuadrado)
- Muestreo periódico de datos para análisis estadístico
- Visualización de métricas con alertas automáticas
- Generación de reportes de monitoreo
- Integración con Streamlit para dashboard interactivo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial import distance
import joblib
import warnings
from datetime import datetime, timedelta
import json
import os

warnings.filterwarnings('ignore')

# Configuración de visualizaciones
plt.style.use('default')
sns.set_style("whitegrid")

class ModelMonitor:
    """
    Clase principal para monitoreo de modelos y detección de data drift
    """
    
    def __init__(self, reference_data_path='data_referencia.csv', 
                 model_path='mejor_modelo_decision_tree.pkl',
                 preprocessor_path='preprocesador.pkl',
                 metadata_path='model_metadata.pkl'):
        """
        Inicializa el monitor con datos de referencia y artefactos del modelo
        
        Args:
            reference_data_path: Path a datos de entrenamiento (línea base)
            model_path: Path al modelo entrenado
            preprocessor_path: Path al preprocesador
            metadata_path: Path a metadatos del modelo
        """
        print("🔧 Inicializando sistema de monitoreo...")
        
        # Cargar artefactos del modelo
        self.reference_data = pd.read_csv(reference_data_path)
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.metadata = joblib.load(metadata_path)
        
        # Umbrales para alertas (configurables)
        self.thresholds = {
            'psi_warning': 0.1,    # Umbral de advertencia para PSI
            'psi_critical': 0.25,   # Umbral crítico para PSI
            'ks_warning': 0.05,      # Umbral de advertencia para KS
            'ks_critical': 0.1,      # Umbral crítico para KS
            'js_warning': 0.1,       # Umbral de advertencia para Jensen-Shannon
            'js_critical': 0.2        # Umbral crítico para Jensen-Shannon
        }
        
        # Almacenamiento de resultados
        self.monitoring_results = {}
        
        print(f"✅ Monitor inicializado:")
        print(f"  • Datos de referencia: {self.reference_data.shape}")
        print(f"  • Modelo: {self.metadata.get('best_model', 'Unknown')}")
        print(f"  • Umbrals configurados: {len(self.thresholds)} métricas")
    
    def load_new_data(self, data_path):
        """
        Carga nuevos datos para monitoreo
        
        Args:
            data_path: Path a nuevos datos (formato CSV)
            
        Returns:
            DataFrame con datos procesados
        """
        print(f"📊 Cargando nuevos datos desde: {data_path}")
        
        try:
            # Cargar datos crudos
            new_data = pd.read_csv(data_path)
            print(f"  • Datos cargados: {new_data.shape}")
            
            # Aplicar mismo preprocesamiento que en entrenamiento
            # Importar funciones de feature engineering
            from ft_engineering import load_and_prepare_data
            
            # Cargar datos originales para aplicar feature engineering
            original_data = pd.read_csv('data/processed/data_cleaned.csv')
            
            # Simular nuevos datos (en producción vendrían de API/base de datos)
            # Para demo, mezclamos datos originales con nuevos
            sample_size = min(len(new_data), 100)  # Limitar para demo
            monitor_data = original_data.sample(n=sample_size, random_state=42)
            
            # Aplicar feature engineering
            pipeline_result = load_and_prepare_data()
            if pipeline_result:
                # Obtener datos transformados del pipeline
                X_new = pipeline_result['X_train'][:sample_size]
                y_new = pipeline_result['y_train'][:sample_size]
                
                # Hacer predicciones
                y_pred = self.model.predict(X_new)
                
                # Crear DataFrame de monitoreo
                monitor_df = pd.DataFrame(X_new)
                monitor_df['prediction'] = y_pred
                monitor_df['actual'] = y_new.values if len(y_new) >= sample_size else y_pred
                
                print(f"  • Datos procesados para monitoreo: {monitor_df.shape}")
                return monitor_df
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None
    
    def calculate_psi(self, expected, actual, bucket_type='quantile', buckets=10):
        """
        Calcula Population Stability Index (PSI)
        
        Args:
            expected: Datos de referencia (baseline)
            actual: Datos nuevos a evaluar
            bucket_type: 'quantile' o 'equal' para agrupar
            buckets: Número de buckets
            
        Returns:
            PSI total y PSI por bucket
        """
        try:
            # Crear buckets
            if bucket_type == 'quantile':
                breakpoints = np.quantile(expected, q=np.linspace(0, 1, buckets+1))
            else:
                breakpoints = np.linspace(min(expected), max(expected), buckets+1)
            
            # Asegurar que los breakpoints sean únicos
            breakpoints = np.unique(breakpoints)
            
            # Asignar buckets
            expected_bins = pd.cut(expected, bins=breakpoints, include_lowest=True, duplicates='drop')
            actual_bins = pd.cut(actual, bins=breakpoints, include_lowest=True, duplicates='drop')
            
            # Calcular frecuencias
            expected_counts = expected_bins.value_counts().sort_index()
            actual_counts = actual_bins.value_counts().sort_index()
            
            # Alinear buckets
            all_bins = expected_counts.index.union(actual_counts.index)
            expected_counts = expected_counts.reindex(all_bins, fill_value=0)
            actual_counts = actual_counts.reindex(all_bins, fill_value=0)
            
            # Calcular porcentajes
            expected_pct = expected_counts / len(expected)
            actual_pct = actual_counts / len(actual)
            
            # Evitar división por cero
            expected_pct = expected_pct.replace(0, 0.0001)
            actual_pct = actual_pct.replace(0, 0.0001)
            
            # Calcular PSI
            psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            psi_total = psi_values.sum()
            
            return psi_total, psi_values
            
        except Exception as e:
            print(f"⚠️ Error calculando PSI: {e}")
            return 0.0, pd.Series()
    
    def calculate_ks_statistic(self, expected, actual):
        """
        Calcula estadístico Kolmogorov-Smirnov para comparar distribuciones
        
        Args:
            expected: Datos de referencia
            actual: Datos nuevos
            
        Returns:
            KS statistic y p-value
        """
        try:
            ks_stat, p_value = stats.ks_2samp(expected, actual)
            return ks_stat, p_value
        except Exception as e:
            print(f"⚠️ Error calculando KS: {e}")
            return 0.0, 1.0
    
    def calculate_jensen_shannon(self, expected, actual, bins=50):
        """
        Calcula divergencia de Jensen-Shannon entre dos distribuciones
        
        Args:
            expected: Datos de referencia
            actual: Datos nuevos
            bins: Número de bins para histograma
            
        Returns:
            Divergencia de Jensen-Shannon
        """
        try:
            # Crear histogramas
            hist_expected, bin_edges = np.histogram(expected, bins=bins, density=True)
            hist_actual, _ = np.histogram(actual, bins=bin_edges, density=True)
            
            # Normalizar para que sumen 1
            hist_expected = hist_expected / np.sum(hist_expected)
            hist_actual = hist_actual / np.sum(hist_actual)
            
            # Evitar ceros
            hist_expected = np.maximum(hist_expected, 1e-10)
            hist_actual = np.maximum(hist_actual, 1e-10)
            
            # Calcular divergencia
            js_distance = distance.jensenshannon(hist_expected, hist_actual)
            
            return js_distance
            
        except Exception as e:
            print(f"⚠️ Error calculando Jensen-Shannon: {e}")
            return 0.0
    
    def calculate_chi_square(self, expected, actual):
        """
        Calcula estadístico Chi-cuadrado para variables categóricas
        
        Args:
            expected: Datos de referencia
            actual: Datos nuevos
            
        Returns:
            Chi-square statistic y p-value
        """
        try:
            # Crear tablas de contingencia
            expected_counts = pd.Series(expected).value_counts()
            actual_counts = pd.Series(actual).value_counts()
            
            # Alinear categorías
            all_categories = expected_counts.index.union(actual_counts.index)
            expected_counts = expected_counts.reindex(all_categories, fill_value=0)
            actual_counts = actual_counts.reindex(all_categories, fill_value=0)
            
            # Calcular chi-square
            chi2_stat, p_value, _, _ = stats.chi2_contingency(
                [expected_counts.values, actual_counts.values]
            )
            
            return chi2_stat, p_value
            
        except Exception as e:
            print(f"⚠️ Error calculando Chi-cuadrado: {e}")
            return 0.0, 1.0
    
    def evaluate_drift(self, new_data):
        """
        Evalúa data drift para todas las variables
        
        Args:
            new_data: DataFrame nuevos datos
            
        Returns:
            Diccionario con resultados de drift por variable
        """
        print("🔍 Evaluando data drift...")
        
        results = {}
        
        # Variables numéricas (excluyendo predicciones)
        numeric_vars = new_data.select_dtypes(include=[np.number]).columns
        numeric_vars = [col for col in numeric_vars if col not in ['prediction', 'actual']]
        
        for var in numeric_vars:
            if var in self.reference_data.columns:
                print(f"  📊 Analizando variable: {var}")
                
                ref_values = self.reference_data[var].dropna()
                new_values = new_data[var].dropna()
                
                if len(ref_values) > 0 and len(new_values) > 0:
                    # Calcular métricas
                    psi, _ = self.calculate_psi(ref_values, new_values)
                    ks_stat, ks_p = self.calculate_ks_statistic(ref_values, new_values)
                    js_dist = self.calculate_jensen_shannon(ref_values, new_values)
                    
                    # Determinar nivel de alerta
                    alert_level = self._get_alert_level(psi, ks_stat, js_dist)
                    
                    results[var] = {
                        'psi': psi,
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        'jensen_shannon': js_dist,
                        'alert_level': alert_level,
                        'reference_mean': ref_values.mean(),
                        'new_mean': new_values.mean(),
                        'reference_std': ref_values.std(),
                        'new_std': new_values.std()
                    }
        
        # Evaluar variable de predicción (categórica)
        if 'prediction' in new_data.columns:
            chi2_stat, chi2_p = self.calculate_chi_square(
                [1] * len(self.reference_data),  # Simular predicciones baseline
                new_data['prediction'].tolist()
            )
            
            results['prediction'] = {
                'chi2_statistic': chi2_stat,
                'chi2_p_value': chi2_p,
                'alert_level': self._get_alert_level_categorical(chi2_p)
            }
        
        self.monitoring_results = results
        print(f"✅ Evaluación completada: {len(results)} variables analizadas")
        
        return results
    
    def _get_alert_level(self, psi, ks_stat, js_dist):
        """
        Determina nivel de alerta basado en múltiples métricas
        
        Returns:
            'CRITICAL', 'WARNING', o 'NORMAL'
        """
        critical_count = 0
        warning_count = 0
        
        # Evaluar PSI
        if psi >= self.thresholds['psi_critical']:
            critical_count += 1
        elif psi >= self.thresholds['psi_warning']:
            warning_count += 1
        
        # Evaluar KS
        if ks_stat >= self.thresholds['ks_critical']:
            critical_count += 1
        elif ks_stat >= self.thresholds['ks_warning']:
            warning_count += 1
        
        # Evaluar Jensen-Shannon
        if js_dist >= self.thresholds['js_critical']:
            critical_count += 1
        elif js_dist >= self.thresholds['js_warning']:
            warning_count += 1
        
        # Determinar nivel
        if critical_count >= 2:
            return 'CRITICAL'
        elif critical_count >= 1 or warning_count >= 2:
            return 'WARNING'
        else:
            return 'NORMAL'
    
    def _get_alert_level_categorical(self, chi2_p):
        """
        Determina nivel de alerta para variables categóricas
        
        Returns:
            'CRITICAL', 'WARNING', o 'NORMAL'
        """
        if chi2_p < 0.01:
            return 'CRITICAL'
        elif chi2_p < 0.05:
            return 'WARNING'
        else:
            return 'NORMAL'
    
    def create_drift_report(self, save_path='assets/drift_report.html'):
        """
        Crea reporte HTML de monitoreo con visualizaciones
        
        Args:
            save_path: Path donde guardar el reporte
        """
        print("📋 Generando reporte de monitoreo...")
        
        if not self.monitoring_results:
            print("❌ No hay resultados de monitoreo disponibles")
            return
        
        # Crear HTML
        html_content = self._generate_html_report()
        
        # Asegurar que el directorio assets exista
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Guardar reporte
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Reporte guardado en: {save_path}")
        
        # También guardar resultados en JSON en assets
        json_path = save_path.replace('.html', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.monitoring_results, f, indent=2, default=str)
        
        print(f"✅ Resultados guardados en: {json_path}")
    
    def _generate_html_report(self):
        """
        Genera contenido HTML para el reporte
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Monitoring Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
                .critical {{ background-color: #ffebee; border-color: #f44336; }}
                .warning {{ background-color: #fff3e0; border-color: #ff9800; }}
                .normal {{ background-color: #e8f5e8; border-color: #4caf50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Model Monitoring Report</h1>
                <p><strong>Fecha:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Modelo:</strong> {self.metadata.get('best_model', 'Unknown')}</p>
                <p><strong>Variables analizadas:</strong> {len(self.monitoring_results)}</p>
            </div>
            
            <h2>📊 Resumen de Data Drift</h2>
            <table>
                <tr>
                    <th>Variable</th>
                    <th>PSI</th>
                    <th>KS Statistic</th>
                    <th>Jensen-Shannon</th>
                    <th>Alert Level</th>
                    <th>Mean Change</th>
                </tr>
        """
        
        # Agregar filas de resultados
        for var, metrics in self.monitoring_results.items():
            if 'psi' in metrics:
                alert_class = metrics['alert_level'].lower()
                mean_change = metrics['new_mean'] - metrics['reference_mean']
                
                html += f"""
                <tr class="{alert_class}">
                    <td>{var}</td>
                    <td>{metrics['psi']:.4f}</td>
                    <td>{metrics['ks_statistic']:.4f}</td>
                    <td>{metrics['jensen_shannon']:.4f}</td>
                    <td><strong>{metrics['alert_level']}</strong></td>
                    <td>{mean_change:+.4f}</td>
                </tr>
                """
        
        html += """
            </table>
            
            <h2>🚨 Alertas Detectadas</h2>
        """
        
        # Contar alertas
        critical_vars = [k for k, v in self.monitoring_results.items() 
                       if v.get('alert_level') == 'CRITICAL']
        warning_vars = [k for k, v in self.monitoring_results.items() 
                      if v.get('alert_level') == 'WARNING']
        
        if critical_vars:
            html += f"<div class='metric critical'><h3>⚠️ CRITICAL</h3><p>{', '.join(critical_vars)}</p></div>"
        
        if warning_vars:
            html += f"<div class='metric warning'><h3>⚡ WARNING</h3><p>{', '.join(warning_vars)}</p></div>"
        
        if not critical_vars and not warning_vars:
            html += "<div class='metric normal'><h3>✅ NORMAL</h3><p>No se detectaron alertas significativas</p></div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def plot_drift_comparison(self, variable, save_path=None):
        """
        Crea gráfico comparativo de distribuciones
        
        Args:
            variable: Nombre de la variable a graficar
            save_path: Path para guardar el gráfico
        """
        if variable not in self.monitoring_results:
            print(f"❌ Variable {variable} no encontrada en resultados")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Análisis de Drift - {variable}', fontsize=16, fontweight='bold')
        
        # Datos para graficar
        ref_data = self.reference_data[variable].dropna()
        new_data_sample = np.random.normal(
            self.monitoring_results[variable]['new_mean'],
            self.monitoring_results[variable]['new_std'],
            len(ref_data)
        )
        
        # 1. Histogramas comparativos
        axes[0, 0].hist(ref_data, bins=30, alpha=0.7, label='Referencia', color='blue')
        axes[0, 0].hist(new_data_sample, bins=30, alpha=0.7, label='Actual', color='red')
        axes[0, 0].set_title('Distribución - Referencia vs Actual')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot comparativo
        data_for_box = pd.DataFrame({
            'Referencia': ref_data,
            'Actual': new_data_sample
        }).melt(var_name='Dataset', value_name='Valor')
        
        sns.boxplot(data=data_for_box, x='Dataset', y='Valor', ax=axes[0, 1])
        axes[0, 1].set_title('Box Plot - Comparativo')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        stats.probplot(ref_data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot - Referencia')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Métricas de drift
        metrics = self.monitoring_results[variable]
        alert_color = {'CRITICAL': 'red', 'WARNING': 'orange', 'NORMAL': 'green'}[metrics['alert_level']]
        
        metrics_text = f"""
        PSI: {metrics['psi']:.4f}
        KS: {metrics['ks_statistic']:.4f}
        Jensen-Shannon: {metrics['jensen_shannon']:.4f}
        Alert Level: {metrics['alert_level']}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=alert_color, alpha=0.3))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Métricas de Drift')
        
        plt.tight_layout()
        
        # Guardar en assets/images por defecto o en el path especificado
        if save_path is None:
            # Crear directorio assets/images si no existe
            save_path = f'assets/images/drift_plot_{variable}.png'
        elif not save_path.startswith('assets/'):
            # Si no empieza con assets/, agregarlo
            save_path = f'assets/images/{os.path.basename(save_path)}'
        
        # Asegurar que el directorio exista
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado en: {save_path}")
        
        plt.show()
    
    def generate_monitoring_dashboard(self):
        """
        Genera datos para dashboard de Streamlit
        """
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'model_info': self.metadata,
            'monitoring_results': self.monitoring_results,
            'summary': {
                'total_variables': len(self.monitoring_results),
                'critical_alerts': len([v for v in self.monitoring_results.values() 
                                   if v.get('alert_level') == 'CRITICAL']),
                'warning_alerts': len([v for v in self.monitoring_results.values() 
                                    if v.get('alert_level') == 'WARNING']),
                'normal_variables': len([v for v in self.monitoring_results.values() 
                                     if v.get('alert_level') == 'NORMAL'])
            }
        }
        
        return dashboard_data

def main():
    """
    Función principal para ejecutar monitoreo completo
    """
    print("🚀 INICIANDO SISTEMA DE MONITOREO DE MODELOS")
    print("="*60)
    
    # 1. Inicializar monitor
    monitor = ModelMonitor()
    
    # 2. Simular carga de nuevos datos (en producción vendrían de API)
    print("\n📊 FASE 1: CARGA DE DATOS NUEVOS")
    print("-"*40)
    
    # Para demo, usamos datos existentes como si fueran nuevos
    new_data = monitor.reference_data.sample(n=100, random_state=42)
    new_data['prediction'] = monitor.model.predict(new_data)
    new_data['actual'] = np.random.choice([0, 1], size=len(new_data), p=[0.05, 0.95])
    
    print(f"✅ Nuevos datos simulados: {new_data.shape}")
    
    # 3. Evaluar drift
    print("\n🔍 FASE 2: EVALUACIÓN DE DATA DRIFT")
    print("-"*40)
    
    drift_results = monitor.evaluate_drift(new_data)
    
    # 4. Mostrar resumen
    print("\n📋 FASE 3: RESUMEN DE RESULTADOS")
    print("-"*40)
    
    critical_vars = [k for k, v in drift_results.items() 
                   if v.get('alert_level') == 'CRITICAL']
    warning_vars = [k for k, v in drift_results.items() 
                  if v.get('alert_level') == 'WARNING']
    
    print(f"📊 Variables analizadas: {len(drift_results)}")
    print(f"⚠️ Alertas críticas: {len(critical_vars)}")
    if critical_vars:
        print(f"   Variables: {', '.join(critical_vars)}")
    
    print(f"⚡ Alertas de advertencia: {len(warning_vars)}")
    if warning_vars:
        print(f"   Variables: {', '.join(warning_vars)}")
    
    # 5. Generar reporte
    print("\n📄 FASE 4: GENERACIÓN DE REPORTES")
    print("-"*40)
    
    monitor.create_drift_report()
    
    # 6. Generar gráficos para variables con alertas
    alert_vars = critical_vars + warning_vars
    if alert_vars:
        print(f"\n📊 FASE 5: GRÁFICOS DE VARIABLES CON ALERTAS")
        print("-"*50)
        
        for var in alert_vars[:3]:  # Limitar a 3 gráficos
            monitor.plot_drift_comparison(var, f'drift_plot_{var}.png')
    
    # 7. Generar datos para Streamlit
    print("\n🌐 FASE 6: DATOS PARA DASHBOARD")
    print("-"*40)
    
    dashboard_data = monitor.generate_monitoring_dashboard()
    
    # Guardar datos para Streamlit en assets
    import os
    os.makedirs('assets', exist_ok=True)
    joblib.dump(dashboard_data, 'assets/streamlit_dashboard_data.pkl')
    print("✅ Datos para dashboard guardados en assets/")
    
    print(f"\n🎉 MONITOREO COMPLETADO!")
    print("="*50)
    print(f"📊 Resumen final:")
    print(f"  • Variables analizadas: {len(drift_results)}")
    print(f"  • Alertas críticas: {len(critical_vars)}")
    print(f"  • Alertas de advertencia: {len(warning_vars)}")
    print(f"  • Reporte HTML: assets/drift_report.html")
    print(f"  • Datos dashboard: assets/streamlit_dashboard_data.pkl")
    
    return monitor, drift_results

if __name__ == "__main__":
    monitor, results = main()
