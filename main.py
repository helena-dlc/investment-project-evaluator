"""
Evaluador de Proyectos - Versión Completa
Aplicación Streamlit para análisis financiero de proyectos de inversión
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import fsolve
import warnings
import io
import base64
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Evaluador de Proyectos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para interfaz moderna
st.markdown("""
<style>
    /* Estilo del título principal */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .main-title h1 {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-title p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Sidebar moderno */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9ff 0%, #e8f0ff 100%);
    }
    
    /* Métricas personalizadas */
    .metric-container {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    .success-metric {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left-color: #28a745;
    }
    
    .warning-metric {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left-color: #ffc107;
    }
    
    .danger-metric {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left-color: #dc3545;
    }
    
    /* Botones modernos */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox moderno */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    /* Tabs modernos */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #667eea;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Footer moderno */
    .modern-footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


class FinancialAnalyzer:
    """Clase para análisis financiero avanzado de proyectos"""
    
    @staticmethod
    def calculate_npv(cash_flows, discount_rate):
        """Calcula el Valor Actual Neto (VAN/NPV)"""
        npv_value = 0
        for period, flow in enumerate(cash_flows):
            discount_factor = (1 + discount_rate) ** period
            present_value = flow / discount_factor
            npv_value += present_value
        return npv_value
    
    @staticmethod
    def calculate_npv_detailed(cash_flows, discount_rate):
        """Calcula el VAN con detalle paso a paso"""
        detailed_calculation = []
        npv_total = 0
        
        for period, flow in enumerate(cash_flows):
            discount_factor = (1 + discount_rate) ** period
            present_value = flow / discount_factor
            npv_total += present_value
            
            detailed_calculation.append({
                'periodo': period,
                'flujo': flow,
                'factor_descuento': discount_factor,
                'valor_presente': present_value,
                'npv_acumulado': npv_total
            })
        
        return npv_total, detailed_calculation
    
    @staticmethod
    def calculate_irr(cash_flows, max_iterations=100):
        """Calcula la Tasa Interna de Retorno (TIR/IRR)"""
        def npv_function(rate):
            return FinancialAnalyzer.calculate_npv(cash_flows, rate)
        
        initial_guesses = [0.1, 0.2, 0.5, -0.1, -0.5, 1.0]
        
        for guess in initial_guesses:
            try:
                irr_value = fsolve(npv_function, guess, maxfev=max_iterations)[0]
                if abs(npv_function(irr_value)) < 1e-6:
                    return irr_value
            except (RuntimeWarning, ValueError, OverflowError):
                continue
        
        return None
    
    @staticmethod
    def calculate_payback(cash_flows):
        """Calcula el período de recuperación"""
        cumulative_flow = 0
        for period, flow in enumerate(cash_flows):
            cumulative_flow += flow
            if cumulative_flow >= 0:
                return period
        return None
    
    @staticmethod
    def calculate_discounted_payback(cash_flows, discount_rate):
        """Calcula el período de recuperación descontado"""
        cumulative_pv = 0
        for period, flow in enumerate(cash_flows):
            pv = flow / (1 + discount_rate) ** period
            cumulative_pv += pv
            if cumulative_pv >= 0:
                return period
        return None
    
    @staticmethod
    def calculate_profitability_index(cash_flows, discount_rate):
        """Calcula el índice de rentabilidad"""
        initial_investment = abs(cash_flows[0])
        if initial_investment == 0:
            return 0
        
        present_value_positive_flows = sum([
            flow / (1 + discount_rate) ** period 
            for period, flow in enumerate(cash_flows[1:], 1)
            if flow > 0
        ])
        
        return present_value_positive_flows / initial_investment
    
    @staticmethod
    def calculate_caue(cash_flows, discount_rate):
        """
        Calcula el Costo Anual Uniforme Equivalente (CAUE)
        CAUE = VAN * [r(1+r)^n] / [(1+r)^n - 1]
        """
        npv = FinancialAnalyzer.calculate_npv(cash_flows, discount_rate)
        n = len(cash_flows) - 1  # Períodos (excluyendo período 0)
        
        if n <= 0 or discount_rate <= 0:
            return 0
        
        # Factor de recuperación de capital
        if discount_rate == 0:
            crf = 1 / n
        else:
            crf = (discount_rate * (1 + discount_rate) ** n) / ((1 + discount_rate) ** n - 1)
        
        caue = npv * crf
        return caue
    
    @staticmethod
    def sensitivity_analysis(cash_flows, base_discount_rate, sensitivity_range=0.1, points=21):
        """Análisis de sensibilidad del VAN respecto a la tasa de descuento"""
        rates = np.linspace(
            max(0.001, base_discount_rate - sensitivity_range),
            base_discount_rate + sensitivity_range,
            points
        )
        
        npvs = []
        for rate in rates:
            npv = FinancialAnalyzer.calculate_npv(cash_flows, rate)
            npvs.append(npv)
        
        return rates, npvs
    
    @staticmethod
    def cash_flow_sensitivity(cash_flows, discount_rate, variation_percent=0.2, points=11):
        """Análisis de sensibilidad de los flujos de caja"""
        variations = np.linspace(-variation_percent, variation_percent, points)
        results = {
            'variations': [],
            'npvs': [],
            'irrs': []
        }
        
        for var in variations:
            # Aplicar variación a flujos positivos (excluyendo inversión inicial)
            modified_flows = cash_flows.copy()
            for i in range(1, len(modified_flows)):
                if modified_flows[i] > 0:
                    modified_flows[i] = modified_flows[i] * (1 + var)
            
            npv = FinancialAnalyzer.calculate_npv(modified_flows, discount_rate)
            irr = FinancialAnalyzer.calculate_irr(modified_flows)
            
            results['variations'].append(var * 100)
            results['npvs'].append(npv)
            results['irrs'].append(irr * 100 if irr else 0)
        
        return results
    
    @staticmethod
    def project_timing_analysis(cash_flows, discount_rate, is_repeatable=False):
        """
        Análisis de momento óptimo considerando si el proyecto es repetible
        """
        results = {
            'periods': [],
            'cumulative_npv': [],
            'marginal_npv': [],
            'caue_values': [],
            'optimal_period': None,
            'recommendation': ''
        }
        
        n_periods = len(cash_flows)
        previous_npv = 0
        
        for period in range(2, n_periods + 1):
            period_flows = cash_flows[:period]
            
            # NPV acumulado
            npv = FinancialAnalyzer.calculate_npv(period_flows, discount_rate)
            
            # NPV marginal
            marginal_npv = npv - previous_npv
            
            # CAUE para proyectos repetibles
            caue = FinancialAnalyzer.calculate_caue(period_flows, discount_rate)
            
            results['periods'].append(period)
            results['cumulative_npv'].append(npv)
            results['marginal_npv'].append(marginal_npv)
            results['caue_values'].append(caue)
            
            previous_npv = npv
        
        # Determinar momento óptimo
        if is_repeatable:
            # Para proyectos repetibles, maximizar CAUE
            max_caue_idx = np.argmax(results['caue_values'])
            optimal_period = results['periods'][max_caue_idx]
            optimal_caue = results['caue_values'][max_caue_idx]
            
            results['optimal_period'] = optimal_period
            results['recommendation'] = f"""
            **Proyecto Repetible - Momento Óptimo:** Período {optimal_period}
            
            **CAUE Óptimo:** ${optimal_caue:,.2f} anual
            
            **Criterio:** Maximizar el Costo Anual Uniforme Equivalente para proyectos que se pueden repetir indefinidamente.
            """
        else:
            # Para proyectos únicos, maximizar VAN
            max_npv_idx = np.argmax(results['cumulative_npv'])
            optimal_period = results['periods'][max_npv_idx]
            optimal_npv = results['cumulative_npv'][max_npv_idx]
            
            results['optimal_period'] = optimal_period
            results['recommendation'] = f"""
            **Proyecto Único - Momento Óptimo:** Período {optimal_period}
            
            **VAN Óptimo:** ${optimal_npv:,.2f}
            
            **Criterio:** Maximizar el Valor Actual Neto para proyectos de una sola ejecución.
            """
        
        return results
    
    @staticmethod
    def create_downloadable_table(df, filename):
        """Crea un enlace de descarga para tabla CSV"""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">📥 Descargar {filename}</a>'
        return href


def create_cash_flow_input(key_prefix, max_periods=30):
    """Crea interfaz para entrada de flujos de caja (hasta 30 períodos)"""
    st.subheader("💰 Configuración de Flujos de Caja")
    
    # Número de períodos
    n_periods = st.number_input(
        "Número de períodos", 
        min_value=2, 
        max_value=max_periods, 
        value=5,
        key=f"{key_prefix}_periods"
    )
    
    cash_flows = []
    
    # Inversión inicial
    initial_investment = st.number_input(
        "Inversión inicial ($)", 
        value=-100000.0,
        key=f"{key_prefix}_initial"
    )
    cash_flows.append(initial_investment)
    
    # Crear flujos futuros en una cuadrícula más organizada
    st.write("**Flujos de caja futuros:**")
    
    cols_per_row = 4
    rows_needed = (n_periods - 1 + cols_per_row - 1) // cols_per_row
    
    for row in range(rows_needed):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            period = row * cols_per_row + col_idx + 1
            if period < n_periods:
                with cols[col_idx]:
                    flow = st.number_input(
                        f"Período {period}", 
                        value=30000.0,
                        key=f"{key_prefix}_flow_{period}"
                    )
                    cash_flows.append(flow)
    
    return cash_flows


def main():
    """Función principal de la aplicación"""
    
    # Título principal moderno
    st.markdown("""
    <div class="main-title">
        <h1>📊 Evaluador de Proyectos</h1>
        <p>Herramienta avanzada para análisis financiero y toma de decisiones de inversión</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar moderno para navegación
    with st.sidebar:
        st.markdown("### 🎯 Herramientas Disponibles")
        st.markdown("Selecciona la herramienta que necesitas:")
        
        tool = st.selectbox(
            "",
            [
                "🏦 Calculadora de Intereses",
                "📈 Análisis VAN/TIR/CAUE", 
                "🔍 Análisis de Sensibilidad",
                "⏰ Análisis de Momento Óptimo",
                "⚖️ Comparación de Proyectos"
            ]
        )
        
        st.markdown("---")
        st.markdown("### 📚 Indicadores Financieros")
        st.info("""
        **Principales métricas:**
        
        • **VAN:** Valor presente neto
        • **TIR:** Tasa interna de retorno
        • **CAUE:** Costo anual uniforme equivalente
        • **Payback:** Período de recuperación
        • **IP:** Índice de rentabilidad
        """)
        
        st.markdown("### 📊 Criterios de Decisión")
        st.success("""
        **Proyecto viable si:**
        • VAN > 0
        • TIR > Tasa de descuento
        • CAUE > 0 (para proyectos repetibles)
        • IP > 1
        """)
    
    # Herramienta 1: Calculadora de Intereses
    if tool == "🏦 Calculadora de Intereses":
        st.header("🏦 Calculadora de Intereses")
        
        tab1, tab2 = st.tabs(["Interés Simple", "Interés Compuesto"])
        
        with tab1:
            st.subheader("📊 Interés Simple")
            col1, col2 = st.columns(2)
            
            with col1:
                principal_simple = st.number_input("Capital inicial ($)", value=10000.0, key="simple_principal")
                rate_simple = st.number_input("Tasa de interés anual (%)", value=5.0, key="simple_rate") / 100
                time_simple = st.number_input("Tiempo (años)", value=1.0, key="simple_time")
            
            with col2:
                if st.button("💰 Calcular Interés Simple", key="calc_simple"):
                    final_amount = principal_simple * (1 + rate_simple * time_simple)
                    interest_earned = final_amount - principal_simple
                    
                    st.metric("Monto Final", f"${final_amount:,.2f}")
                    st.metric("Interés Ganado", f"${interest_earned:,.2f}")
                    
                    # Gráfico de crecimiento
                    years = np.linspace(0, time_simple, int(time_simple * 4) + 1)
                    amounts = [principal_simple * (1 + rate_simple * t) for t in years]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=years, y=amounts, mode='lines+markers', 
                                           name='Crecimiento', line=dict(width=3, color='#667eea')))
                    fig.update_layout(
                        title="Crecimiento con Interés Simple",
                        xaxis_title="Años",
                        yaxis_title="Monto ($)",
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("📈 Interés Compuesto")
            col1, col2 = st.columns(2)
            
            with col1:
                principal_compound = st.number_input("Capital inicial ($)", value=10000.0, key="compound_principal")
                rate_compound = st.number_input("Tasa de interés anual (%)", value=5.0, key="compound_rate") / 100
                time_compound = st.number_input("Tiempo (años)", value=1.0, key="compound_time")
                compounding = st.selectbox(
                    "Capitalización", 
                    [1, 2, 4, 12, 365], 
                    format_func=lambda x: {1:"Anual", 2:"Semestral", 4:"Trimestral", 12:"Mensual", 365:"Diaria"}[x]
                )
            
            with col2:
                if st.button("📈 Calcular Interés Compuesto", key="calc_compound"):
                    final_amount = principal_compound * (1 + rate_compound / compounding) ** (compounding * time_compound)
                    interest_earned = final_amount - principal_compound
                    
                    st.metric("Monto Final", f"${final_amount:,.2f}")
                    st.metric("Interés Ganado", f"${interest_earned:,.2f}")
                    
                    # Comparación con interés simple
                    simple_amount = principal_compound * (1 + rate_compound * time_compound)
                    difference = final_amount - simple_amount
                    st.metric("Ventaja del Interés Compuesto", f"${difference:,.2f}")
                    
                    # Gráfico comparativo
                    years = np.linspace(0, time_compound, 50)
                    compound_amounts = [principal_compound * (1 + rate_compound / compounding) ** (compounding * t) for t in years]
                    simple_amounts = [principal_compound * (1 + rate_compound * t) for t in years]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=years, y=compound_amounts, mode='lines', 
                                           name='Interés Compuesto', line=dict(width=3, color='#667eea')))
                    fig.add_trace(go.Scatter(x=years, y=simple_amounts, mode='lines', 
                                           name='Interés Simple', line=dict(width=3, color='#764ba2', dash='dash')))
                    fig.update_layout(
                        title="Comparación: Interés Compuesto vs Simple",
                        xaxis_title="Años",
                        yaxis_title="Monto ($)",
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Herramienta 2: Análisis VAN/TIR/CAUE
    elif tool == "📈 Análisis VAN/TIR/CAUE":
        st.header("📈 Análisis Financiero Completo")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Parámetros del Proyecto")
            discount_rate = st.number_input("Tasa de descuento (%)", value=10.0, min_value=0.1) / 100
            project_type = st.radio("Tipo de proyecto:", ["Único", "Repetible"])
            
            cash_flows = create_cash_flow_input("van_tir")
        
        with col2:
            if st.button("🔬 Realizar Análisis Completo", key="calc_indicators"):
                # Cálculos principales
                npv_result = FinancialAnalyzer.calculate_npv(cash_flows, discount_rate)
                irr_result = FinancialAnalyzer.calculate_irr(cash_flows)
                payback_result = FinancialAnalyzer.calculate_payback(cash_flows)
                discounted_payback_result = FinancialAnalyzer.calculate_discounted_payback(cash_flows, discount_rate)
                pi_result = FinancialAnalyzer.calculate_profitability_index(cash_flows, discount_rate)
                caue_result = FinancialAnalyzer.calculate_caue(cash_flows, discount_rate)
                
                # Resultados principales
                st.subheader("📊 Indicadores Financieros")
                
                col_res1, col_res2, col_res3, col_res4, col_res5 = st.columns(5)
                
                with col_res1:
                    delta_npv = "Viable ✅" if npv_result > 0 else "No viable ❌"
                    st.metric("VAN ($)", f"{npv_result:,.0f}", delta=delta_npv)
                
                with col_res2:
                    if irr_result is not None:
                        delta_irr = "Aceptable ✅" if irr_result > discount_rate else "Rechazar ❌"
                        st.metric("TIR (%)", f"{irr_result*100:.2f}", delta=delta_irr)
                    else:
                        st.metric("TIR", "No calculable")
                
                with col_res3:
                    delta_caue = "Positivo ✅" if caue_result > 0 else "Negativo ❌"
                    st.metric("CAUE ($)", f"{caue_result:,.0f}", delta=delta_caue)
                
                with col_res4:
                    if payback_result is not None:
                        st.metric("Payback Simple", f"{payback_result} períodos")
                    else:
                        st.metric("Payback Simple", "No recupera")
                
                with col_res5:
                    if discounted_payback_result is not None:
                        st.metric("Payback Descontado", f"{discounted_payback_result} períodos")
                    else:
                        st.metric("Payback Descontado", "No recupera")
                
                # Índice de rentabilidad
                col_pi1, col_pi2 = st.columns(2)
                with col_pi1:
                    delta_pi = "Rentable ✅" if pi_result > 1 else "No rentable ❌"
                    st.metric("Índice de Rentabilidad", f"{pi_result:.3f}", delta=delta_pi)
                
                # Perfil VAN-TIR
                st.subheader("📈 Perfil del Proyecto")
                
                rates = np.linspace(0, 0.5, 100)
                npvs = [FinancialAnalyzer.calculate_npv(cash_flows, rate) for rate in rates]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rates*100, y=npvs, mode='lines', name='VAN', 
                    line=dict(width=3, color='#667eea')
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="VAN = 0")
                
                if irr_result is not None and 0 <= irr_result <= 0.5:
                    fig.add_vline(
                        x=irr_result*100, 
                        line_dash="dash", 
                        line_color="green", 
                        annotation_text=f"TIR: {irr_result*100:.2f}%"
                    )
                
                fig.add_vline(
                    x=discount_rate*100, 
                    line_dash="dot", 
                    line_color="orange", 
                    annotation_text=f"Tasa objetivo: {discount_rate*100:.1f}%"
                )
                
                fig.update_layout(
                    title="Perfil VAN vs Tasa de Descuento",
                    xaxis_title="Tasa de descuento (%)",
                    yaxis_title="VAN ($)",
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla detallada de flujos
                st.subheader("💰 Análisis Detallado de Flujos")
                
                npv_detailed, calculation_steps = FinancialAnalyzer.calculate_npv_detailed(cash_flows, discount_rate)
                
                df_flows = pd.DataFrame(calculation_steps)
                df_flows_display = pd.DataFrame({
                    'Período': df_flows['periodo'],
                    'Flujo de Caja ($)': [f"{flow:,.2f}" for flow in df_flows['flujo']],
                    'Factor Descuento': [f"{factor:.6f}" for factor in df_flows['factor_descuento']],
                    'Valor Presente ($)': [f"{pv:,.2f}" for pv in df_flows['valor_presente']],
                    'VAN Acumulado ($)': [f"{acc:,.2f}" for acc in df_flows['npv_acumulado']]
                })
                
                st.dataframe(df_flows_display, use_container_width=True)
                
                # Descarga de resultados
                st.markdown(
                    FinancialAnalyzer.create_downloadable_table(df_flows_display, "analisis_flujos"),
                    unsafe_allow_html=True
                )
    
    # Herramienta 3: Análisis de Sensibilidad
    elif tool == "🔍 Análisis de Sensibilidad":
        st.header("🔍 Análisis de Sensibilidad")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuración del Análisis")
            discount_rate_sens = st.number_input("Tasa de descuento base (%)", value=12.0) / 100
            
            # Parámetros de sensibilidad
            st.subheader("📊 Parámetros de Sensibilidad")
            rate_variation = st.slider("Variación en tasa de descuento (%)", 1, 20, 10) / 100
            cashflow_variation = st.slider("Variación en flujos de caja (%)", 5, 50, 20) / 100
            
            cash_flows_sens = create_cash_flow_input("sensitivity")
        
        with col2:
            if st.button("🔍 Realizar Análisis de Sensibilidad", key="calc_sensitivity"):
                st.subheader("📊 Resultados del Análisis")
                
                # Análisis de sensibilidad de la tasa de descuento
                rates, npvs = FinancialAnalyzer.sensitivity_analysis(
                    cash_flows_sens, discount_rate_sens, rate_variation
                )
                
                # Análisis de sensibilidad de flujos de caja
                cashflow_sens = FinancialAnalyzer.cash_flow_sensitivity(
                    cash_flows_sens, discount_rate_sens, cashflow_variation
                )
                
                # Gráficos de sensibilidad
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        'Sensibilidad VAN vs Tasa de Descuento',
                        'Sensibilidad VAN vs Variación de Flujos',
                        'Sensibilidad TIR vs Variación de Flujos',
                        'Análisis Tornado'
                    ]
                )
                
                # VAN vs Tasa de descuento
                fig.add_trace(
                    go.Scatter(
                        x=rates*100, y=npvs,
                        mode='lines+markers',
                        name='VAN vs Tasa',
                        line=dict(color='#667eea', width=3)
                    ),
                    row=1, col=1
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
                fig.add_vline(x=discount_rate_sens*100, line_dash="dot", line_color="orange", row=1, col=1)
                
                # VAN vs Variación de flujos
                fig.add_trace(
                    go.Scatter(
                        x=cashflow_sens['variations'], y=cashflow_sens['npvs'],
                        mode='lines+markers',
                        name='VAN vs Flujos',
                        line=dict(color='#764ba2', width=3)
                    ),
                    row=1, col=2
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
                fig.add_vline(x=0, line_dash="dot", line_color="orange", row=1, col=2)
                
                # TIR vs Variación de flujos
                fig.add_trace(
                    go.Scatter(
                        x=cashflow_sens['variations'], y=cashflow_sens['irrs'],
                        mode='lines+markers',
                        name='TIR vs Flujos',
                        line=dict(color='#28a745', width=3)
                    ),
                    row=2, col=1
                )
                fig.add_hline(y=discount_rate_sens*100, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_vline(x=0, line_dash="dot", line_color="orange", row=2, col=1)
                
                # Análisis Tornado - Sensibilidad relativa
                base_npv = FinancialAnalyzer.calculate_npv(cash_flows_sens, discount_rate_sens)
                
                # Calcular sensibilidad para diferentes variables
                rate_high = FinancialAnalyzer.calculate_npv(cash_flows_sens, discount_rate_sens + 0.05)
                rate_low = FinancialAnalyzer.calculate_npv(cash_flows_sens, discount_rate_sens - 0.05)
                
                # Flujos modificados
                flows_high = cash_flows_sens.copy()
                flows_low = cash_flows_sens.copy()
                for i in range(1, len(flows_high)):
                    if flows_high[i] > 0:
                        flows_high[i] *= 1.2
                        flows_low[i] *= 0.8
                
                flows_high_npv = FinancialAnalyzer.calculate_npv(flows_high, discount_rate_sens)
                flows_low_npv = FinancialAnalyzer.calculate_npv(flows_low, discount_rate_sens)
                
                tornado_data = {
                    'Variable': ['Tasa de Descuento', 'Flujos de Caja'],
                    'Bajo': [rate_low - base_npv, flows_low_npv - base_npv],
                    'Alto': [rate_high - base_npv, flows_high_npv - base_npv]
                }
                
                for i, var in enumerate(tornado_data['Variable']):
                    fig.add_trace(
                        go.Bar(
                            x=[tornado_data['Bajo'][i]], y=[var],
                            orientation='h', name=f'{var} (Bajo)',
                            marker_color='red', opacity=0.7
                        ),
                        row=2, col=2
                    )
                    fig.add_trace(
                        go.Bar(
                            x=[tornado_data['Alto'][i]], y=[var],
                            orientation='h', name=f'{var} (Alto)',
                            marker_color='green', opacity=0.7
                        ),
                        row=2, col=2
                    )
                
                fig.update_layout(height=800, showlegend=False, template='plotly_white')
                fig.update_xaxes(title_text="Tasa de Descuento (%)", row=1, col=1)
                fig.update_xaxes(title_text="Variación Flujos (%)", row=1, col=2)
                fig.update_xaxes(title_text="Variación Flujos (%)", row=2, col=1)
                fig.update_xaxes(title_text="Impacto en VAN ($)", row=2, col=2)
                fig.update_yaxes(title_text="VAN ($)", row=1, col=1)
                fig.update_yaxes(title_text="VAN ($)", row=1, col=2)
                fig.update_yaxes(title_text="TIR (%)", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de sensibilidad
                st.subheader("📋 Tabla de Sensibilidad")
                
                sensitivity_table = pd.DataFrame({
                    'Tasa de Descuento (%)': [f"{r*100:.1f}" for r in rates[::4]],
                    'VAN ($)': [f"{npv:,.0f}" for npv in npvs[::4]],
                    'Variación Flujos (%)': [f"{v:.1f}" for v in cashflow_sens['variations'][::2]],
                    'VAN con Variación ($)': [f"{npv:,.0f}" for npv in cashflow_sens['npvs'][::2]]
                })
                
                st.dataframe(sensitivity_table, use_container_width=True)
                
                # Interpretación de resultados
                st.subheader("🎯 Interpretación de Resultados")
                
                # Encontrar punto de equilibrio (VAN = 0)
                npv_zero_rate = None
                for i, npv in enumerate(npvs):
                    if i > 0 and npvs[i-1] * npv <= 0:
                        npv_zero_rate = rates[i]
                        break
                
                col_interp1, col_interp2 = st.columns(2)
                
                with col_interp1:
                    if npv_zero_rate:
                        st.info(f"""
                        **Punto de Equilibrio:**
                        
                        VAN = 0 cuando tasa ≈ {npv_zero_rate*100:.2f}%
                        
                        Margen de seguridad: {(npv_zero_rate - discount_rate_sens)*100:.2f} puntos porcentuales
                        """)
                    else:
                        st.warning("No se encontró punto de equilibrio en el rango analizado")
                
                with col_interp2:
                    # Análisis de riesgo
                    npv_std = np.std(npvs)
                    risk_level = "Bajo" if npv_std < abs(base_npv) * 0.1 else "Medio" if npv_std < abs(base_npv) * 0.3 else "Alto"
                    
                    st.info(f"""
                    **Análisis de Riesgo:**
                    
                    Desviación estándar VAN: ${npv_std:,.0f}
                    
                    Nivel de riesgo: **{risk_level}**
                    
                    Variable más sensible: {'Flujos de caja' if abs(flows_high_npv - flows_low_npv) > abs(rate_high - rate_low) else 'Tasa de descuento'}
                    """)
                
                # Descarga de análisis
                st.markdown(
                    FinancialAnalyzer.create_downloadable_table(sensitivity_table, "analisis_sensibilidad"),
                    unsafe_allow_html=True
                )
    
    # Herramienta 4: Análisis de Momento Óptimo
    elif tool == "⏰ Análisis de Momento Óptimo":
        st.header("⏰ Análisis de Momento Óptimo")
        
        st.info("""
        **Teoría del Momento Óptimo:**
        - **Proyectos únicos:** Maximizar VAN total
        - **Proyectos repetibles:** Maximizar CAUE (Costo Anual Uniforme Equivalente)
        - **Criterio marginal:** Continuar mientras VAN marginal > 0
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuración del Análisis")
            discount_rate_opt = st.number_input("Tasa de descuento (%)", value=12.0, key="opt_rate") / 100
            is_repeatable = st.checkbox("Proyecto repetible", value=False)
            
            cash_flows_opt = create_cash_flow_input("optimal")
        
        with col2:
            if st.button("🎯 Realizar Análisis de Momento Óptimo", key="calc_optimal"):
                results = FinancialAnalyzer.project_timing_analysis(
                    cash_flows_opt, discount_rate_opt, is_repeatable
                )
                
                st.subheader("📊 Resultados del Análisis")
                
                # Gráficos del análisis
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        'VAN Acumulado por Período',
                        'CAUE por Período' if is_repeatable else 'VAN Marginal por Período',
                        'Evolución Comparativa',
                        'Recomendación Óptima'
                    ]
                )
                
                # VAN Acumulado
                fig.add_trace(
                    go.Scatter(
                        x=results['periods'], 
                        y=results['cumulative_npv'],
                        mode='lines+markers',
                        name='VAN Acumulado',
                        line=dict(color='#667eea', width=3)
                    ),
                    row=1, col=1
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
                
                # CAUE o VAN Marginal
                if is_repeatable:
                    fig.add_trace(
                        go.Scatter(
                            x=results['periods'], 
                            y=results['caue_values'],
                            mode='lines+markers',
                            name='CAUE',
                            line=dict(color='#28a745', width=3)
                        ),
                        row=1, col=2
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
                    
                    # Marcar período óptimo
                    optimal_idx = results['periods'].index(results['optimal_period'])
                    fig.add_trace(
                        go.Scatter(
                            x=[results['optimal_period']], 
                            y=[results['caue_values'][optimal_idx]],
                            mode='markers',
                            name='Óptimo',
                            marker=dict(color='red', size=15, symbol='star')
                        ),
                        row=1, col=2
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=results['periods'], 
                            y=results['marginal_npv'],
                            mode='lines+markers',
                            name='VAN Marginal',
                            line=dict(color='#764ba2', width=3)
                        ),
                        row=1, col=2
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
                
                # Evolución comparativa
                fig.add_trace(
                    go.Scatter(
                        x=results['periods'], 
                        y=results['cumulative_npv'],
                        mode='lines+markers',
                        name='VAN',
                        line=dict(color='#667eea', width=2)
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=results['periods'], 
                        y=results['caue_values'],
                        mode='lines+markers',
                        name='CAUE',
                        line=dict(color='#28a745', width=2),
                        yaxis='y2'
                    ),
                    row=2, col=1
                )
                
                # Gráfico de barras con recomendación
                colors = ['red' if i != results['optimal_period'] else 'green' for i in results['periods']]
                metric_values = results['caue_values'] if is_repeatable else results['cumulative_npv']
                
                fig.add_trace(
                    go.Bar(
                        x=results['periods'],
                        y=metric_values,
                        name='Métrica Óptima',
                        marker_color=colors
                    ),
                    row=2, col=2
                )
                
                fig.update_layout(height=800, template='plotly_white')
                fig.update_xaxes(title_text="Período", row=1, col=1)
                fig.update_xaxes(title_text="Período", row=1, col=2)
                fig.update_xaxes(title_text="Período", row=2, col=1)
                fig.update_xaxes(title_text="Período", row=2, col=2)
                fig.update_yaxes(title_text="VAN ($)", row=1, col=1)
                fig.update_yaxes(title_text="CAUE ($)" if is_repeatable else "VAN Marginal ($)", row=1, col=2)
                fig.update_yaxes(title_text="VAN ($)", row=2, col=1)
                fig.update_yaxes(title_text="CAUE ($)" if is_repeatable else "VAN ($)", row=2, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recomendación
                st.subheader("🎯 Recomendación Estratégica")
                st.markdown(results['recommendation'])
                
                # Tabla detallada
                st.subheader("📋 Análisis Detallado por Período")
                df_optimal = pd.DataFrame({
                    'Período': results['periods'],
                    'VAN Acumulado ($)': [f"{npv:,.0f}" for npv in results['cumulative_npv']],
                    'VAN Marginal ($)': [f"{mnpv:,.0f}" for mnpv in results['marginal_npv']],
                    'CAUE ($)': [f"{caue:,.0f}" for caue in results['caue_values']],
                    'Recomendación': ['ÓPTIMO ⭐' if p == results['optimal_period'] else 'Subóptimo' for p in results['periods']]
                })
                
                st.dataframe(df_optimal, use_container_width=True)
                
                # Análisis económico adicional
                st.subheader("📊 Análisis Económico Adicional")
                
                col_eco1, col_eco2, col_eco3 = st.columns(3)
                
                optimal_idx = results['periods'].index(results['optimal_period'])
                optimal_npv = results['cumulative_npv'][optimal_idx]
                optimal_caue = results['caue_values'][optimal_idx]
                
                with col_eco1:
                    st.metric(
                        "VAN Óptimo", 
                        f"${optimal_npv:,.0f}",
                        f"Período {results['optimal_period']}"
                    )
                
                with col_eco2:
                    st.metric(
                        "CAUE Óptimo", 
                        f"${optimal_caue:,.0f}",
                        "Anual equivalente"
                    )
                
                with col_eco3:
                    if len(results['cumulative_npv']) > 1:
                        efficiency = optimal_npv / results['periods'][optimal_idx] if results['periods'][optimal_idx] > 0 else 0
                        st.metric(
                            "Eficiencia", 
                            f"${efficiency:,.0f}",
                            "VAN/Período"
                        )
                
                # Descarga de resultados
                st.markdown(
                    FinancialAnalyzer.create_downloadable_table(df_optimal, "momento_optimo"),
                    unsafe_allow_html=True
                )
    
    # Herramienta 5: Comparación de Proyectos
    elif tool == "⚖️ Comparación de Proyectos":
        st.header("⚖️ Comparación de Proyectos")
        
        st.info("Compare hasta 5 proyectos diferentes para tomar la mejor decisión de inversión")
        
        # Configuración general
        col_config1, col_config2, col_config3 = st.columns(3)
        with col_config1:
            num_projects = st.selectbox("Número de proyectos a comparar", [2, 3, 4, 5])
        with col_config2:
            common_discount_rate = st.number_input("Tasa de descuento común (%)", value=10.0) / 100
        with col_config3:
            analysis_type = st.selectbox("Tipo de análisis", ["Completo", "Básico"])
        
        # Datos de proyectos
        projects_data = []
        
        # Crear tabs para cada proyecto
        project_tabs = st.tabs([f"Proyecto {i+1}" for i in range(num_projects)])
        
        for i, tab in enumerate(project_tabs):
            with tab:
                st.subheader(f"🔧 Configuración Proyecto {i+1}")
                
                col_name, col_periods, col_type = st.columns(3)
                with col_name:
                    project_name = st.text_input("Nombre del proyecto", value=f"Proyecto {i+1}", key=f"name_{i}")
                with col_periods:
                    n_periods_proj = st.number_input("Períodos", min_value=2, max_value=30, value=5, key=f"periods_proj_{i}")
                with col_type:
                    proj_repeatable = st.checkbox("Repetible", key=f"repeat_{i}")
                
                # Flujos de caja del proyecto
                project_flows = []
                
                # Inversión inicial
                initial_inv = st.number_input("Inversión inicial ($)", value=-100000.0, key=f"inv_proj_{i}")
                project_flows.append(initial_inv)
                
                # Flujos futuros en cuadrícula
                st.write("**Flujos futuros:**")
                cols_per_row = 5
                rows_needed = (n_periods_proj - 1 + cols_per_row - 1) // cols_per_row
                
                for row in range(rows_needed):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        period = row * cols_per_row + col_idx + 1
                        if period < n_periods_proj:
                            with cols[col_idx]:
                                flow = st.number_input(f"Año {period}", value=30000.0, key=f"flow_proj_{i}_{period}")
                                project_flows.append(flow)
                
                projects_data.append({
                    'name': project_name,
                    'flows': project_flows,
                    'repeatable': proj_repeatable
                })
        
        if st.button("🔍 Comparar Todos los Proyectos", key="compare_projects"):
            st.subheader("📊 Resultados de la Comparación")
            
            # Calcular indicadores para todos los proyectos
            comparison_results = []
            
            for project in projects_data:
                npv_proj = FinancialAnalyzer.calculate_npv(project['flows'], common_discount_rate)
                irr_proj = FinancialAnalyzer.calculate_irr(project['flows'])
                payback_proj = FinancialAnalyzer.calculate_payback(project['flows'])
                discounted_payback_proj = FinancialAnalyzer.calculate_discounted_payback(project['flows'], common_discount_rate)
                pi_proj = FinancialAnalyzer.calculate_profitability_index(project['flows'], common_discount_rate)
                caue_proj = FinancialAnalyzer.calculate_caue(project['flows'], common_discount_rate)
                
                comparison_results.append({
                    'Proyecto': project['name'],
                    'Tipo': 'Repetible' if project['repeatable'] else 'Único',
                    'VAN ($)': f"{npv_proj:,.0f}",
                    'TIR (%)': f"{irr_proj*100:.2f}" if irr_proj else "N/A",
                    'CAUE ($)': f"{caue_proj:,.0f}",
                    'Payback Simple': f"{payback_proj}" if payback_proj else "No recupera",
                    'Payback Descontado': f"{discounted_payback_proj}" if discounted_payback_proj else "No recupera",
                    'Índice Rentabilidad': f"{pi_proj:.3f}",
                    'npv_numeric': npv_proj,
                    'irr_numeric': irr_proj if irr_proj else 0,
                    'caue_numeric': caue_proj,
                    'pi_numeric': pi_proj,
                    'repeatable': project['repeatable']
                })
            
            # Mostrar tabla comparativa completa
            df_comparison = pd.DataFrame(comparison_results)
            
            # Tabla básica o completa según selección
            if analysis_type == "Básico":
                columns_to_show = ['Proyecto', 'VAN ($)', 'TIR (%)', 'CAUE ($)', 'Índice Rentabilidad']
            else:
                columns_to_show = ['Proyecto', 'Tipo', 'VAN ($)', 'TIR (%)', 'CAUE ($)', 
                                 'Payback Simple', 'Payback Descontado', 'Índice Rentabilidad']
            
            st.dataframe(df_comparison[columns_to_show], use_container_width=True)
            
            # Descarga de tabla completa
            st.markdown(
                FinancialAnalyzer.create_downloadable_table(df_comparison, "comparacion_proyectos"),
                unsafe_allow_html=True
            )
            
            # Gráficos comparativos avanzados
            tab_graf1, tab_graf2, tab_graf3 = st.tabs(["Indicadores Principales", "Análisis de Riesgo", "Ranking Integral"])
            
            with tab_graf1:
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    fig_npv = px.bar(
                        df_comparison, 
                        x='Proyecto', 
                        y='npv_numeric',
                        title='Comparación VAN',
                        color='npv_numeric',
                        color_continuous_scale='RdYlGn',
                        text='VAN ($)'
                    )
                    fig_npv.update_layout(yaxis_title='VAN ($)', template='plotly_white')
                    fig_npv.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_npv, use_container_width=True)
                
                with col_chart2:
                    fig_caue = px.bar(
                        df_comparison, 
                        x='Proyecto', 
                        y='caue_numeric',
                        title='Comparación CAUE',
                        color='caue_numeric',
                        color_continuous_scale='RdYlGn',
                        text='CAUE ($)'
                    )
                    fig_caue.update_layout(yaxis_title='CAUE ($)', template='plotly_white')
                    fig_caue.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_caue, use_container_width=True)
                
                # TIR vs Índice de Rentabilidad
                # Crear tamaños normalizados y positivos para el scatter
                df_comparison['size_normalized'] = df_comparison['npv_numeric'].apply(
                    lambda x: max(abs(x), 1000)  # Usar valor absoluto con mínimo de 1000
                )
                
                fig_scatter = px.scatter(
                    df_comparison,
                    x='irr_numeric',
                    y='pi_numeric',
                    size='size_normalized',
                    color='Proyecto',
                    hover_name='Proyecto',
                    title='Matriz TIR vs Índice de Rentabilidad',
                    labels={
                        'irr_numeric': 'TIR',
                        'pi_numeric': 'Índice de Rentabilidad',
                        'size_normalized': 'VAN Absoluto ($)'
                    },
                    hover_data={
                        'npv_numeric': ':,.0f',
                        'size_normalized': False
                    }
                )
                fig_scatter.update_layout(
                    xaxis_tickformat='.2%',
                    height=500,
                    template='plotly_white'
                )
                fig_scatter.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="IP = 1")
                fig_scatter.add_vline(x=common_discount_rate, line_dash="dash", line_color="red", 
                                     annotation_text=f"Tasa objetivo = {common_discount_rate*100:.1f}%")
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab_graf2:
                st.subheader("📊 Análisis de Riesgo-Rentabilidad")
                
                # Calcular métricas de riesgo para cada proyecto
                risk_analysis = []
                for i, project in enumerate(projects_data):
                    # Análisis de sensibilidad rápido
                    rates_risk, npvs_risk = FinancialAnalyzer.sensitivity_analysis(
                        project['flows'], common_discount_rate, 0.05, 11
                    )
                    
                    npv_volatility = np.std(npvs_risk)
                    min_npv = min(npvs_risk)
                    max_npv = max(npvs_risk)
                    
                    risk_analysis.append({
                        'Proyecto': project['name'],
                        'Volatilidad VAN': npv_volatility,
                        'VAN Mínimo': min_npv,
                        'VAN Máximo': max_npv,
                        'VAN Base': comparison_results[i]['npv_numeric']
                    })
                
                df_risk = pd.DataFrame(risk_analysis)
                
                col_risk1, col_risk2 = st.columns(2)
                
                with col_risk1:
                    # Crear tamaños seguros para el gráfico de riesgo
                    df_risk['size_safe'] = df_risk['VAN Base'].apply(lambda x: max(abs(x), 1000))
                    
                    fig_risk1 = px.scatter(
                        df_risk,
                        x='Volatilidad VAN',
                        y='VAN Base',
                        size='size_safe',
                        color='Proyecto',
                        title='Riesgo vs Rentabilidad',
                        hover_data={
                            'VAN Base': ':,.0f',
                            'Volatilidad VAN': ':,.0f',
                            'size_safe': False
                        }
                    )
                    fig_risk1.update_layout(template='plotly_white')
                    st.plotly_chart(fig_risk1, use_container_width=True)
                
                with col_risk2:
                    # Gráfico de barras con rangos de VAN
                    fig_range = go.Figure()
                    
                    for i, row in df_risk.iterrows():
                        fig_range.add_trace(go.Bar(
                            name=row['Proyecto'],
                            x=[row['Proyecto']],
                            y=[row['VAN Base']],
                            error_y=dict(
                                type='data',
                                symmetric=False,
                                array=[row['VAN Máximo'] - row['VAN Base']],
                                arrayminus=[row['VAN Base'] - row['VAN Mínimo']]
                            )
                        ))
                    
                    fig_range.update_layout(
                        title='Rangos de VAN (Análisis de Sensibilidad)',
                        yaxis_title='VAN ($)',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_range, use_container_width=True)
                
                st.dataframe(df_risk, use_container_width=True)
            
            with tab_graf3:
                st.subheader("🏆 Ranking Integral de Proyectos")
                
                # Sistema de puntuación integral
                scoring_criteria = {
                    'VAN': 0.3,
                    'TIR': 0.25,
                    'CAUE': 0.2,
                    'Índice_Rentabilidad': 0.15,
                    'Payback': 0.1
                }
                
                # Normalizar y calcular puntuación
                df_scoring = df_comparison.copy()
                
                # Convertir payback a numérico (menor es mejor)
                df_scoring['payback_numeric'] = df_scoring['Payback Simple'].apply(
                    lambda x: float(x) if x != 'No recupera' else 999
                )
                
                # Normalizar métricas (0-100) con manejo de casos especiales
                for metric in ['npv_numeric', 'irr_numeric', 'caue_numeric', 'pi_numeric']:
                    metric_min = df_scoring[metric].min()
                    metric_max = df_scoring[metric].max()
                    
                    if metric_max != metric_min and not pd.isna(metric_max) and not pd.isna(metric_min):
                        df_scoring[f'{metric}_norm'] = (
                            (df_scoring[metric] - metric_min) / 
                            (metric_max - metric_min) * 100
                        )
                    else:
                        df_scoring[f'{metric}_norm'] = 50
                    
                    # Asegurar que no haya valores negativos o NaN
                    df_scoring[f'{metric}_norm'] = df_scoring[f'{metric}_norm'].fillna(0).clip(0, 100)
                
                # Payback normalizado (invertido - menor es mejor) con manejo seguro
                payback_min = df_scoring['payback_numeric'].min()
                payback_max = df_scoring['payback_numeric'].max()
                
                if payback_max != payback_min and payback_max < 900:  # Evitar valores "No recupera"
                    df_scoring['payback_norm'] = (
                        100 - (df_scoring['payback_numeric'] - payback_min) / 
                        (payback_max - payback_min) * 100
                    ).clip(0, 100)
                else:
                    # Si todos tienen el mismo payback o hay "No recupera", asignar valor neutro
                    df_scoring['payback_norm'] = df_scoring['payback_numeric'].apply(
                        lambda x: 50 if x < 900 else 0
                    )
                
                # Calcular puntuación total
                df_scoring['Puntuación_Total'] = (
                    df_scoring['npv_numeric_norm'] * scoring_criteria['VAN'] +
                    df_scoring['irr_numeric_norm'] * scoring_criteria['TIR'] +
                    df_scoring['caue_numeric_norm'] * scoring_criteria['CAUE'] +
                    df_scoring['pi_numeric_norm'] * scoring_criteria['Índice_Rentabilidad'] +
                    df_scoring['payback_norm'] * scoring_criteria['Payback']
                )
                
                # Ordenar por puntuación
                df_scoring_sorted = df_scoring.sort_values('Puntuación_Total', ascending=False)
                
                # Crear ranking visual
                col_rank1, col_rank2 = st.columns(2)
                
                with col_rank1:
                    fig_ranking = px.bar(
                        df_scoring_sorted,
                        x='Puntuación_Total',
                        y='Proyecto',
                        orientation='h',
                        title='Ranking Integral de Proyectos',
                        color='Puntuación_Total',
                        color_continuous_scale='RdYlGn'
                    )
                    fig_ranking.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig_ranking, use_container_width=True)
                
                with col_rank2:
                    # Radar chart del mejor proyecto
                    best_project = df_scoring_sorted.iloc[0]
                    
                    categories = ['VAN', 'TIR', 'CAUE', 'Í. Rentabilidad', 'Payback']
                    values = [
                        best_project['npv_numeric_norm'],
                        best_project['irr_numeric_norm'],
                        best_project['caue_numeric_norm'],
                        best_project['pi_numeric_norm'],
                        best_project['payback_norm']
                    ]
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]],  # Cerrar el polígono
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=best_project['Proyecto'],
                        line_color='#667eea'
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )
                        ),
                        title=f"Perfil del Mejor Proyecto: {best_project['Proyecto']}",
                        template='plotly_white',
                        height=400
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                # Tabla de ranking detallada
                ranking_table = pd.DataFrame({
                    'Ranking': range(1, len(df_scoring_sorted) + 1),
                    'Proyecto': df_scoring_sorted['Proyecto'],
                    'Puntuación Total': [f"{score:.1f}" for score in df_scoring_sorted['Puntuación_Total']],
                    'VAN Score': [f"{score:.1f}" for score in df_scoring_sorted['npv_numeric_norm']],
                    'TIR Score': [f"{score:.1f}" for score in df_scoring_sorted['irr_numeric_norm']],
                    'CAUE Score': [f"{score:.1f}" for score in df_scoring_sorted['caue_numeric_norm']],
                    'IP Score': [f"{score:.1f}" for score in df_scoring_sorted['pi_numeric_norm']],
                    'Payback Score': [f"{score:.1f}" for score in df_scoring_sorted['payback_norm']]
                })
                
                st.subheader("📋 Tabla de Ranking Detallada")
                st.dataframe(ranking_table, use_container_width=True)
                
                # Explicación del sistema de puntuación
                with st.expander("ℹ️ Metodología de Puntuación"):
                    st.write("""
                    **Sistema de Puntuación Integral:**
                    
                    El ranking se calcula usando un sistema ponderado donde cada métrica contribuye según su importancia:
                    
                    • **VAN (30%):** Valor absoluto creado por el proyecto
                    • **TIR (25%):** Rentabilidad relativa del proyecto  
                    • **CAUE (20%):** Equivalencia anual (importante para proyectos repetibles)
                    • **Índice Rentabilidad (15%):** Eficiencia de la inversión
                    • **Payback (10%):** Velocidad de recuperación
                    
                    Cada métrica se normaliza a una escala 0-100, donde 100 representa el mejor desempeño.
                    """)
            
            # Ranking y recomendaciones finales
            st.subheader("🏆 Ranking y Recomendaciones Finales")
            
            # Identificar mejores proyectos por cada criterio
            best_van = df_comparison.loc[df_comparison['npv_numeric'].idxmax()]
            best_tir = df_comparison.loc[df_comparison['irr_numeric'].idxmax()]
            best_caue = df_comparison.loc[df_comparison['caue_numeric'].idxmax()]
            best_pi = df_comparison.loc[df_comparison['pi_numeric'].idxmax()]
            
            col_final1, col_final2, col_final3, col_final4 = st.columns(4)
            
            with col_final1:
                st.success(f"""
                **🥇 Mejor VAN**
                
                **{best_van['Proyecto']}**
                
                VAN: ${best_van['npv_numeric']:,.0f}
                """)
            
            with col_final2:
                st.success(f"""
                **🎯 Mejor TIR**
                
                **{best_tir['Proyecto']}**
                
                TIR: {best_tir['irr_numeric']*100:.2f}%
                """)
            
            with col_final3:
                st.success(f"""
                **💎 Mejor CAUE**
                
                **{best_caue['Proyecto']}**
                
                CAUE: ${best_caue['caue_numeric']:,.0f}
                """)
            
            with col_final4:
                st.success(f"""
                **⚡ Mejor Eficiencia**
                
                **{best_pi['Proyecto']}**
                
                Índice: {best_pi['pi_numeric']:.3f}
                """)
            
            # Recomendación final integral
            st.subheader("💡 Recomendación Final Integral")
            
            # Contar proyectos viables
            viable_projects = [p for p in comparison_results if p['npv_numeric'] > 0]
            repeatable_projects = [p for p in comparison_results if p['repeatable'] and p['caue_numeric'] > 0]
            
            if len(viable_projects) == 0:
                st.error("❌ **Ningún proyecto es viable financieramente.** Considere replantear los proyectos o revisar los supuestos.")
            else:
                best_overall = df_scoring_sorted.iloc[0]
                
                recommendation_text = f"""
                🎯 **Recomendación Principal:** Ejecutar **{best_overall['Proyecto']}**
                
                **Justificación:**
                • Puntuación integral más alta: {best_overall['Puntuación_Total']:.1f}/100
                • VAN: ${best_overall['npv_numeric']:,.0f}
                • TIR: {best_overall['irr_numeric']*100:.2f}%
                • CAUE: ${best_overall['caue_numeric']:,.0f}
                
                **Análisis por tipo de proyecto:**
                """
                
                if len(repeatable_projects) > 0:
                    best_repeatable = max(repeatable_projects, key=lambda x: x['caue_numeric'])
                    recommendation_text += f"""
                • **Para proyectos repetibles:** {best_repeatable['Proyecto']} (CAUE: ${best_repeatable['caue_numeric']:,.0f})"""
                
                unique_projects = [p for p in comparison_results if not p['repeatable'] and p['npv_numeric'] > 0]
                if len(unique_projects) > 0:
                    best_unique = max(unique_projects, key=lambda x: x['npv_numeric'])
                    recommendation_text += f"""
                • **Para proyectos únicos:** {best_unique['Proyecto']} (VAN: ${best_unique['npv_numeric']:,.0f})"""
                
                recommendation_text += f"""
                
                **Consideraciones adicionales:**
                • {len(viable_projects)} de {len(comparison_results)} proyectos son financieramente viables
                • Evalúe disponibilidad de capital y capacidad de ejecución
                • Considere factores cualitativos no incluidos en el análisis financiero
                """
                
                st.info(recommendation_text)
    
    # Footer moderno
    st.markdown("""
    <div class="modern-footer">
        <h3>📊 Evaluador de Proyectos</h3>
        <p>Herramienta avanzada para análisis financiero y toma de decisiones de inversión</p>
        <p><strong>Características:</strong> VAN • TIR • CAUE • Análisis de Sensibilidad • Momento Óptimo • Comparación Integral</p>
        <p><em>Desarrollado por Helena De La Cruz Vergara - Ingeniería Comercial UdeC</em></p>
        <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
            💡 Ideal para estudiantes, profesores y profesionales en evaluación de proyectos
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()