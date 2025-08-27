"""
Analizador Económico de Proyectos - Versión Mejorada
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
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Analizador Económico de Proyectos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .danger-metric {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


class FinancialAnalyzer:
    """Clase para análisis financiero de proyectos"""
    
  @staticmethod
    def calculate_npv(cash_flows, discount_rate):
        """
        Calcula el Valor Actual Neto (VAN/NPV)
        
        Fórmula: VAN = Σ(CFt / (1+r)^t) donde:
        - CFt = Flujo de caja en el período t
        - r = tasa de descuento
        - t = período (0, 1, 2, ...)
        
        Período 0: Inversión inicial (no se descuenta)
        Período 1+: Flujos futuros (se descuentan)
        """
        npv_value = 0
        for period, flow in enumerate(cash_flows):
            discount_factor = (1 + discount_rate) ** period
            present_value = flow / discount_factor
            npv_value += present_value
        return npv_value
    
    @staticmethod
    def calculate_irr(cash_flows, max_iterations=100):
        """Calcula la Tasa Interna de Retorno (TIR/IRR)"""
        def npv_function(rate):
            return FinancialAnalyzer.calculate_npv(cash_flows, rate)
        
        # Intentar diferentes valores iniciales
        initial_guesses = [0.1, 0.2, 0.5, -0.1, -0.5, 1.0]
        
        for guess in initial_guesses:
            try:
                irr_value = fsolve(npv_function, guess, maxfev=max_iterations)[0]
                # Verificar que la solución es válida
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
    def calculate_simple_interest(principal, rate, time):
        """Calcula interés simple"""
        return principal * (1 + rate * time)
    
    @staticmethod
    def calculate_compound_interest(principal, rate, time, compounding_frequency=1):
        """Calcula interés compuesto"""
        return principal * (1 + rate / compounding_frequency) ** (compounding_frequency * time)
    
    @staticmethod
    def optimal_timing_analysis(cash_flows, discount_rate):
        """
        Análisis del momento óptimo basado en teoría financiera
        
        Teoría: El momento óptimo de terminación de un proyecto es cuando
        la TIR marginal iguala la tasa de descuento (costo de oportunidad)
        """
        results = {
            'periods': [],
            'cumulative_npv': [],
            'marginal_irr': [],
            'total_irr': [],
            'recommendation': ''
        }
        
        n_periods = len(cash_flows)
        
        for period in range(2, n_periods + 1):
            period_flows = cash_flows[:period]
            
            # NPV acumulado
            npv = FinancialAnalyzer.calculate_npv(period_flows, discount_rate)
            results['cumulative_npv'].append(npv)
            
            # TIR total hasta el período
            total_irr = FinancialAnalyzer.calculate_irr(period_flows)
            results['total_irr'].append(total_irr if total_irr else 0)
            
            # TIR marginal (solo del último flujo vs inversión total)
            if period > 2:
                marginal_flow = [cash_flows[0], cash_flows[period-1]]
                marginal_irr = FinancialAnalyzer.calculate_irr(marginal_flow)
                results['marginal_irr'].append(marginal_irr if marginal_irr else 0)
            else:
                results['marginal_irr'].append(total_irr if total_irr else 0)
            
            results['periods'].append(period)
        
        # Encontrar momento óptimo
        optimal_period = 2
        max_npv = float('-inf')
        
        for i, npv in enumerate(results['cumulative_npv']):
            if npv > max_npv:
                max_npv = npv
                optimal_period = results['periods'][i]
        
        # Recomendación basada en teoría
        if max_npv > 0:
            results['recommendation'] = f"""
            **Momento Óptimo:** Período {optimal_period}
            
            **Razón:** Maximiza el VAN del proyecto ({max_npv:,.2f})
            
            **Criterio teórico:** Continuar mientras VAN marginal > 0
            """
        else:
            results['recommendation'] = """
            **Recomendación:** No ejecutar el proyecto
            
            **Razón:** VAN negativo en todos los períodos analizados
            """
        
        return results


def create_cash_flow_input(key_prefix, max_periods=15):
    """Crea interfaz para entrada de flujos de caja"""
    st.subheader("💰 Flujos de Caja")
    
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
    
    # Crear columnas para flujos futuros
    cols_per_row = 3
    for i in range(1, n_periods):
        if (i - 1) % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        
        col_index = (i - 1) % cols_per_row
        with cols[col_index]:
            flow = st.number_input(
                f"Período {i}", 
                value=30000.0,
                key=f"{key_prefix}_flow_{i}"
            )
            cash_flows.append(flow)
    
    return cash_flows


def main():
    """Función principal de la aplicación"""
    
    # Título principal
    st.title("📊 Analizador Económico de Proyectos")
    st.markdown("*Herramienta profesional para análisis de inversiones*")
    st.markdown("---")
    
    # Sidebar para navegación
    st.sidebar.title("🎯 Herramientas Financieras")
    st.sidebar.markdown("Selecciona la herramienta que deseas utilizar:")
    
    tool = st.sidebar.selectbox(
        "",
        [
            "🏦 Calculadora de Intereses",
            "📈 Análisis VAN/TIR", 
            "⏰ Análisis de Momento Óptimo",
            "⚖️ Comparación de Proyectos"
        ]
    )
    
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
                    final_amount = FinancialAnalyzer.calculate_simple_interest(
                        principal_simple, rate_simple, time_simple
                    )
                    interest_earned = final_amount - principal_simple
                    
                    st.success(f"**Monto final:** ${final_amount:,.2f}")
                    st.info(f"**Interés ganado:** ${interest_earned:,.2f}")
                    
                    # Gráfico de crecimiento
                    years = np.linspace(0, time_simple, int(time_simple * 4) + 1)
                    amounts = [FinancialAnalyzer.calculate_simple_interest(principal_simple, rate_simple, t) for t in years]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=years, y=amounts, mode='lines', name='Crecimiento'))
                    fig.update_layout(title="Crecimiento con Interés Simple", xaxis_title="Años", yaxis_title="Monto ($)")
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
                    final_amount = FinancialAnalyzer.calculate_compound_interest(
                        principal_compound, rate_compound, time_compound, compounding
                    )
                    interest_earned = final_amount - principal_compound
                    
                    st.success(f"**Monto final:** ${final_amount:,.2f}")
                    st.info(f"**Interés ganado:** ${interest_earned:,.2f}")
                    
                    # Comparación con interés simple
                    simple_amount = FinancialAnalyzer.calculate_simple_interest(
                        principal_compound, rate_compound, time_compound
                    )
                    difference = final_amount - simple_amount
                    st.metric("Ventaja del interés compuesto", f"${difference:,.2f}")
    
    # Herramienta 2: Análisis VAN/TIR
    elif tool == "📈 Análisis VAN/TIR":
        st.header("📈 Análisis VAN/TIR")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Parámetros del Proyecto")
            discount_rate = st.number_input("Tasa de descuento (%)", value=10.0) / 100
            
            cash_flows = create_cash_flow_input("van_tir")
        
        with col2:
            if st.button("🔬 Calcular Indicadores Financieros", key="calc_indicators"):
                # Cálculos
                npv_result = FinancialAnalyzer.calculate_npv(cash_flows, discount_rate)
                irr_result = FinancialAnalyzer.calculate_irr(cash_flows)
                payback_result = FinancialAnalyzer.calculate_payback(cash_flows)
                pi_result = FinancialAnalyzer.calculate_profitability_index(cash_flows, discount_rate)
                
                # Resultados principales
                st.subheader("📊 Resultados del Análisis")
                
                col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                
                with col_res1:
                    npv_color = "normal" if npv_result > 0 else "inverse"
                    st.metric(
                        "VAN ($)", 
                        f"{npv_result:,.0f}",
                        delta="Rentable ✅" if npv_result > 0 else "No rentable ❌"
                    )
                
                with col_res2:
                    if irr_result is not None:
                        irr_vs_discount = "Aceptable ✅" if irr_result > discount_rate else "Rechazar ❌"
                        st.metric("TIR (%)", f"{irr_result*100:.2f}", delta=irr_vs_discount)
                    else:
                        st.metric("TIR", "No calculable")
                
                with col_res3:
                    if payback_result is not None:
                        st.metric("Payback", f"{payback_result} períodos")
                    else:
                        st.metric("Payback", "No recupera")
                
                with col_res4:
                    pi_status = "Rentable ✅" if pi_result > 1 else "No rentable ❌"
                    st.metric("Índice Rentabilidad", f"{pi_result:.2f}", delta=pi_status)
                
                # Análisis de sensibilidad VAN vs Tasa de descuento
                st.subheader("📈 Análisis de Sensibilidad")
                
                rates = np.linspace(0, 0.5, 100)
                npvs = [FinancialAnalyzer.calculate_npv(cash_flows, rate) for rate in rates]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rates*100, y=npvs, mode='lines', name='VAN', line=dict(width=3)))
                fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="VAN = 0")
                
                if irr_result is not None and 0 <= irr_result <= 0.5:
                    fig.add_vline(
                        x=irr_result*100, 
                        line_dash="dash", 
                        line_color="green", 
                        annotation_text=f"TIR: {irr_result*100:.2f}%"
                    )
                
                fig.update_layout(
                    title="Perfil VAN - Tasa de Descuento",
                    xaxis_title="Tasa de descuento (%)",
                    yaxis_title="VAN ($)",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de flujos descontados
                st.subheader("💰 Análisis Detallado de Flujos")
                
                periods = list(range(len(cash_flows)))
                discount_factors = [1 / (1 + discount_rate) ** i for i in periods]
                present_values = [cf * df for cf, df in zip(cash_flows, discount_factors)]
                
                df_flows = pd.DataFrame({
                    'Período': periods,
                    'Flujo de Caja ($)': cash_flows,
                    'Factor Descuento': discount_factors,
                    'Valor Presente ($)': present_values
                })
                
                st.dataframe(df_flows, use_container_width=True)
    
    # Herramienta 3: Análisis de Momento Óptimo
    elif tool == "⏰ Análisis de Momento Óptimo":
        st.header("⏰ Análisis de Momento Óptimo")
        
        st.info("""
        **Teoría del Momento Óptimo:**
        - Un proyecto debe continuarse mientras genere valor (VAN marginal > 0)
        - El momento óptimo de finalización es cuando se maximiza el VAN total
        - La TIR marginal debe compararse con el costo de oportunidad
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuración del Análisis")
            discount_rate_opt = st.number_input("Tasa de descuento (%)", value=12.0, key="opt_rate") / 100
            
            cash_flows_opt = create_cash_flow_input("optimal", max_periods=15)
        
        with col2:
            if st.button("🎯 Realizar Análisis de Momento Óptimo", key="calc_optimal"):
                results = FinancialAnalyzer.optimal_timing_analysis(cash_flows_opt, discount_rate_opt)
                
                st.subheader("📊 Resultados del Análisis")
                
                # Gráficos del análisis
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[
                        'VAN Acumulado por Período',
                        'TIR Total vs TIR Marginal',
                        'Evolución de Indicadores',
                        'Decisión de Continuidad'
                    ],
                    specs=[[{"secondary_y": False}, {"secondary_y": True}],
                           [{"secondary_y": True}, {"secondary_y": False}]]
                )
                
                # VAN Acumulado
                fig.add_trace(
                    go.Scatter(
                        x=results['periods'], 
                        y=results['cumulative_npv'],
                        mode='lines+markers',
                        name='VAN Acumulado',
                        line=dict(color='blue', width=3)
                    ),
                    row=1, col=1
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
                
                # TIR Total vs Marginal
                fig.add_trace(
                    go.Scatter(
                        x=results['periods'], 
                        y=[r*100 if r else 0 for r in results['total_irr']],
                        mode='lines+markers',
                        name='TIR Total',
                        line=dict(color='green', width=2)
                    ),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(
                        x=results['periods'], 
                        y=[r*100 if r else 0 for r in results['marginal_irr']],
                        mode='lines+markers',
                        name='TIR Marginal',
                        line=dict(color='orange', width=2, dash='dot')
                    ),
                    row=1, col=2
                )
                fig.add_hline(y=discount_rate_opt*100, line_dash="dash", line_color="red", 
                             annotation_text=f"Tasa objetivo: {discount_rate_opt*100:.1f}%", row=1, col=2)
                
                fig.update_layout(height=800, showlegend=True, title_text="Análisis Integral de Momento Óptimo")
                fig.update_xaxes(title_text="Período", row=1, col=1)
                fig.update_xaxes(title_text="Período", row=1, col=2)
                fig.update_yaxes(title_text="VAN ($)", row=1, col=1)
                fig.update_yaxes(title_text="TIR (%)", row=1, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recomendación
                st.subheader("🎯 Recomendación Estratégica")
                st.markdown(results['recommendation'])
                
                # Tabla detallada
                st.subheader("📋 Análisis Detallado por Período")
                df_optimal = pd.DataFrame({
                    'Período': results['periods'],
                    'VAN Acumulado ($)': [f"{npv:,.0f}" for npv in results['cumulative_npv']],
                    'TIR Total (%)': [f"{irr*100:.2f}" if irr else "N/A" for irr in results['total_irr']],
                    'TIR Marginal (%)': [f"{irr*100:.2f}" if irr else "N/A" for irr in results['marginal_irr']]
                })
                
                st.dataframe(df_optimal, use_container_width=True)
    
    # Herramienta 4: Comparación de Proyectos
    elif tool == "⚖️ Comparación de Proyectos":
        st.header("⚖️ Comparación de Proyectos")
        
        st.info("Compare hasta 4 proyectos diferentes para tomar la mejor decisión de inversión")
        
        # Configuración general
        col_config1, col_config2 = st.columns(2)
        with col_config1:
            num_projects = st.selectbox("Número de proyectos a comparar", [2, 3, 4])
        with col_config2:
            common_discount_rate = st.number_input("Tasa de descuento común (%)", value=10.0) / 100
        
        # Datos de proyectos
        projects_data = []
        
        # Crear tabs para cada proyecto
        project_tabs = st.tabs([f"Proyecto {i+1}" for i in range(num_projects)])
        
        for i, tab in enumerate(project_tabs):
            with tab:
                st.subheader(f"📁 Configuración Proyecto {i+1}")
                
                col_name, col_periods = st.columns(2)
                with col_name:
                    project_name = st.text_input("Nombre del proyecto", value=f"Proyecto {i+1}", key=f"name_{i}")
                with col_periods:
                    n_periods_proj = st.number_input("Períodos", min_value=2, max_value=15, value=5, key=f"periods_proj_{i}")
                
                # Flujos de caja del proyecto
                project_flows = []
                
                # Inversión inicial
                initial_inv = st.number_input("Inversión inicial ($)", value=-100000.0, key=f"inv_proj_{i}")
                project_flows.append(initial_inv)
                
                # Flujos futuros en columnas
                st.write("**Flujos futuros:**")
                cols_per_row = 4
                for j in range(1, n_periods_proj):
                    if (j - 1) % cols_per_row == 0:
                        cols = st.columns(cols_per_row)
                    
                    col_index = (j - 1) % cols_per_row
                    with cols[col_index]:
                        flow = st.number_input(f"Año {j}", value=30000.0, key=f"flow_proj_{i}_{j}")
                        project_flows.append(flow)
                
                projects_data.append({
                    'name': project_name,
                    'flows': project_flows
                })
        
        if st.button("🔍 Comparar Todos los Proyectos", key="compare_projects"):
            st.subheader("📊 Resultados de la Comparación")
            
            # Calcular indicadores para todos los proyectos
            comparison_results = []
            
            for project in projects_data:
                npv_proj = FinancialAnalyzer.calculate_npv(project['flows'], common_discount_rate)
                irr_proj = FinancialAnalyzer.calculate_irr(project['flows'])
                payback_proj = FinancialAnalyzer.calculate_payback(project['flows'])
                pi_proj = FinancialAnalyzer.calculate_profitability_index(project['flows'], common_discount_rate)
                
                comparison_results.append({
                    'Proyecto': project['name'],
                    'VAN ($)': f"{npv_proj:,.0f}",
                    'TIR (%)': f"{irr_proj*100:.2f}" if irr_proj else "N/A",
                    'Payback': f"{payback_proj}" if payback_proj else "No recupera",
                    'Índice Rentabilidad': f"{pi_proj:.2f}",
                    'npv_numeric': npv_proj,
                    'irr_numeric': irr_proj if irr_proj else 0,
                    'pi_numeric': pi_proj
                })
            
            # Mostrar tabla comparativa
            df_comparison = pd.DataFrame(comparison_results)
            st.dataframe(
                df_comparison[['Proyecto', 'VAN ($)', 'TIR (%)', 'Payback', 'Índice Rentabilidad']], 
                use_container_width=True
            )
            
            # Gráficos comparativos
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig_npv = px.bar(
                    df_comparison, 
                    x='Proyecto', 
                    y='npv_numeric',
                    title='Comparación VAN',
                    color='npv_numeric',
                    color_continuous_scale='RdYlGn'
                )
                fig_npv.update_layout(yaxis_title='VAN ($)')
                fig_npv.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_npv, use_container_width=True)
            
            with col_chart2:
                fig_irr = px.bar(
                    df_comparison, 
                    x='Proyecto', 
                    y='irr_numeric',
                    title='Comparación TIR',
                    color='irr_numeric',
                    color_continuous_scale='RdYlGn'
                )
                fig_irr.update_layout(
                    yaxis_title='TIR',
                    yaxis_tickformat='.2%'
                )
                fig_irr.add_hline(y=common_discount_rate, line_dash="dash", line_color="red", 
                                 annotation_text=f"Tasa objetivo: {common_discount_rate*100:.1f}%")
                st.plotly_chart(fig_irr, use_container_width=True)
            
            # Ranking y recomendaciones
            st.subheader("🏆 Ranking y Recomendaciones")
            
            col_rank1, col_rank2, col_rank3 = st.columns(3)
            
            # Mejor por VAN
            best_npv = df_comparison.loc[df_comparison['npv_numeric'].idxmax()]
            with col_rank1:
                st.success(f"""
                **🥇 Mejor VAN**
                
                **{best_npv['Proyecto']}**
                
                VAN: ${best_npv['npv_numeric']:,.0f}
                """)
            
            # Mejor por TIR
            best_irr = df_comparison.loc[df_comparison['irr_numeric'].idxmax()]
            with col_rank2:
                st.success(f"""
                **🎯 Mejor TIR**
                
                **{best_irr['Proyecto']}**
                
                TIR: {best_irr['irr_numeric']*100:.2f}%
                """)
            
            # Mejor índice de rentabilidad
            best_pi = df_comparison.loc[df_comparison['pi_numeric'].idxmax()]
            with col_rank3:
                st.success(f"""
                **💎 Mejor Índice**
                
                **{best_pi['Proyecto']}**
                
                Índice: {best_pi['pi_numeric']:.2f}
                """)
            
            # Análisis de riesgo-rentabilidad
            st.subheader("📈 Análisis Riesgo-Rentabilidad")
            
            fig_scatter = px.scatter(
                df_comparison,
                x='irr_numeric',
                y='npv_numeric',
                size='pi_numeric',
                hover_name='Proyecto',
                title='Matriz Riesgo-Rentabilidad',
                labels={
                    'irr_numeric': 'TIR',
                    'npv_numeric': 'VAN ($)',
                    'pi_numeric': 'Índice de Rentabilidad'
                }
            )
            fig_scatter.update_layout(
                xaxis_tickformat='.2%',
                height=500
            )
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="VAN = 0")
            fig_scatter.add_vline(x=common_discount_rate, line_dash="dash", line_color="red", 
                                 annotation_text=f"Tasa objetivo = {common_discount_rate*100:.1f}%")
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Recomendación final
            st.subheader("💡 Recomendación Final")
            
            # Contar proyectos rentables
            profitable_projects = len([p for p in comparison_results if p['npv_numeric'] > 0])
            
            if profitable_projects == 0:
                st.error("❌ **Ningún proyecto es rentable.** Considere replantear los proyectos o buscar alternativas.")
            elif profitable_projects == 1:
                profitable = [p for p in comparison_results if p['npv_numeric'] > 0][0]
                st.success(f"✅ **Recomendación:** Ejecutar **{profitable['Proyecto']}** (único proyecto rentable)")
            else:
                # Análisis multifactorial para múltiples proyectos rentables
                st.info(f"""
                📊 **Análisis Multifactorial ({profitable_projects} proyectos rentables):**
                
                - **Para maximizar valor:** {best_npv['Proyecto']} (VAN: ${best_npv['npv_numeric']:,.0f})
                - **Para maximizar rentabilidad:** {best_irr['Proyecto']} (TIR: {best_irr['irr_numeric']*100:.2f}%)
                - **Para optimizar eficiencia:** {best_pi['Proyecto']} (Índice: {best_pi['pi_numeric']:.2f})
                
                **Recomendación:** Si tiene capital limitado, priorice por índice de rentabilidad. 
                Si busca maximizar valor absoluto, elija el proyecto con mayor VAN.
                """)

    # Información adicional en sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Información")
    st.sidebar.info("""
    **Indicadores Financieros:**
    
    • **VAN (NPV):** Valor presente de flujos futuros
    • **TIR (IRR):** Tasa que iguala VAN a cero
    • **Payback:** Período de recuperación
    • **Índice de Rentabilidad:** VAN/Inversión inicial
    """)
    
    st.sidebar.markdown("### 🔧 Criterios de Decisión")
    st.sidebar.success("""
    **Proyecto Rentable si:**
    • VAN > 0
    • TIR > Tasa de descuento
    • Índice de Rentabilidad > 1
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>📊 Analizador Económico de Proyectos</strong></p>
        <p>Herramienta profesional para análisis financiero y toma de decisiones de inversión</p>
        <p><em>Versión 2.0 - Análisis financiero avanzado</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()