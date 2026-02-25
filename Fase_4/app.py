# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. CONFIGURAÇÃO DA PÁGINA (DEVE SER O PRIMEIRO COMANDO STREAMLIT)
st.set_page_config(page_title="Sistema Médico - Obesidade", layout="wide")

# 2. DESCOBRIR O CAMINHO EXATO DA PASTA DO ARQUIVO
# Isso resolve o erro de FileNotFoundError no Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 3. CARREGAMENTO DOS DADOS E MODELO
@st.cache_data
def load_data():
    caminho_csv = os.path.join(BASE_DIR, 'Obesity.csv')
    df = pd.read_csv(caminho_csv)
    return df

@st.cache_resource
def load_model():
    caminho_modelo = os.path.join(BASE_DIR, 'model_obesity.pkl')
    return joblib.load(caminho_modelo)

df = load_data()
model = load_model()

# 4. INTERFACE PRINCIPAL (TABS)
st.title("🩺 Sistema de Apoio à Decisão: Risco de Obesidade")
tab1, tab2 = st.tabs(["🤖 Preditor Analítico (IA)", "📊 Dashboard de Insights Médicos"])

# ==========================================
# ABA 1: PREDITOR DE OBESIDADE (MACHINE LEARNING)
# ==========================================
with tab1:
    st.header("Simulador de Diagnóstico Preditivo")
    st.markdown("Insira os dados do paciente para estimar a categoria de peso baseada em Machine Learning.")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gênero", ["Female", "Male"])
        age = st.number_input("Idade", min_value=14, max_value=80, value=25)
        height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        weight = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.1)
        family_hist = st.selectbox("Histórico Familiar de Sobrepeso?", ["yes", "no"])

    with col2:
        favc = st.selectbox("Consome alimentos calóricos (Fast Food)?", ["yes", "no"])
        fcvc = st.slider("Frequência de vegetais nas refeições (1-3)", 1, 3, 2)
        ncp = st.slider("Refeições principais por dia", 1, 4, 3)
        caec = st.selectbox("Come lanches entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Fumante?", ["yes", "no"])

    with col3:
        ch2o = st.slider("Consumo de água (L/dia)", 1, 3, 2)
        scc = st.selectbox("Monitora calorias diárias?", ["yes", "no"])
        faf = st.slider("Dias de atividade física na semana (0-3)", 0, 3, 1)
        tue = st.slider("Horas de uso de eletrônicos/dia (0-2)", 0, 2, 1)
        calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Meio de transporte principal", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

    if st.button("Realizar Diagnóstico", type="primary"):
        # Preparar dados para o modelo
        input_data = pd.DataFrame([[
            gender, age, height, weight, family_hist, favc, fcvc, ncp,
            caec, smoke, ch2o, scc, faf, tue, calc, mtrans
        ]], columns=['Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'])

        prediction = model.predict(input_data)[0]
        st.success(f"### Categoria Prevista: **{prediction.replace('_', ' ')}**")

# ==========================================
# ABA 2: VISÃO ANALÍTICA (ITEM 3 DO REQUISITO)
# ==========================================
with tab2:
    st.header("Painel Analítico: Principais Fatores da Obesidade")
    st.markdown("Insights derivados de dados para apoio a políticas de saúde e conscientização de pacientes.")

    # Ordem das categorias de peso
    obesity_order = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']

    colA, colB = st.columns(2)

    with colA:
        # Gráfico 1: Genética
        st.subheader("1. Fator Genético")
        fig_gen = px.histogram(df, x="Obesity", color="family_history", barmode="group",
                               category_orders={"Obesity": obesity_order},
                               title="Obesidade x Histórico Familiar",
                               color_discrete_sequence=["#EF553B", "#4C78A8"])
        st.plotly_chart(fig_gen, use_container_width=True)
        st.caption("**Insight:** Pacientes sem histórico familiar raramente ultrapassam o nível de 'Overweight'. A Obesidade Tipo II e III é quase inteiramente composta por pacientes com predisposição genética.")

    with colB:
        # Gráfico 2: Hábitos Alimentares (FAVC)
        st.subheader("2. Hábitos Alimentares (Fast Food)")
        fig_favc = px.histogram(df, x="Obesity", color="FAVC", barmode="group",
                                category_orders={"Obesity": obesity_order},
                                title="Obesidade x Consumo de Alimentos Calóricos",
                                color_discrete_sequence=["#FFA15A", "#19D3F3"])
        st.plotly_chart(fig_favc, use_container_width=True)
        st.caption("**Insight:** O consumo frequente de alimentos altamente calóricos (FAVC) é onipresente nos graus mais severos de obesidade, demonstrando forte correlação ambiental.")

    # Gráfico 3: Sedentarismo x Telas
    st.subheader("3. Estilo de Vida: Telas (TUE) vs Atividade Física (FAF)")
    fig_life = px.density_heatmap(df, x="FAF", y="TUE", facet_col="Obesity",
                                  category_orders={"Obesity": obesity_order},
                                  title="Concentração de Sedentarismo por Categoria de Peso",
                                  color_continuous_scale="Viridis")
    st.plotly_chart(fig_life, use_container_width=True)
    st.caption("**Insight:** Pacientes com 'Normal Weight' e 'Overweight' possuem distribuição de exercícios mais variada. Contudo, nas faixas de 'Obesity Type', há uma brutal concentração na zona de (0 Atividade Física / Alto Tempo de Tela).")