import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Passos Mágicos — Risco de Defasagem",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paleta ────────────────────────────────────────────────────────────────────
CORES = {
    'primaria':   '#155088',
    'destaque':   '#ec3237',
    'verde':      '#10b33b',
    'amarelo':    '#f7d342',
    'secundaria': '#b0c3d4',
    'cinza':      '#d9e2eb',
}

# ── CSS corrigido — força fundo claro e texto escuro em qualquer tema ─────────
st.markdown("""
<style>
    /* Fundo geral */
    .stApp { background-color: #f4f6fb !important; }
    section[data-testid="stSidebar"] { background-color: #eef2f7 !important; }

    /* Força texto escuro em TODO o app */
    .stApp, .stApp p, .stApp span, .stApp div,
    .stApp label, .stApp h1, .stApp h2, .stApp h3 {
        color: #1a1a2e !important;
    }

    /* Métricas — fundo branco, texto azul escuro */
    div[data-testid="metric-container"] {
        background-color: #ffffff !important;
        border: 1px solid #d0dce8 !important;
        border-left: 5px solid #155088 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
    }
    div[data-testid="metric-container"] label {
        color: #155088 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #1a1a2e !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }

    /* Boxes de resultado de risco */
    .risk-alto {
        background-color: #fde8e8 !important;
        border-left: 6px solid #ec3237 !important;
        border-radius: 10px !important;
        padding: 18px 22px !important;
        color: #7b0000 !important;
        font-size: 16px !important;
    }
    .risk-alto strong { color: #c0392b !important; font-size: 18px !important; }

    .risk-mod {
        background-color: #fff8e1 !important;
        border-left: 6px solid #f7d342 !important;
        border-radius: 10px !important;
        padding: 18px 22px !important;
        color: #5a4000 !important;
        font-size: 16px !important;
    }
    .risk-mod strong { color: #b07d00 !important; font-size: 18px !important; }

    .risk-baixo {
        background-color: #e8f5e9 !important;
        border-left: 6px solid #10b33b !important;
        border-radius: 10px !important;
        padding: 18px 22px !important;
        color: #1b5e20 !important;
        font-size: 16px !important;
    }
    .risk-baixo strong { color: #1b5e20 !important; font-size: 18px !important; }

    /* Cards de KPI do painel histórico */
    .kpi-card {
        background-color: #ffffff !important;
        border: 1px solid #d0dce8 !important;
        border-left: 5px solid #155088 !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        margin-bottom: 8px !important;
    }

    /* Avisos e caixas de mensagem */
    div[data-testid="stAlert"],
    div[data-testid="stMessage"],
    div[role="alert"] {
        background-color: #f2f2f2 !important;
        color: #000000 !important;
        border: 1px solid #d0d0d0 !important;
        border-radius: 12px !important;
    }
    div[data-testid="stAlert"] *,
    div[data-testid="stMessage"] *,
    div[role="alert"] * {
        color: #000000 !important;
    }

    /* Botões e controles de seleção */
    .stButton>button,
    div[data-testid="stForm"] button,
    div[data-testid="stFileUploader"] button {
        background-color: #f2f2f2 !important;
        color: #000000 !important;
        border: 1px solid #d0d0d0 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] > div > div,
    div[data-baseweb="select"] button,
    div[data-baseweb="select"] span,
    div[data-testid="stFileUploader"] {
        background-color: #f2f2f2 !important;
        color: #000000 !important;
        border: 1px solid #d0d0d0 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stFileUploader"] * {
        color: #000000 !important;
    }
    div[data-baseweb="select"] div[role="button"] {
        color: #000000 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #000000 !important;
    }
    .kpi-label {
        color: #155088 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        margin-bottom: 4px !important;
    }
    .kpi-value {
        color: #1a1a2e !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* Subtítulos e divisores */
    hr { border-color: #d0dce8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Carregar modelo ───────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    model_path = os.path.join(BASE_DIR, "modelo_risco_defasagem.pkl")
    features_path = os.path.join(BASE_DIR, "features_modelo.pkl")
    if os.path.exists(model_path) and os.path.exists(features_path):
        modelo   = joblib.load(model_path)
        features = joblib.load(features_path)
        return modelo, features
    return None, None

modelo, FEATURES = carregar_modelo()

# ── Carregar base histórica ───────────────────────────────────────────────────
@st.cache_data
def carregar_base():
    parquet_path = os.path.join(BASE_DIR, "base_tratada_final.parquet")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return None

df_hist = carregar_base()

# ── Médias históricas ─────────────────────────────────────────────────────────
MEDIAS_PADRAO = {
    'iaa': 7.2, 'ieg': 7.0, 'ips': 7.1, 'ida': 6.8, 'ipv': 6.5,
    'nota_matematica': 6.5, 'nota_portugues': 6.8, 'nota_ingles': 5.0,
    'media_notas': 6.3, 'media_indicadores': 7.0,
}
if df_hist is not None:
    for col in MEDIAS_PADRAO:
        if col in df_hist.columns:
            MEDIAS_PADRAO[col] = float(df_hist[col].median())

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("## 🎓")
with col_title:
    st.markdown("# Passos Mágicos — Preditor de Risco de Defasagem")
    st.markdown("**POSTECH · FIAP · Datathon 2026**")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Modo de uso")
modo = st.sidebar.radio(
    "Selecione:",
    ["📊 Painel Histórico", "🧑 Avaliação Individual", "📂 Lote (CSV/Excel)", "📘 Dicionário de Campos"],
    index=0
)

# ── Função auxiliar: card KPI ─────────────────────────────────────────────────
def kpi_card(label, value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODO 1 — AVALIAÇÃO INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
if modo == "🧑 Avaliação Individual":
    st.subheader("🧑 Avaliação Individual do Aluno")
    st.info("Preencha os indicadores do aluno e clique em **Calcular Risco**.")

    with st.form("form_individual"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### 📚 Perfil")
            fase          = st.selectbox("Fase", [0,1,2,3,4,5,6,7,8],
                                         format_func=lambda x: "ALFA" if x==0 else f"Fase {x}")
            genero_fem    = st.selectbox("Gênero", ["Feminino","Masculino"])
            anos_programa = st.slider("Anos no Programa", 0, 12, 2)
            instituicao   = st.selectbox("Tipo de Escola",
                                         ["Pública","Privada","Concluiu 3º EM","Universitário Formado","Nenhuma"])
            pedra_ano     = st.selectbox("Pedra Atual", [1,2,3,4,-1],
                                         format_func=lambda x: {1:"Quartzo",2:"Ágata",3:"Ametista",4:"Topázio"}.get(x,"Sem pedra"))
            pedra_2020    = st.selectbox("Pedra 2020", [-1,1,2,3,4],
                                         format_func=lambda x: {1:"Quartzo",2:"Ágata",3:"Ametista",4:"Topázio"}.get(x,"Não informado"))
            pedra_2021    = st.selectbox("Pedra 2021", [-1,1,2,3,4],
                                         format_func=lambda x: {1:"Quartzo",2:"Ágata",3:"Ametista",4:"Topázio"}.get(x,"Não informado"))

        with c2:
            st.markdown("#### 📊 Indicadores Psicossociais")
            iaa = st.slider("IAA — Autoavaliação",   0.0, 10.0, MEDIAS_PADRAO['iaa'],   0.1)
            ieg = st.slider("IEG — Engajamento",     0.0, 10.0, MEDIAS_PADRAO['ieg'],   0.1)
            ips = st.slider("IPS — Psicossocial",    0.0, 10.0, MEDIAS_PADRAO['ips'],   0.1)
            ida = st.slider("IDA — Desempenho",      0.0, 10.0, MEDIAS_PADRAO['ida'],   0.1)
            ipv = st.slider("IPV — Ponto de Virada", 0.0, 10.0, MEDIAS_PADRAO['ipv'],   0.1)

        with c3:
            st.markdown("#### 📝 Notas Escolares")
            nota_mat = st.slider("Nota Matemática", 0.0, 10.0, MEDIAS_PADRAO['nota_matematica'], 0.1)
            nota_por = st.slider("Nota Português",  0.0, 10.0, MEDIAS_PADRAO['nota_portugues'],  0.1)
            nota_ing = st.slider("Nota Inglês",     0.0, 10.0, MEDIAS_PADRAO['nota_ingles'],     0.1)
            media_notas = round((nota_mat + nota_por + nota_ing) / 3, 2)
            media_ind   = round((iaa + ieg + ips) / 3, 2)
            st.metric("📐 Média das Notas",     media_notas)
            st.metric("📐 Média Indicadores",   media_ind)

        calcular = st.form_submit_button("🔍 Calcular Risco", use_container_width=True)

    if calcular:
        INST_MAP = {"Pública":0,"Privada":1,"Concluiu 3º EM":2,"Universitário Formado":3,"Nenhuma":-1}
        entrada = {
            'fase': fase, 'genero_feminino': 1 if genero_fem=="Feminino" else 0,
            'instituicao_cod': INST_MAP.get(instituicao, 0),
            'anos_no_programa': anos_programa,
            'iaa': iaa, 'ieg': ieg, 'ips': ips, 'ida': ida, 'ipv': ipv,
            'nota_matematica': nota_mat, 'nota_portugues': nota_por, 'nota_ingles': nota_ing,
            'media_notas': media_notas, 'media_indicadores': media_ind,
            'pedra_ano': pedra_ano, 'pedra_2020': pedra_2020, 'pedra_2021': pedra_2021,
        }

        if modelo is None:
            score = 0.0
            if ida < 5.0: score += 0.30
            if ieg < 5.0: score += 0.25
            if ips < 5.0: score += 0.20
            if iaa < 5.0: score += 0.15
            if ipv < 5.0: score += 0.10
            prob = min(score, 1.0)
            st.warning("⚠️ Modelo .pkl não encontrado — usando heurística de demonstração.")
        else:
            feat_cols = FEATURES if FEATURES else list(entrada.keys())
            X_pred = pd.DataFrame([entrada])[feat_cols]
            prob = float(modelo.predict_proba(X_pred)[0, 1])

        # ── Resultado ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Resultado da Avaliação")

        if prob >= 0.6:
            nivel        = "🔴 ALTO RISCO"
            css_class    = "risk-alto"
            recomendacao = "Intervenção imediata — acionar equipe pedagógica e psicossocial."
        elif prob >= 0.3:
            nivel        = "🟡 RISCO MODERADO"
            css_class    = "risk-mod"
            recomendacao = "Monitoramento intensivo — reavaliar em 30 dias."
        else:
            nivel        = "🟢 BAIXO RISCO"
            css_class    = "risk-baixo"
            recomendacao = "Manutenção — continuar acompanhamento regular."

        # Cards de KPI com HTML próprio (sem depender do tema do Streamlit)
        r1, r2, r3 = st.columns(3)
        with r1:
            kpi_card("🎯 Probabilidade de Risco", f"{prob*100:.1f}%")
        with r2:
            kpi_card("📊 Nível de Risco", nivel)
        with r3:
            kpi_card("💡 Ação Recomendada", "Ver abaixo ↓")

        # Box de recomendação
        st.markdown(f"""
        <div class="{css_class}">
            <strong>{nivel}</strong><br><br>
            {recomendacao}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gráficos ──────────────────────────────────────────────────────────
        st.markdown("### 📈 Perfil do Aluno vs. Média Histórica")
        categorias = ['IAA','IEG','IPS','IDA','IPV','Mat.','Port.']
        vals_aluno = [iaa, ieg, ips, ida, ipv, nota_mat, nota_por]
        vals_media = [
            MEDIAS_PADRAO['iaa'], MEDIAS_PADRAO['ieg'], MEDIAS_PADRAO['ips'],
            MEDIAS_PADRAO['ida'], MEDIAS_PADRAO['ipv'],
            MEDIAS_PADRAO['nota_matematica'], MEDIAS_PADRAO['nota_portugues']
        ]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.patch.set_facecolor('white')

        x, w = np.arange(len(categorias)), 0.35
        axes[0].set_facecolor('white')
        axes[0].bar(x-w/2, vals_aluno, w, label='Aluno',          color=CORES['primaria'],   alpha=0.85, edgecolor='white')
        axes[0].bar(x+w/2, vals_media,  w, label='Média Histórica',color=CORES['secundaria'], alpha=0.85, edgecolor='white')
        axes[0].set_xticks(x); axes[0].set_xticklabels(categorias, color='#1a1a2e')
        axes[0].set_ylim(0, 11); axes[0].set_ylabel('Pontuação', color='#1a1a2e')
        axes[0].set_title('Aluno vs. Média Histórica', fontweight='bold', color='#1a1a2e')
        axes[0].tick_params(colors='#1a1a2e')
        axes[0].axhline(6.5, color=CORES['destaque'], ls='--', lw=1.2, alpha=0.7)
        axes[0].legend(facecolor='white', edgecolor='#d0dce8', labelcolor='#1a1a2e')
        axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
        for spine in ['left','bottom']:
            axes[0].spines[spine].set_color('#aaaaaa')

        # Gauge
        from matplotlib.patches import Wedge, Circle

        axes[1].set_facecolor('white')
        for start_deg, end_deg, cor in [
            (0,   60,  CORES['verde']),
            (60, 120, CORES['amarelo']),
            (120,180, CORES['destaque'])
        ]:
            wedge = Wedge((0, 0), 1.0, start_deg, end_deg,
                          width=0.35, facecolor=cor, edgecolor='white', linewidth=1.5, alpha=0.9)
            axes[1].add_patch(wedge)

        axes[1].add_patch(Circle((0, 0), 0.35, facecolor='white', edgecolor='none'))

        angle = np.pi * (1 - prob)
        axes[1].annotate('', xy=(0.78*np.cos(angle), 0.78*np.sin(angle)), xytext=(0, 0),
                          arrowprops=dict(arrowstyle='->', color='#1a1a2e', lw=2.5))
        axes[1].set_xlim(-1.1, 1.1)
        axes[1].set_ylim(-0.2, 1.1)
        axes[1].set_aspect('equal', adjustable='box')
        axes[1].axis('off')
        axes[1].text(0, -0.08, f'{prob*100:.1f}%', ha='center', va='center',
                     fontsize=22, fontweight='bold', color=CORES['primaria'])
        axes[1].set_title('Probabilidade de Risco', fontweight='bold', color='#1a1a2e', pad=10)
        axes[1].text(-1.05, 0.02, 'Baixo',    fontsize=10, color=CORES['verde'],   fontweight='bold')
        axes[1].text(-0.18, 1.05, 'Moderado', fontsize=10, color='#b07d00',        fontweight='bold', ha='center')
        axes[1].text(0.82, 0.02, 'Alto',     fontsize=10, color=CORES['destaque'], fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MODO 2 — LOTE
# ══════════════════════════════════════════════════════════════════════════════
elif modo == "📂 Lote (CSV/Excel)":
    st.subheader("📂 Predição em Lote")
    st.info("Faça upload de um arquivo CSV ou Excel com colunas: `fase`, `iaa`, `ieg`, `ips`, `ida`, `ipv`, `nota_matematica`, `nota_portugues`, `nota_ingles`.")

    arquivo = st.file_uploader("📁 Selecione o arquivo", type=["csv","xlsx","xls"])

    if arquivo:
        try:
            df_up = pd.read_csv(arquivo) if arquivo.name.endswith('.csv') else pd.read_excel(arquivo)
            st.success(f"✅ Arquivo carregado: **{df_up.shape[0]} alunos** | {df_up.shape[1]} colunas")
            st.dataframe(df_up.head(5), use_container_width=True)

            if st.button("🔍 Calcular Risco para Todos os Alunos", use_container_width=True):
                df_result = df_up.copy()
                defaults = {
                    'fase':0,'genero_feminino':0,'instituicao_cod':0,'anos_no_programa':2,
                    'iaa':7.2,'ieg':7.0,'ips':7.1,'ida':6.8,'ipv':6.5,
                    'nota_matematica':6.5,'nota_portugues':6.8,'nota_ingles':5.0,
                    'media_notas':6.3,'media_indicadores':7.0,'pedra_ano':2,'pedra_2020':-1,'pedra_2021':-1
                }
                if 'media_notas' not in df_result.columns:
                    notas = [c for c in ['nota_matematica','nota_portugues','nota_ingles'] if c in df_result.columns]
                    df_result['media_notas'] = df_result[notas].mean(axis=1) if notas else 6.3
                if 'media_indicadores' not in df_result.columns:
                    inds = [c for c in ['iaa','ieg','ips'] if c in df_result.columns]
                    df_result['media_indicadores'] = df_result[inds].mean(axis=1) if inds else 7.0
                if 'genero_feminino' not in df_result.columns and 'genero' in df_result.columns:
                    df_result['genero_feminino'] = (df_result['genero'] == 'Feminino').astype(int)
                for col, val in defaults.items():
                    if col not in df_result.columns:
                        df_result[col] = val

                if modelo is None:
                    df_result['prob_risco'] = (
                        (df_result.get('ida',7) < 5.0).astype(float)*0.35 +
                        (df_result.get('ieg',7) < 5.0).astype(float)*0.30 +
                        (df_result.get('ips',7) < 5.0).astype(float)*0.20 +
                        (df_result.get('ipv',7) < 5.0).astype(float)*0.15
                    ).clip(0, 1)
                    st.warning("⚠️ Modelo .pkl não encontrado — usando heurística.")
                else:
                    feat_cols = FEATURES if FEATURES else list(defaults.keys())
                    X_b = df_result[feat_cols].fillna(df_result[feat_cols].median())
                    df_result['prob_risco'] = modelo.predict_proba(X_b)[:, 1]

                df_result['nivel_risco'] = pd.cut(
                    df_result['prob_risco'], bins=[0,0.3,0.6,1.0],
                    labels=['Baixo Risco','Risco Moderado','Alto Risco']
                )

                st.markdown("### 📊 Resumo dos Resultados")
                m1, m2, m3, m4 = st.columns(4)
                with m1: kpi_card("Total de Alunos", f"{len(df_result):,}")
                with m2: kpi_card("🟢 Baixo Risco",    f"{(df_result['nivel_risco']=='Baixo Risco').sum():,}")
                with m3: kpi_card("🟡 Risco Moderado", f"{(df_result['nivel_risco']=='Risco Moderado').sum():,}")
                with m4: kpi_card("🔴 Alto Risco",      f"{(df_result['nivel_risco']=='Alto Risco').sum():,}")

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.patch.set_facecolor('white')
                for ax in axes: ax.set_facecolor('white')

                dist = df_result['nivel_risco'].value_counts()
                axes[0].bar(dist.index, dist.values,
                            color=[CORES['verde'],CORES['amarelo'],CORES['destaque']], edgecolor='white')
                axes[0].set_title('Distribuição de Risco', fontweight='bold', color='#1a1a2e')
                axes[0].tick_params(colors='#1a1a2e')
                axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

                axes[1].hist(df_result['prob_risco'], bins=30,
                             color=CORES['primaria'], alpha=0.8, edgecolor='white')
                axes[1].axvline(0.3, color=CORES['amarelo'], ls='--', lw=1.5, label='0.3')
                axes[1].axvline(0.6, color=CORES['destaque'], ls='--', lw=1.5, label='0.6')
                axes[1].set_title('Distribuição das Probabilidades', fontweight='bold', color='#1a1a2e')
                axes[1].set_xlabel('P(Em Risco)', color='#1a1a2e')
                axes[1].tick_params(colors='#1a1a2e')
                axes[1].legend(facecolor='white', labelcolor='#1a1a2e')
                axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close()

                st.markdown("### 🔴 Alunos em Alto Risco (top 20)")
                cols_show = ['prob_risco','nivel_risco'] + \
                            [c for c in ['fase','iaa','ieg','ips','ida','ipv'] if c in df_result.columns]
                alto = df_result[df_result['nivel_risco']=='Alto Risco'][cols_show]\
                         .sort_values('prob_risco', ascending=False).head(20).copy()
                alto['prob_risco'] = alto['prob_risco'].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(alto, use_container_width=True)

                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Baixar resultado completo (CSV)", data=csv,
                                   file_name="resultado_risco_defasagem.csv",
                                   mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODO 3 — DICIONÁRIO DE CAMPOS
# ══════════════════════════════════════════════════════════════════════════════
elif modo == "📘 Dicionário de Campos":
    st.subheader("📘 Dicionário de Campos")
    st.markdown(
        "Este dicionário descreve todos os campos usados pelo app em avaliações individuais, cargas em lote e no painel histórico."
    )
    st.markdown("---")
    campos = [
        ("fase", "Fase atual do aluno no programa. 0 = ALFA, 1–8 = fases do curso."),
        ("genero", "Gênero do aluno — usado para análises e filtros de painel histórico."),
        ("genero_feminino", "Indicador 1/0 para gênero feminino, usado pelo modelo em lote e individual."),
        ("anos_no_programa", "Quantidade de anos que o aluno está no programa."),
        ("instituicao", "Tipo de escola / formação do aluno."),
        ("instituicao_cod", "Código numérico do tipo de instituição usado pelo modelo."),
        ("pedra_ano", "Classifica o aluno com base no INDE: Quartzo = 2,405 a 5,506; Ágata = 5,506 a 6,868; Ametista = 6,868 a 8,230; Topázio = 8,230 a 9,294. Cada pedra possui sua descrição."),
        ("pedra_2020", "Classificação de pedra em 2020 com o mesmo critério de INDE: Quartzo, Ágata, Ametista ou Topázio. -1 = não informado."),
        ("pedra_2021", "Classificação de pedra em 2021 com o mesmo critério de INDE: Quartzo, Ágata, Ametista ou Topázio. -1 = não informado."),
        ("iaa", "Autoavaliação psicossocial do aluno (0 a 10)."),
        ("ieg", "Engajamento geral do aluno (0 a 10)."),
        ("ips", "Índice psicossocial (0 a 10)."),
        ("ida", "Desempenho acadêmico e comportamento (0 a 10)."),
        ("ipv", "Ponto de virada psicológico do aluno (0 a 10)."),
        ("nota_matematica", "Nota de matemática do aluno (0 a 10)."),
        ("nota_portugues", "Nota de português do aluno (0 a 10)."),
        ("nota_ingles", "Nota de inglês do aluno (0 a 10)."),
        ("media_notas", "Média das notas de matemática, português e inglês."),
        ("media_indicadores", "Média dos indicadores psicossociais IAA, IEG e IPS."),
        ("prob_risco", "Probabilidade estimada de risco de defasagem gerada pelo modelo ou heurística."),
        ("nivel_risco", "Classificação final do risco: Baixo Risco, Risco Moderado ou Alto Risco."),
        ("inde_ano", "Índice de desempenho educacional anual, presente na base histórica."),
        ("ian", "Indicador de adequação ao nível do aluno, presente na base histórica."),
    ]
    df_campos = pd.DataFrame(campos, columns=["Campo", "Descrição"])
    st.table(df_campos)


# ══════════════════════════════════════════════════════════════════════════════
#  MODO 3 — PAINEL HISTÓRICO
# ══════════════════════════════════════════════════════════════════════════════
elif modo == "📊 Painel Histórico":
    st.subheader("📊 Painel Histórico — Base 2022–2024")

    if df_hist is None:
        st.warning("⚠️ Arquivo `base_tratada_final.parquet` não encontrado. Execute o Script_Passos_Magicos.py primeiro.")
        st.stop()

    with st.sidebar:
        st.markdown("### 🔎 Filtros")
        anos = sorted(df_hist['ano_referencia'].dropna().unique().tolist())
        anos_sel = st.multiselect("Ano", anos, default=anos)
        if 'fase' in df_hist.columns:
            fases = sorted(df_hist['fase'].dropna().unique().tolist())
            fases_sel = st.multiselect("Fase", fases, default=fases)
        else:
            fases_sel = []

    df_f = df_hist[df_hist['ano_referencia'].isin(anos_sel)]
    if fases_sel and 'fase' in df_f.columns:
        df_f = df_f[df_f['fase'].isin(fases_sel)]

    # KPIs com cards HTML
    st.markdown("### 📌 Indicadores Gerais")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi_card("Total Alunos", f"{len(df_f):,}")
    with k2: kpi_card("INDE Médio",   f"{df_f['inde_ano'].mean():.2f}"  if 'inde_ano' in df_f.columns else "—")
    with k3: kpi_card("IDA Médio",    f"{df_f['ida'].mean():.2f}"       if 'ida'      in df_f.columns else "—")
    with k4: kpi_card("IEG Médio",    f"{df_f['ieg'].mean():.2f}"       if 'ieg'      in df_f.columns else "—")
    pct_def = (df_f['ian'] <= 5.0).mean()*100 if 'ian' in df_f.columns else 0
    with k5: kpi_card("% Defasados",  f"{pct_def:.1f}%")

    st.markdown("---")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor('white')
    for ax in axes.flatten(): ax.set_facecolor('white')

    def fmt_ax(ax):
        ax.tick_params(colors='#1a1a2e')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for spine in ['left','bottom']: ax.spines[spine].set_color('#aaaaaa')

    # 1 — INDE por ano
    if 'inde_ano' in df_f.columns:
        inde_ano = df_f.groupby('ano_referencia')['inde_ano'].mean()
        axes[0,0].plot(inde_ano.index, inde_ano.values, marker='o',
                       color=CORES['primaria'], lw=2.5, ms=9)
        axes[0,0].fill_between(inde_ano.index, inde_ano.values, alpha=0.1, color=CORES['primaria'])
        for a, v in inde_ano.items():
            axes[0,0].text(a, v+0.05, f'{v:.2f}', ha='center', fontweight='bold', color='#1a1a2e')
        axes[0,0].set_title('INDE Médio por Ano', fontweight='bold', color='#1a1a2e')
        axes[0,0].set_ylim(0, 10); fmt_ax(axes[0,0])

    # 2 — IAN distribuição
    if 'ian' in df_f.columns:
        def classif(v):
            return 'Adequado' if v==10.0 else ('Moderado' if v==5.0 else 'Severo')
        df_f2 = df_f.copy()
        df_f2['ian_c'] = df_f2['ian'].apply(classif)
        cnt = df_f2['ian_c'].value_counts().reindex(['Adequado','Moderado','Severo'])
        wedges, texts, autotexts = axes[0,1].pie(
            cnt, labels=cnt.index, autopct='%1.1f%%',
            colors=[CORES['verde'],CORES['amarelo'],CORES['destaque']],
            startangle=90, wedgeprops=dict(width=0.55, edgecolor='white')
        )
        for t in texts:     t.set_color('#1a1a2e')
        for at in autotexts: at.set_color('#1a1a2e'); at.set_fontweight('bold')
        axes[0,1].set_title('Adequação do Nível (IAN)', fontweight='bold', color='#1a1a2e')

    # 3 — INDE por pedra
    if 'pedra_ano' in df_f.columns:
        PEDRA_L = {1:'Quartzo',2:'Ágata',3:'Ametista',4:'Topázio'}
        dp = df_f[df_f['pedra_ano'].isin([1,2,3,4])].copy()
        dp['pedra_n'] = dp['pedra_ano'].map(PEDRA_L)
        medias_p = dp.groupby('pedra_n')['inde_ano'].mean().reindex(['Quartzo','Ágata','Ametista','Topázio'])
        bars = axes[0,2].bar(medias_p.index, medias_p.values,
                             color=[CORES['cinza'],CORES['amarelo'],CORES['secundaria'],CORES['primaria']],
                             edgecolor='white')
        for bar, v in zip(bars, medias_p.values):
            axes[0,2].text(bar.get_x()+bar.get_width()/2, v+0.05, f'{v:.2f}',
                           ha='center', fontweight='bold', fontsize=9, color='#1a1a2e')
        axes[0,2].set_title('INDE Médio por Pedra', fontweight='bold', color='#1a1a2e')
        axes[0,2].set_ylim(0, 10.5); fmt_ax(axes[0,2])

    # 4 — IEG × IDA
    if 'ieg' in df_f.columns and 'ida' in df_f.columns:
        samp = df_f[['ieg','ida']].dropna().sample(min(500, len(df_f)), random_state=1)
        axes[1,0].scatter(samp['ieg'], samp['ida'], alpha=0.2, color=CORES['secundaria'], s=10)
        m, b = np.polyfit(samp['ieg'], samp['ida'], 1)
        xl = np.linspace(samp['ieg'].min(), samp['ieg'].max(), 100)
        axes[1,0].plot(xl, m*xl+b, color=CORES['destaque'], lw=2)
        r = samp.corr().iloc[0,1]
        axes[1,0].set_title(f'IEG × IDA  (r={r:.2f})', fontweight='bold', color='#1a1a2e')
        fmt_ax(axes[1,0])

    # 5 — Gênero × INDE
    if 'genero' in df_f.columns and 'inde_ano' in df_f.columns:
        gi = df_f.groupby(['ano_referencia','genero'])['inde_ano'].mean().unstack()
        gi.plot(kind='bar', ax=axes[1,1],
                color=[CORES['destaque'],CORES['secundaria']], edgecolor='white', width=0.55)
        axes[1,1].set_title('INDE por Gênero e Ano', fontweight='bold', color='#1a1a2e')
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=0, color='#1a1a2e')
        axes[1,1].legend(title='Gênero', facecolor='white', labelcolor='#1a1a2e')
        fmt_ax(axes[1,1])

    # 6 — Tempo no programa × INDE
    if 'anos_no_programa' in df_f.columns and 'inde_ano' in df_f.columns:
        t_i = df_f[df_f['anos_no_programa'].between(0,12)].groupby('anos_no_programa')['inde_ano'].mean()
        axes[1,2].plot(t_i.index, t_i.values, marker='o', color=CORES['verde'], lw=2.5, ms=7)
        axes[1,2].fill_between(t_i.index, t_i.values, alpha=0.12, color=CORES['verde'])
        axes[1,2].set_title('INDE × Anos no Programa', fontweight='bold', color='#1a1a2e')
        axes[1,2].set_xlabel('Anos no Programa', color='#1a1a2e')
        fmt_ax(axes[1,2])

    plt.suptitle(f'Painel Histórico — {len(df_f):,} alunos',
                 fontsize=13, fontweight='bold', color='#1a1a2e', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#555555;'><small>Passos Mágicos · POSTECH FIAP · Datathon 2026 · "
    "Desenvolvido com ❤️ para apoiar a educação</small></center>",
    unsafe_allow_html=True
)