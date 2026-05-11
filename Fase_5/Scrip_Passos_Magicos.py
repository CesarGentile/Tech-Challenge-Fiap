# =============================================================================
#  PASSOS MÁGICOS — DATATHON POSTECH / FIAP
#  Script Python puro — rode com: python Script_Passos_Magicos.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')          # salva figuras em disco sem precisar de janela
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO
# -----------------------------------------------------------------------------
INPUT_FILE   = "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"   # Excel original
PARQUET_FILE = "base_tratada_final.parquet"                 # base já tratada
OUTPUT_DIR   = "output_figuras"                             # pasta das figuras

os.makedirs(OUTPUT_DIR, exist_ok=True)

CORES = {
    'primaria':   '#155088',
    'secundaria': '#b0c3d4',
    'destaque':   '#ec3237',
    'verde':      '#10b33b',
    'amarelo':    '#f7d342',
    'cinza':      '#d9e2eb',
}
ESTILO = [CORES['primaria'], CORES['secundaria'], CORES['amarelo'],
          CORES['verde'], CORES['destaque'], CORES['cinza']]

plt.rcParams.update({
    'figure.dpi': 130,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def salvar(nome):
    path = os.path.join(OUTPUT_DIR, nome)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"   Figura salva: {path}")

print("=" * 60)
print("  PASSOS MÁGICOS — DATATHON POSTECH / FIAP")
print("=" * 60)


# =============================================================================
# SEÇÃO 1 — FUNÇÕES AUXILIARES DE TRATAMENTO
# =============================================================================
def drop_fully_null_cols(df, label):
    cols = [c for c in df.columns if df[c].isna().all()]
    if cols:
        print(f"  [{label}] Removidas colunas 100% nulas: {cols}")
    return df.drop(columns=cols)


def impute_by_group(df, col, group_col='fase', strategy='median'):
    if col not in df.columns:
        return df
    if strategy == 'median':
        df[col] = df.groupby(group_col)[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())
    elif strategy == 'mode':
        df[col] = df.groupby(group_col)[col].transform(
            lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x)
        mode_g = df[col].mode()
        if not mode_g.empty:
            df[col] = df[col].fillna(mode_g.iloc[0])
    return df


def extrair_fase(valor):
    if pd.isna(valor):
        return np.nan
    s = str(valor).strip().upper()
    if s in ('ALFA', '0'):
        return 0
    if s.startswith('FASE'):
        try:
            return int(s.split()[1])
        except:
            return np.nan
    digits = ''.join(c for c in s if c.isdigit())
    try:
        return int(digits) if digits else np.nan
    except:
        return np.nan


def normalizar_instituicao(valor):
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip().lower()
    if 'concluiu' in v:             return 'Concluiu 3o EM'
    if any(k in v for k in ['publica','escola publica','rede decisao']): return 'Publica'
    if any(k in v for k in ['privada','empresa parceira','bolsa 100']): return 'Privada'
    if any(k in v for k in ['formado','bolsista universitario']): return 'Universitario Formado'
    if 'nenhuma' in v:              return 'Nenhuma'
    return str(valor).strip()


def calcular_ipp_reverso(row):
    fase = row.get('fase')
    inde = row.get('inde_ano')
    campos = [row.get(c) for c in ['ian','ida','ieg','iaa','ips','ipv']]
    if pd.isna(fase) or fase > 7:
        return np.nan
    if any(pd.isna(v) for v in [inde] + campos):
        return np.nan
    ian, ida, ieg, iaa, ips, ipv = campos
    soma = ian*0.1 + ida*0.2 + ieg*0.2 + iaa*0.1 + ips*0.1 + ipv*0.2
    return float(np.clip(round((inde - soma) / 0.1, 1), 0.0, 10.0))


def add_features(df, ano):
    df['anos_no_programa'] = (ano - df['ano_ingresso']).clip(lower=0)
    notas = [c for c in ['nota_matematica','nota_portugues','nota_ingles'] if c in df.columns]
    df['media_notas'] = df[notas].replace(0, np.nan).mean(axis=1)
    indicadores = [c for c in ['iaa','ieg','ips','ipp'] if c in df.columns]
    df['media_indicadores'] = df[indicadores].mean(axis=1)
    if ano == 2023 and 'pedra_2022' in df.columns and 'pedra_ano' in df.columns:
        df['evolucao_pedra'] = df['pedra_ano'].replace(-1,np.nan) - df['pedra_2022'].replace(-1,np.nan)
    elif ano == 2024 and 'pedra_2023' in df.columns and 'pedra_ano' in df.columns:
        df['evolucao_pedra'] = df['pedra_ano'].replace(-1,np.nan) - df['pedra_2023'].replace(-1,np.nan)
    else:
        df['evolucao_pedra'] = np.nan
    df['genero_feminino'] = (df['genero'] == 'Feminino').astype(int)
    INST_MAP = {'Publica':0,'Privada':1,'Concluiu 3o EM':2,'Universitario Formado':3,'Nenhuma':-1}
    df['instituicao_cod'] = df['instituicao'].map(INST_MAP).fillna(0).astype(int)
    return df


def safe_select(df, cols):
    return df[[c for c in cols if c in df.columns]].copy()


def classificar_ian(v):
    if v == 10.0: return 'Adequado'
    if v == 5.0:  return 'Moderado'
    return 'Severo'


# =============================================================================
# SEÇÃO 2 — CARREGAMENTO E TRATAMENTO DOS DADOS
# =============================================================================
print("\n[1/14] Carregando dados...")

RENAME_2022 = {
    'RA':'ra','Fase':'fase','Turma':'turma','Nome':'nome','Ano nasc':'ano_nascimento',
    'Idade 22':'idade','G\u00eanero':'genero','Ano ingresso':'ano_ingresso',
    'Institui\u00e7\u00e3o de ensino':'instituicao','Pedra 20':'pedra_2020',
    'Pedra 21':'pedra_2021','Pedra 22':'pedra_ano','INDE 22':'inde_ano',
    'Cg':'cg','Cf':'cf','Ct':'ct','N\u00ba Av':'num_avaliadores',
    'Avaliador1':'avaliador_1','Rec Av1':'rec_avaliador_1',
    'Avaliador2':'avaliador_2','Avaliador3':'avaliador_3','Avaliador4':'avaliador_4',
    'IAA':'iaa','IEG':'ieg','IPS':'ips','Rec Psicologia':'rec_psicologia',
    'IDA':'ida','Matem':'nota_matematica','Portug':'nota_portugues',
    'Ingl\u00eas':'nota_ingles','Indicado':'indicado','Atingiu PV':'atingiu_pv',
    'IPV':'ipv','IAN':'ian','Fase ideal':'fase_ideal','Defas':'defasagem',
    'Destaque IEG':'destaque_ieg','Destaque IDA':'destaque_ida','Destaque IPV':'destaque_ipv'
}
RENAME_2023 = {
    'RA':'ra','Fase':'fase','INDE 2023':'inde_ano','Pedra 2023':'pedra_ano',
    'Turma':'turma','Nome Anonimizado':'nome','Data de Nasc':'data_nascimento',
    'Idade':'idade','G\u00eanero':'genero','Ano ingresso':'ano_ingresso',
    'Institui\u00e7\u00e3o de ensino':'instituicao','Pedra 20':'pedra_2020',
    'Pedra 21':'pedra_2021','Pedra 22':'pedra_2022','INDE 22':'inde_2022',
    'INDE 23':'inde_2023','N\u00ba Av':'num_avaliadores',
    'Avaliador1':'avaliador_1','Avaliador2':'avaliador_2',
    'Avaliador3':'avaliador_3','Avaliador4':'avaliador_4',
    'IAA':'iaa','IEG':'ieg','IPS':'ips','IPP':'ipp','Rec Psicologia':'rec_psicologia',
    'IDA':'ida','Mat':'nota_matematica','Por':'nota_portugues','Ing':'nota_ingles',
    'Indicado':'indicado','Atingiu PV':'atingiu_pv','IPV':'ipv','IAN':'ian',
    'Fase Ideal':'fase_ideal','Defasagem':'defasagem',
    'Destaque IEG':'destaque_ieg','Destaque IDA':'destaque_ida','Destaque IPV':'destaque_ipv'
}
RENAME_2024 = {
    'RA':'ra','Fase':'fase','INDE 2024':'inde_ano','Pedra 2024':'pedra_ano',
    'Turma':'turma','Nome Anonimizado':'nome','Data de Nasc':'data_nascimento',
    'Idade':'idade','G\u00eanero':'genero','Ano ingresso':'ano_ingresso',
    'Institui\u00e7\u00e3o de ensino':'instituicao','Pedra 20':'pedra_2020',
    'Pedra 21':'pedra_2021','Pedra 22':'pedra_2022','Pedra 23':'pedra_2023',
    'INDE 22':'inde_2022','INDE 23':'inde_2023','N\u00ba Av':'num_avaliadores',
    'Avaliador1':'avaliador_1','Avaliador2':'avaliador_2',
    'Avaliador3':'avaliador_3','Avaliador4':'avaliador_4',
    'Avaliador5':'avaliador_5','Avaliador6':'avaliador_6',
    'IAA':'iaa','IEG':'ieg','IPS':'ips','IPP':'ipp','Rec Psicologia':'rec_psicologia',
    'IDA':'ida','Mat':'nota_matematica','Por':'nota_portugues','Ing':'nota_ingles',
    'Indicado':'indicado','Atingiu PV':'atingiu_pv','IPV':'ipv','IAN':'ian',
    'Fase Ideal':'fase_ideal','Defasagem':'defasagem',
    'Destaque IEG':'destaque_ieg','Destaque IDA':'destaque_ida','Destaque IPV':'destaque_ipv',
    'Escola':'escola','Ativo/ Inativo':'status_ativo','Ativo/ Inativo.1':'status_ativo_2'
}

USE_PARQUET = False
if os.path.exists(PARQUET_FILE):
    print(f"  Arquivo parquet encontrado. Carregando {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    USE_PARQUET = True
    print(f"  Base carregada: {df.shape[0]:,} linhas | {df.shape[1]} colunas")
elif os.path.exists(INPUT_FILE):
    print(f"  Carregando Excel: {INPUT_FILE}")
    sheets    = pd.read_excel(INPUT_FILE, sheet_name=None)
    df22_raw  = sheets['PEDE2022'].copy()
    df23_raw  = sheets['PEDE2023'].copy()
    df24_raw  = sheets['PEDE2024'].copy()
    print(f"  PEDE2022: {df22_raw.shape[0]:,} | PEDE2023: {df23_raw.shape[0]:,} | PEDE2024: {df24_raw.shape[0]:,}")
else:
    raise FileNotFoundError(
        f"\nERRO: Nenhum arquivo encontrado!\n"
        f"Coloque '{INPUT_FILE}' ou '{PARQUET_FILE}' na mesma pasta deste script.\n"
        f"Pasta atual: {os.getcwd()}"
    )

if not USE_PARQUET:
    print("\n[2/14] Limpando e padronizando dados...")
    df22 = drop_fully_null_cols(df22_raw.copy(), '2022')
    df23 = drop_fully_null_cols(df23_raw.copy(), '2023')
    df24 = drop_fully_null_cols(df24_raw.copy(), '2024')

    df22 = df22.rename(columns=RENAME_2022)
    df23 = df23.rename(columns=RENAME_2023)
    df24 = df24.rename(columns=RENAME_2024)

    df22['ano_referencia'] = 2022
    df23['ano_referencia'] = 2023
    df24['ano_referencia'] = 2024

    df24['inde_ano'] = pd.to_numeric(df24['inde_ano'].replace('INCLUIR', np.nan), errors='coerce')
    df22['fase'] = pd.to_numeric(df22['fase'], errors='coerce')
    df23['fase'] = df23['fase'].apply(extrair_fase)
    df24['fase'] = df24['fase'].apply(extrair_fase)

    df22['data_nascimento'] = pd.to_datetime(df22['ano_nascimento'].astype(str)+'-01-01', errors='coerce')
    df22.drop(columns=['ano_nascimento'], errors='ignore', inplace=True)
    df23['data_nascimento'] = pd.to_datetime(df23['data_nascimento'], errors='coerce')
    df24['data_nascimento'] = pd.to_datetime(df24['data_nascimento'], errors='coerce')

    NUMERIC_COLS = ['inde_ano','inde_2022','inde_2023','iaa','ieg','ips','ipp','ida',
                    'ipv','ian','nota_matematica','nota_portugues','nota_ingles',
                    'num_avaliadores','defasagem','idade','ano_ingresso']
    for df_l in [df22, df23, df24]:
        for col in NUMERIC_COLS:
            if col in df_l.columns:
                df_l[col] = pd.to_numeric(df_l[col], errors='coerce')

    print("  Inferindo IPP 2022 por engenharia reversa...")
    if 'ipp' not in df22.columns:
        df22['ipp'] = np.nan
    mask = df22['ipp'].isna()
    df22.loc[mask, 'ipp'] = df22[mask].apply(calcular_ipp_reverso, axis=1)

    GENERO_MAP  = {'menina':'Feminino','menino':'Masculino','feminino':'Feminino','masculino':'Masculino'}
    PEDRA_ORDER = {'Quartzo':1,'Agata':2,'Ametista':3,'Topazio':4,
                   '\u00c1gata':2,'Top\u00e1zio':4}
    PEDRA_COLS  = ['pedra_ano','pedra_2020','pedra_2021','pedra_2022','pedra_2023']

    for df_l in [df22, df23, df24]:
        df_l['genero']     = df_l['genero'].str.strip().str.lower().map(GENERO_MAP)
        df_l['instituicao']= df_l['instituicao'].apply(normalizar_instituicao)
        for col in PEDRA_COLS:
            if col in df_l.columns:
                df_l[col] = df_l[col].map(PEDRA_ORDER)

    INDICATOR_COLS = ['iaa','ieg','ips','ipp','ida','ipv','ian']
    NOTE_COLS      = ['nota_matematica','nota_portugues']
    for idx, df_l in enumerate([df22, df23, df24]):
        for col in INDICATOR_COLS + NOTE_COLS:
            if col in df_l.columns and df_l[col].isna().any():
                df_l = impute_by_group(df_l, col)
        if 'nota_ingles' in df_l.columns:
            df_l['nota_ingles'] = df_l['nota_ingles'].fillna(0)
        for col in PEDRA_COLS:
            if col in df_l.columns:
                df_l[col] = df_l[col].fillna(-1)
        if 'defasagem' in df_l.columns:
            df_l['defasagem'] = df_l['defasagem'].fillna(0)
        for col in ['inde_2022','inde_2023']:
            if col in df_l.columns:
                df_l[col] = df_l[col].fillna(-1)
        if 'inde_ano' in df_l.columns and df_l['inde_ano'].isna().any():
            df_l = impute_by_group(df_l, 'inde_ano')
        if idx == 0:   df22 = df_l
        elif idx == 1: df23 = df_l
        else:          df24 = df_l

    df22 = add_features(df22, 2022)
    df23 = add_features(df23, 2023)
    df24 = add_features(df24, 2024)

    CORE_COLS = [
        'ra','ano_referencia','fase','genero','genero_feminino',
        'idade','data_nascimento','ano_ingresso','anos_no_programa',
        'instituicao','instituicao_cod','pedra_ano','pedra_2020','pedra_2021','pedra_2022',
        'inde_ano','inde_2022','num_avaliadores',
        'iaa','ieg','ips','ipp','ida','ipv','ian',
        'nota_matematica','nota_portugues','nota_ingles',
        'media_notas','media_indicadores','evolucao_pedra','defasagem'
    ]
    df = pd.concat([
        safe_select(df22, CORE_COLS),
        safe_select(df23, CORE_COLS + ['pedra_2023','inde_2023']),
        safe_select(df24, CORE_COLS + ['pedra_2023','inde_2023']),
    ], ignore_index=True)

    df.to_parquet(PARQUET_FILE, index=False)
    print(f"  Base consolidada: {df.shape[0]:,} linhas | {df.shape[1]} colunas")
    print(f"  Salvo: {PARQUET_FILE}")

df['ian_classe'] = df['ian'].apply(classificar_ian)
print(f"\n  Distribuicao por ano:\n{df.groupby('ano_referencia').size().rename('alunos').to_string()}")


# =============================================================================
# SEÇÃO 3 — PERGUNTAS DE NEGÓCIO (P1 a P11)
# =============================================================================

# ── P1: IAN ──────────────────────────────────────────────────────────────────
print("\n[3/14] P1 — Adequacao do Nivel (IAN)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('P1 — Adequacao do Nivel (IAN)', fontsize=14, fontweight='bold')

contagem = df['ian_classe'].value_counts().reindex(['Adequado','Moderado','Severo'])
wedges, texts, autotexts = axes[0].pie(
    contagem, labels=contagem.index, autopct='%1.1f%%',
    colors=[CORES['verde'],CORES['amarelo'],CORES['destaque']],
    startangle=90, wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2))
for at in autotexts: at.set_fontweight('bold')
axes[0].set_title('Distribuicao Geral IAN (2022-2024)', fontweight='bold')

evolucao  = df.groupby(['ano_referencia','ian_classe']).size().reset_index(name='n')
pivot     = evolucao.pivot(index='ano_referencia', columns='ian_classe', values='n').fillna(0)
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
pivot_pct = pivot_pct.reindex(columns=['Adequado','Moderado','Severo'], fill_value=0)
pivot_pct.plot(kind='bar', stacked=True, ax=axes[1],
    color=[CORES['verde'],CORES['amarelo'],CORES['destaque']],
    edgecolor='white', width=0.55)
axes[1].set_title('Evolucao IAN por Ano (%)', fontweight='bold')
axes[1].set_xlabel('Ano'); axes[1].set_ylabel('% Alunos')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(title='Nivel IAN', loc='upper right')
plt.tight_layout(); salvar('fig01_ian.png')

# ── P2: IDA ──────────────────────────────────────────────────────────────────
print("[4/14] P2 — Desempenho Academico (IDA)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('P2 — Desempenho Academico (IDA)', fontsize=14, fontweight='bold')
ida_ano = df.groupby('ano_referencia')['ida'].mean()
bars = axes[0].bar(ida_ano.index.astype(str), ida_ano.values,
    color=ESTILO[:3], edgecolor='white', width=0.5)
for bar, val in zip(bars, ida_ano.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, val+0.05, f'{val:.2f}',
        ha='center', fontweight='bold', fontsize=11)
axes[0].set_title('IDA Medio por Ano', fontweight='bold'); axes[0].set_ylim(0, 10)
fases_v = df[df['fase'].between(0, 8)].copy()
fases_v['fase_label'] = fases_v['fase'].apply(lambda x: 'ALFA' if x==0 else f'Fase {int(x)}')
order = ['ALFA'] + [f'Fase {i}' for i in range(1, 9)]
order = [o for o in order if o in fases_v['fase_label'].unique()]
sns.boxplot(data=fases_v, x='fase_label', y='ida', order=order,
    palette='Blues_r', ax=axes[1], width=0.55, linewidth=1.2,
    flierprops=dict(marker='o', markersize=3, alpha=0.3))
axes[1].set_title('IDA por Fase', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout(); salvar('fig02_ida.png')
tend = ida_ano.iloc[-1] - ida_ano.iloc[0]
print(f"  Tendencia 2022->2024: {tend:+.3f} — {'MELHORA' if tend>0.05 else 'QUEDA' if tend<-0.05 else 'ESTAVEL'}")

# ── P3: IEG ──────────────────────────────────────────────────────────────────
print("[5/14] P3 — Engajamento (IEG) x IDA e IPV...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('P3 — Engajamento (IEG)', fontsize=14, fontweight='bold')
for ax, (y_col, y_label, cor) in zip(axes, [
    ('ida','IDA — Desempenho',CORES['secundaria']),
    ('ipv','IPV — Ponto de Virada',CORES['verde'])
]):
    d = df[['ieg', y_col]].dropna()
    ax.scatter(d['ieg'], d[y_col], alpha=0.12, color=cor, s=12)
    m, b = np.polyfit(d['ieg'], d[y_col], 1)
    xl = np.linspace(d['ieg'].min(), d['ieg'].max(), 100)
    ax.plot(xl, m*xl+b, color=CORES['destaque'], lw=2.2, label=f'y={m:.2f}x+{b:.2f}')
    r = d.corr().iloc[0, 1]
    ax.set_title(f'IEG x {y_col.upper()}  (r={r:.3f})', fontweight='bold')
    ax.set_xlabel('IEG'); ax.set_ylabel(y_label); ax.legend(fontsize=9)
plt.tight_layout(); salvar('fig03_ieg.png')

# ── P4: IAA ──────────────────────────────────────────────────────────────────
print("[6/14] P4 — Autoavaliacao (IAA)...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('P4 — Autoavaliacao (IAA)', fontsize=14, fontweight='bold')
for ax, (y_col, y_label, cor) in zip(axes, [
    ('ida','IDA',CORES['primaria']),
    ('ieg','IEG',CORES['amarelo'])
]):
    d = df[['iaa', y_col]].dropna()
    ax.scatter(d['iaa'], d[y_col], alpha=0.12, color=cor, s=12)
    m, b = np.polyfit(d['iaa'], d[y_col], 1)
    xl = np.linspace(d['iaa'].min(), d['iaa'].max(), 100)
    ax.plot(xl, m*xl+b, color=CORES['destaque'], lw=2.2)
    r = d.corr().iloc[0, 1]
    ax.set_title(f'IAA x {y_col.upper()}  (r={r:.3f})', fontweight='bold')
    ax.set_xlabel('IAA'); ax.set_ylabel(y_label)
plt.tight_layout(); salvar('fig04_iaa.png')

# ── P5: IPS ──────────────────────────────────────────────────────────────────
print("[7/14] P5 — Aspectos Psicossociais (IPS)...")
df['ips_grupo'] = pd.qcut(df['ips'], q=3,
    labels=['IPS Baixo (0-33%)','IPS Medio (33-66%)','IPS Alto (66-100%)'])
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('P5 — Aspectos Psicossociais (IPS)', fontsize=14, fontweight='bold')
pal_ips = ['#E74C3C','#F39C12','#1E8449']
for ax, (y_col, y_label) in zip(axes, [
    ('ida','IDA'),('ieg','IEG'),('inde_ano','INDE')
]):
    sns.boxplot(data=df.dropna(subset=['ips_grupo']), x='ips_grupo', y=y_col,
        palette=pal_ips, ax=ax, width=0.5, linewidth=1.2,
        flierprops=dict(marker='o', markersize=3, alpha=0.3))
    ax.set_title(f'{y_col.upper()} por Grupo IPS', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
plt.tight_layout(); salvar('fig05_ips.png')

# ── P6: IPP ──────────────────────────────────────────────────────────────────
print("[8/14] P6 — Avaliacoes Psicopedagogicas (IPP)...")
df_ipp = df[df['ipp'].notna()].copy()
df_ipp['ian_classe'] = df_ipp['ian'].apply(classificar_ian)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('P6 — Avaliacoes Psicopedagogicas (IPP)', fontsize=14, fontweight='bold')
sns.boxplot(data=df_ipp, x='ian_classe', order=['Adequado','Moderado','Severo'],
    y='ipp', palette=[CORES['verde'],CORES['amarelo'],CORES['destaque']],
    ax=axes[0], width=0.5, linewidth=1.2)
axes[0].set_title('IPP por Nivel IAN', fontweight='bold')
axes[1].scatter(df_ipp['ipp'], df_ipp['ian'], alpha=0.15, color=CORES['primaria'], s=12)
r = df_ipp[['ipp','ian']].corr().iloc[0, 1]
axes[1].set_title(f'IPP x IAN  (r={r:.3f})', fontweight='bold')
plt.tight_layout(); salvar('fig06_ipp.png')

# ── P7: IPV ──────────────────────────────────────────────────────────────────
print("[9/14] P7 — Ponto de Virada (IPV)...")
indicadores = ['iaa','ieg','ips','ipp','ida','ian']
corrs_ipv = df[indicadores+['ipv']].corr()['ipv'].drop('ipv').sort_values(ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('P7 — Ponto de Virada (IPV)', fontsize=14, fontweight='bold')
colors_b = [CORES['destaque'] if v < 0 else CORES['verde'] for v in corrs_ipv]
bars = axes[0].barh(corrs_ipv.index, corrs_ipv.values, color=colors_b, edgecolor='white', height=0.55)
axes[0].axvline(0, color='black', lw=0.8)
for bar, v in zip(bars, corrs_ipv.values):
    off = 0.005 if v >= 0 else -0.005
    axes[0].text(v+off, bar.get_y()+bar.get_height()/2, f'{v:.3f}',
        va='center', ha='left' if v>=0 else 'right', fontsize=9, fontweight='bold')
axes[0].set_title('Correlacao com IPV', fontweight='bold')
ipv_ano = df.groupby('ano_referencia')['ipv'].mean()
axes[1].plot(ipv_ano.index, ipv_ano.values, marker='o', color=CORES['primaria'], lw=2.5, ms=9)
axes[1].fill_between(ipv_ano.index, ipv_ano.values, alpha=0.12, color=CORES['primaria'])
for ano, val in ipv_ano.items():
    axes[1].text(ano, val+0.06, f'{val:.2f}', ha='center', fontweight='bold', fontsize=11)
axes[1].set_title('IPV Medio por Ano', fontweight='bold'); axes[1].set_ylim(0, 10)
plt.tight_layout(); salvar('fig07_ipv.png')

# ── P8: INDE multidim ─────────────────────────────────────────────────────────
print("[10/14] P8 — Multidimensionalidade dos Indicadores...")
indicadores_all = ['iaa','ieg','ips','ipp','ida','ipv','ian',
                   'nota_matematica','nota_portugues','media_indicadores']
corrs_inde = (df[indicadores_all+['inde_ano']].corr()['inde_ano']
              .drop('inde_ano').sort_values(ascending=False))
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('P8 — Multidimensionalidade dos Indicadores', fontsize=14, fontweight='bold')
colors_b = [CORES['verde'] if v>0.5 else CORES['secundaria'] if v>0.3 else CORES['cinza']
            for v in corrs_inde]
axes[0].barh(corrs_inde.index[::-1], corrs_inde.values[::-1],
    color=colors_b[::-1], edgecolor='white', height=0.6)
axes[0].axvline(0.5, color=CORES['destaque'], lw=1.5, ls='--', alpha=0.8, label='r=0.5')
axes[0].axvline(0.3, color=CORES['amarelo'],  lw=1.2, ls=':',  alpha=0.8, label='r=0.3')
axes[0].set_title('Correlacao com INDE', fontweight='bold'); axes[0].legend(fontsize=9)
cols_h = ['inde_ano','ida','ieg','iaa','ips','ipp','ipv','ian']
heat = df[cols_h].corr()
mask = np.triu(np.ones_like(heat, dtype=bool))
sns.heatmap(heat, mask=mask, annot=True, fmt='.2f', cmap='Blues',
    ax=axes[1], linewidths=0.4, annot_kws={'size':9}, vmin=0, vmax=1)
axes[1].set_title('Matriz de Correlacao', fontweight='bold')
plt.tight_layout(); salvar('fig08_multidim.png')

# ── P9: MODELO ML ─────────────────────────────────────────────────────────────
print("[11/14] P9 — Modelo Preditivo de Risco (ML)...")
df_model = df.copy()
df_model['em_risco'] = (
    (df_model['defasagem'] < 0) |
    (df_model['ian'] <= 5.0)    |
    (df_model['inde_ano'] < 6.5)
).astype(int)

FEATURES = [
    'fase','genero_feminino','instituicao_cod','anos_no_programa',
    'iaa','ieg','ips','ida','ipv',
    'nota_matematica','nota_portugues','nota_ingles',
    'media_notas','media_indicadores',
    'pedra_ano','pedra_2020','pedra_2021',
]
df_clean = df_model[FEATURES+['em_risco']].dropna(subset=['em_risco'])
X = df_clean[FEATURES]
y = df_clean['em_risco']

vc = y.value_counts()
print(f"  Em Risco: {vc.get(1,0):,} ({vc.get(1,0)/len(y)*100:.1f}%)  |  "
      f"Sem Risco: {vc.get(0,0):,} ({vc.get(0,0)/len(y)*100:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipe_lr = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('scl', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])
pipe_rf = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('clf', RandomForestClassifier(n_estimators=300, random_state=42,
        class_weight='balanced', n_jobs=-1, max_depth=10, min_samples_leaf=5))
])
pipe_gb = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('clf', GradientBoostingClassifier(n_estimators=300, random_state=42,
        learning_rate=0.05, max_depth=4, subsample=0.8))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados_cv = {}
print("  Cross-Validation (5-fold):")
for nome, pipe in [('Logistic Regression',pipe_lr),('Random Forest',pipe_rf),('Gradient Boosting',pipe_gb)]:
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    resultados_cv[nome] = scores
    print(f"    {nome:22s} | AUC: {scores.mean():.4f} +/- {scores.std():.4f}")

melhor_nome = max(resultados_cv, key=lambda k: resultados_cv[k].mean())
PIPES = {'Logistic Regression':pipe_lr,'Random Forest':pipe_rf,'Gradient Boosting':pipe_gb}
melhor_pipe = PIPES[melhor_nome]
melhor_pipe.fit(X_train, y_train)

y_pred = melhor_pipe.predict(X_test)
y_prob = melhor_pipe.predict_proba(X_test)[:, 1]
auc_test = roc_auc_score(y_test, y_prob)
ap_test  = average_precision_score(y_test, y_prob)

print(f"  Melhor modelo: {melhor_nome}")
print(f"  AUC-ROC: {auc_test:.4f}  |  Average Precision: {ap_test:.4f}")
print("\n  Relatorio de Classificacao:")
print(classification_report(y_test, y_pred, target_names=['Sem Risco','Em Risco']))

joblib.dump(melhor_pipe, 'modelo_risco_defasagem.pkl')
joblib.dump(FEATURES,    'features_modelo.pkl')
print("  Modelo salvo: modelo_risco_defasagem.pkl")

# Visualizações modelo
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(f'P9 — Modelo Preditivo ({melhor_nome})', fontsize=14, fontweight='bold')

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0,0].plot(fpr, tpr, color=CORES['primaria'], lw=2.5, label=f'AUC={auc_test:.3f}')
axes[0,0].plot([0,1],[0,1],'k--',alpha=0.4,lw=1)
axes[0,0].fill_between(fpr, tpr, alpha=0.08, color=CORES['primaria'])
axes[0,0].set_title('Curva ROC', fontweight='bold')
axes[0,0].set_xlabel('FPR'); axes[0,0].set_ylabel('TPR'); axes[0,0].legend()

prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[0,1].plot(rec, prec, color=CORES['verde'], lw=2.5, label=f'AP={ap_test:.3f}')
axes[0,1].fill_between(rec, prec, alpha=0.08, color=CORES['verde'])
axes[0,1].set_title('Curva Precision-Recall', fontweight='bold')
axes[0,1].set_xlabel('Recall'); axes[0,1].set_ylabel('Precision'); axes[0,1].legend()

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Sem Risco','Em Risco']).plot(
    ax=axes[0,2], colorbar=False, cmap='Blues')
axes[0,2].set_title('Matriz de Confusao', fontweight='bold')

if hasattr(melhor_pipe['clf'], 'feature_importances_'):
    imp = pd.Series(melhor_pipe['clf'].feature_importances_, index=FEATURES).sort_values(ascending=True)
    imp.plot(kind='barh', ax=axes[1,0], color=CORES['primaria'], edgecolor='white')
    axes[1,0].set_title('Importancia das Features', fontweight='bold')
else:
    coef = pd.Series(np.abs(melhor_pipe['clf'].coef_[0]), index=FEATURES).sort_values(ascending=True)
    coef.plot(kind='barh', ax=axes[1,0], color=CORES['primaria'], edgecolor='white')
    axes[1,0].set_title('|Coeficientes|', fontweight='bold')

axes[1,1].hist(y_prob[y_test==0], bins=30, alpha=0.65, color=CORES['verde'], label='Sem Risco', density=True)
axes[1,1].hist(y_prob[y_test==1], bins=30, alpha=0.65, color=CORES['destaque'], label='Em Risco', density=True)
axes[1,1].axvline(0.5, color='black', ls='--', lw=1.5)
axes[1,1].set_title('Distribuicao P(Em Risco)', fontweight='bold')
axes[1,1].set_xlabel('P(Em Risco)'); axes[1,1].legend()

pd.DataFrame(resultados_cv).boxplot(ax=axes[1,2], patch_artist=True, grid=False)
axes[1,2].set_title('Comparacao AUC — CV 5-fold', fontweight='bold')
axes[1,2].set_ylabel('AUC'); axes[1,2].set_ylim(0.7, 1.0)
axes[1,2].tick_params(axis='x', rotation=15)

plt.tight_layout(); salvar('fig09_modelo.png')

# ── P10: Efetividade ──────────────────────────────────────────────────────────
print("[12/14] P10 — Efetividade do Programa (Pedras)...")
PEDRA_LABEL = {1:'Quartzo',2:'Agata',3:'Ametista',4:'Topazio'}
df_pedra = df[df['pedra_ano'].isin([1,2,3,4])].copy()
df_pedra['pedra_nome'] = df_pedra['pedra_ano'].map(PEDRA_LABEL)
ordem_p  = ['Quartzo','Agata','Ametista','Topazio']
cores_p  = [CORES['cinza'],CORES['amarelo'],CORES['secundaria'],CORES['primaria']]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('P10 — Efetividade do Programa por Pedra', fontsize=14, fontweight='bold')
axes = axes.flatten()
for ax, (col, label) in zip(axes, [
    ('inde_ano','INDE Global'),('ida','IDA'),('ieg','IEG'),('ipv','IPV')
]):
    medias = df_pedra.groupby('pedra_nome')[col].mean().reindex(ordem_p)
    bars = ax.bar(ordem_p, medias.values, color=cores_p, edgecolor='white', width=0.55)
    for bar, v in zip(bars, medias.values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.05, f'{v:.2f}',
            ha='center', fontweight='bold', fontsize=10)
    ax.plot(range(len(medias)), medias.values, 'o--',
        color=CORES['destaque'], lw=1.5, ms=6, alpha=0.7)
    ax.set_title(f'{label} por Pedra', fontweight='bold')
    ax.set_ylabel(label); ax.set_ylim(0, 10.5)
plt.tight_layout(); salvar('fig10_efetividade.png')

# ── P11: Insights ─────────────────────────────────────────────────────────────
print("[13/14] P11 — Insights Adicionais...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('P11 — Insights Adicionais', fontsize=14, fontweight='bold')

genero_inde = df.groupby(['ano_referencia','genero'])['inde_ano'].mean().unstack()
genero_inde.plot(kind='bar', ax=axes[0],
    color=[CORES['destaque'],CORES['secundaria']], edgecolor='white', width=0.55)
axes[0].set_title('INDE Medio por Genero e Ano', fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].legend(title='Genero')

df_t = df[df['anos_no_programa'].between(0, 12)].copy()
t_inde = df_t.groupby('anos_no_programa')['inde_ano'].mean()
axes[1].plot(t_inde.index, t_inde.values, marker='o', color=CORES['verde'], lw=2.5, ms=7)
axes[1].fill_between(t_inde.index, t_inde.values, alpha=0.12, color=CORES['verde'])
axes[1].set_title('INDE Medio x Anos no Programa', fontweight='bold')
axes[1].set_xlabel('Anos no Programa')

alunos_ano = df.groupby('ano_referencia').size()
bars = axes[2].bar(alunos_ano.index.astype(str), alunos_ano.values,
    color=ESTILO[:3], edgecolor='white', width=0.5)
for bar, n in zip(bars, alunos_ano.values):
    axes[2].text(bar.get_x()+bar.get_width()/2, n+5, f'{n:,}',
        ha='center', fontweight='bold', fontsize=11)
axes[2].set_title('Crescimento da Base de Alunos', fontweight='bold')
plt.tight_layout(); salvar('fig11_insights.png')

# Segmentação de risco por fase
df_r = df.copy()
X_full = df_r[FEATURES].copy()
for col in FEATURES:
    if col in X_full.columns:
        X_full[col] = X_full[col].fillna(X_full[col].median())
df_r['prob_risco'] = melhor_pipe.predict_proba(X_full)[:, 1]
df_r['nivel_risco'] = pd.cut(df_r['prob_risco'], bins=[0,0.3,0.6,1.0],
    labels=['Baixo Risco','Risco Moderado','Alto Risco'])

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Insight Extra — Segmentacao de Risco', fontsize=14, fontweight='bold')
df_fp = df_r[df_r['fase'].between(0, 8)].copy()
df_fp['fase_label'] = df_fp['fase'].apply(lambda x: 'ALFA' if x==0 else f'F{int(x)}')
rf = df_fp.groupby(['fase_label','nivel_risco']).size().unstack(fill_value=0)
of = ['ALFA'] + [f'F{i}' for i in range(1, 9)]
of = [f for f in of if f in rf.index]
rp = rf.div(rf.sum(axis=1), axis=0) * 100
for c in ['Baixo Risco','Risco Moderado','Alto Risco']:
    if c not in rp.columns: rp[c] = 0
rp[['Baixo Risco','Risco Moderado','Alto Risco']].reindex(of).plot(
    kind='bar', stacked=True, ax=axes[0],
    color=[CORES['verde'],CORES['amarelo'],CORES['destaque']],
    edgecolor='white', width=0.65)
axes[0].set_title('Risco por Fase', fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].legend()
ra  = df_r.groupby(['ano_referencia','nivel_risco']).size().unstack(fill_value=0)
rap = ra.div(ra.sum(axis=1), axis=0) * 100
for c in ['Baixo Risco','Risco Moderado','Alto Risco']:
    if c not in rap.columns: rap[c] = 0
rap[['Baixo Risco','Risco Moderado','Alto Risco']].plot(
    kind='bar', stacked=True, ax=axes[1],
    color=[CORES['verde'],CORES['amarelo'],CORES['destaque']],
    edgecolor='white', width=0.5)
axes[1].set_title('Risco por Ano', fontweight='bold')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend()
plt.tight_layout(); salvar('fig11b_risco_fase.png')


# =============================================================================
# SEÇÃO 4 — RESUMO FINAL
# =============================================================================
print("\n[14/14] Resumo final...")
print("\n" + "=" * 60)
print("  RESUMO FINAL — PASSOS MAGICOS DATATHON")
print("=" * 60)
print(f"  Total de alunos analisados : {len(df):,}")
print(f"  Periodo                    : 2022, 2023, 2024")
print(f"  Features do modelo         : {len(FEATURES)}")
print(f"  Modelo selecionado         : {melhor_nome}")
print(f"  AUC-ROC (teste)            : {auc_test:.4f}")
print(f"  Average Precision (teste)  : {ap_test:.4f}")
pct_risco = df_clean['em_risco'].mean() * 100
print(f"  % alunos em risco          : {pct_risco:.1f}%")
print("=" * 60)
print(f"\nArquivos gerados em: ./{OUTPUT_DIR}/")
for i in range(1, 12):
    tag = f"fig{str(i).zfill(2)}"
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith(tag):
            print(f"  {f}")
print(f"\n  modelo_risco_defasagem.pkl")
print(f"  features_modelo.pkl")
print(f"  {PARQUET_FILE}")
print("\nScript concluido com sucesso!")