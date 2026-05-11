# 🎓 Passos Mágicos — Guia Completo de Uso e Deploy

**Curso:** Data Analytics — POSTECH / FIAP  
**Case:** Associação Passos Mágicos — Datathon 2024–2025

---

## 📁 Arquivos do Projeto

| Arquivo | Descrição |
|---------|-----------|
| `Script_Passos_Magicos.py` | Script principal — tratamento, análise e treinamento do modelo |
| `app.py` | App Streamlit — interface interativa para predição de risco |
| `requirements.txt` | Dependências Python |
| `README_deploy.md` | Este arquivo |

### Arquivos gerados automaticamente ao rodar o script:

| Arquivo | Gerado por |
|---------|-----------|
| `base_tratada_final.parquet` | `Script_Passos_Magicos.py` |
| `modelo_risco_defasagem.pkl` | `Script_Passos_Magicos.py` |
| `features_modelo.pkl` | `Script_Passos_Magicos.py` |
| `output_figuras/fig01_ian.png` … `fig11b_risco_fase.png` | `Script_Passos_Magicos.py` |

---

## ⚙️ Pré-requisitos

- Python 3.9 ou superior
- VS Code (recomendado) ou qualquer terminal

---

## 📦 Instalação das Dependências

Abra o terminal no VS Code (`Ctrl + '`) e execute:

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn joblib pyarrow openpyxl streamlit
```

---

## 🚀 Passo 1 — Rodar o Script Principal

O script faz tudo em sequência: trata os dados, responde as 11 perguntas de negócio, treina o modelo ML e salva os arquivos necessários para o app.

**1. Coloque na mesma pasta:**

```
📁 Datathon/
├── Script_Passos_Magicos.py
├── BASE DE DADOS PEDE 2024 - DATATHON.xlsx   ← Excel original
```

> Se o arquivo `base_tratada_final.parquet` já existir na pasta, o script pula o tratamento e vai direto para as análises (muito mais rápido).

**2. Rode no terminal:**

```powershell
python Script_Passos_Magicos.py
```

**O que será gerado:**
- `base_tratada_final.parquet` — base consolidada 2022–2024
- `modelo_risco_defasagem.pkl` — modelo ML treinado
- `features_modelo.pkl` — lista de features do modelo
- `output_figuras/` — pasta com todos os gráficos das 11 perguntas

---

## 🌐 Passo 2 — Rodar o App Streamlit Localmente

Após rodar o script, a pasta estará assim:

```
📁 Datathon/
├── Script_Passos_Magicos.py
├── app.py
├── requirements.txt
├── base_tratada_final.parquet       ← gerado pelo script
├── modelo_risco_defasagem.pkl       ← gerado pelo script
├── features_modelo.pkl              ← gerado pelo script
└── output_figuras/
    ├── fig01_ian.png
    ├── ...
    └── fig11b_risco_fase.png
```

**Rode no terminal:**

```powershell
streamlit run app.py
```

O navegador abrirá automaticamente em `http://localhost:8501`.

> ⚠️ O app funciona mesmo sem os arquivos `.pkl` e `.parquet`, usando uma heurística de demonstração. Para resultados reais, rode o script principal primeiro.

---

## 🖥️ Funcionalidades do App

| Modo | O que faz |
|------|-----------|
| 🧑 **Avaliação Individual** | Preencha os indicadores do aluno via sliders → probabilidade de risco + gauge visual + gráfico comparativo com a média histórica |
| 📂 **Lote (CSV/Excel)** | Upload de arquivo com vários alunos → predição em massa + tabela com alunos em alto risco + download do resultado em CSV |
| 📊 **Painel Histórico** | Visualização da base 2022–2024 com filtros por ano e fase + 6 gráficos analíticos |

---

## ☁️ Passo 3 — Publicar na Web (Streamlit Community Cloud)

Para disponibilizar o app com URL pública e gratuita:

**1. Crie conta em:** https://share.streamlit.io

**2. Crie um repositório público no GitHub e suba os arquivos:**

```
📁 repositorio-github/
├── app.py
├── requirements.txt
├── modelo_risco_defasagem.pkl
├── features_modelo.pkl
└── base_tratada_final.parquet
```

**3. No Streamlit Cloud:**
- Clique em **"New app"**
- Selecione o repositório e o branch
- Em **"Main file path"** selecione `app.py`
- Clique em **"Deploy"**

Em poucos minutos o app estará disponível com URL pública no formato:
```
https://seu-usuario-passosmagicos.streamlit.app
```

---

## 🔁 Fluxo Resumido

```
BASE DE DADOS PEDE 2024 - DATATHON.xlsx
            │
            ▼
  python Script_Passos_Magicos.py
            │
            ├── base_tratada_final.parquet
            ├── modelo_risco_defasagem.pkl
            ├── features_modelo.pkl
            └── output_figuras/ (11 gráficos)
                        │
                        ▼
            streamlit run app.py
                        │
                        ▼
         http://localhost:8501  (local)
                   ou
    https://seu-app.streamlit.app  (web)
```

---

## ❓ Problemas Comuns

| Erro | Solução |
|------|---------|
| `No such file or directory` | Verifique se o terminal está na pasta correta (`cd C:\Users\...\Datathon`) |
| `ModuleNotFoundError` | Rode `pip install -r requirements.txt` |
| `FileNotFoundError: BASE DE DADOS...` | Coloque o Excel na mesma pasta do script |
| App abre sem modelo | Rode `Script_Passos_Magicos.py` primeiro para gerar os `.pkl` |
| `streamlit: command not found` | Rode `pip install streamlit` e reinicie o terminal |

---

*Passos Mágicos · POSTECH FIAP · Datathon 2024–2025*