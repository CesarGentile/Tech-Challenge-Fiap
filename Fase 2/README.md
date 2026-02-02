Este projeto tem como objetivo prever a tendência (alta ou baixa) do índice Ibovespa (IBOV) para o próximo pregão utilizando modelos de Machine Learning treinados com dados históricos dos últimos 25 anos. O processo envolve coleta automática dos dados, teste de estacionariedade, criação de variáveis derivadas (retornos, volatilidade e médias móveis), definição do target binário, treinamento de modelos supervisionados (Logistic Regression e XGBoost), comparação com um baseline simples e geração automática de gráficos de desempenho para análise visual.

As tecnologias utilizadas incluem as bibliotecas yfinance, pandas, numpy, statsmodels, scikit-learn, xgboost e matplotlib.

Para executar o projeto, recomenda-se criar um ambiente virtual com:
python -m venv .venv
e ativá-lo no Windows PowerShell com:
.venv\Scripts\Activate

Em seguida, instale as dependências:
pip install -r requirements.txt

E execute o pipeline principal com:
python -m src.main

Ao rodar o código, os modelos são treinados, avaliados e os gráficos de desempenho são automaticamente gerados dentro da pasta “figures”. Os principais gráficos produzidos são:

Comparação de acurácia entre modelos:


Matriz de confusão do XGBoost:


Tendência real vs prevista (últimos 30 dias):


O projeto demonstra que, embora prever oscilações diárias do Ibovespa seja uma tarefa desafiadora devido à alta volatilidade do mercado financeiro, modelos como o XGBoost conseguem capturar padrões relevantes e superar estratégias simples como o baseline. Há um grande potencial para evolução futura, incluindo testes com redes neurais LSTM/GRU, inserção de indicadores técnicos, expansão do horizonte de previsão, otimização de hiperparâmetros e uso de métodos avançados de regularização.
