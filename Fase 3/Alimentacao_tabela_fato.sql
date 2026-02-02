CREATE OR REPLACE TABLE `techchallenge3-481414.techchallenge_pnad.fact_pnad_indicadores` (
  ref_month DATE,
  tema STRING,
  indicador STRING,

  nivel_territorial STRING,
  abertura_territorial STRING,

  variavel_abertura_1 STRING,
  categoria_abertura_1 STRING,
  variavel_abertura_2 STRING,
  categoria_abertura_2 STRING,

  valor FLOAT64
);



INSERT INTO `techchallenge3-481414.techchallenge_pnad.fact_pnad_indicadores`
SELECT
  DATE '2020-09-01' AS ref_month,
  'saude' AS tema,
  indicador,
  nvel_territorial AS nivel_territorial,
  abertura_territorial,
  varivel_de_abertura_1 AS variavel_abertura_1,
  categoria_de_abertura_1 AS categoria_abertura_1,
  varivel_de_abertura_2  AS variavel_abertura_2,
  categoria_de_abertura_2 AS categoria_abertura_2,
  SAFE_CAST(setembro AS FLOAT64) AS valor
FROM `techchallenge3-481414.techchallenge_pnad.202009_saude_br_gr`

UNION ALL

SELECT
  DATE '2020-10-01' AS ref_month,
  'saude' AS tema,
  indicador,
  nvel_territorial AS nivel_territorial,
  abertura_territorial,
  varivel_de_abertura_1 AS variavel_abertura_1,
  categoria_de_abertura_1 AS categoria_abertura_1,
  varivel_de_abertura_2  AS variavel_abertura_2,
  categoria_de_abertura_2 AS categoria_abertura_2,
  SAFE_CAST(outubro AS FLOAT64) AS valor
FROM `techchallenge3-481414.techchallenge_pnad.202010_saude_br_gr`

UNION ALL

SELECT
  DATE '2020-11-01' AS ref_month,
  'saude' AS tema,
  indicador,
  nvel_territorial AS nivel_territorial,
  abertura_territorial,
  varivel_de_abertura_1 AS variavel_abertura_1,
  categoria_de_abertura_1 AS categoria_abertura_1,
  varivel_de_abertura_2  AS variavel_abertura_2,
  categoria_de_abertura_2 AS categoria_abertura_2,
  SAFE_CAST(novembro AS FLOAT64) AS valor
FROM `techchallenge3-481414.techchallenge_pnad.202011_saude_br_gr`;



INSERT INTO `techchallenge3-481414.techchallenge_pnad.fact_pnad_indicadores`
SELECT
  DATE '2020-09-01',
  'trabalho',
  indicador,
  nvel_territorial,
  abertura_territorial,
  varivel_de_abertura_1,
  categoria_de_abertura_1,
  varivel_de_abertura_2,
  categoria_de_abertura_2,
  SAFE_CAST(setembro AS FLOAT64)
FROM `techchallenge3-481414.techchallenge_pnad.202009_trabalho_br_gr`

UNION ALL
SELECT
  DATE '2020-10-01',
  'trabalho',
  indicador,
  nvel_territorial,
  abertura_territorial,
  varivel_de_abertura_1,
  categoria_de_abertura_1,
  varivel_de_abertura_2,
  categoria_de_abertura_2,
  SAFE_CAST(outubro AS FLOAT64)
FROM `techchallenge3-481414.techchallenge_pnad.202010_trabalho_br_gr`

UNION ALL
SELECT
  DATE '2020-11-01',
  'trabalho',
  indicador,
  nvel_territorial,
  abertura_territorial,
  varivel_de_abertura_1,
  categoria_de_abertura_1,
  varivel_de_abertura_2,
  categoria_de_abertura_2,
  SAFE_CAST(novembro AS FLOAT64)
FROM `techchallenge3-481414.techchallenge_pnad.202011_trabalho_br_gr`;





INSERT INTO `techchallenge3-481414.techchallenge_pnad.fact_pnad_indicadores`
SELECT
  DATE '2020-09-01',
  'itens_limpeza',
  indicador,
  nvel_territorial,
  abertura_territorial,
  varivel_de_abertura_1,
  categoria_de_abertura_1,
  varivel_de_abertura_2,
  categoria_de_abertura_2,
  SAFE_CAST(setembro AS FLOAT64)
FROM `techchallenge3-481414.techchallenge_pnad.202009_itens_de_limepeza`

UNION ALL
SELECT
  DATE '2020-10-01',
  'itens_limpeza',
  indicador,
  nvel_territorial,
  abertura_territorial,
  varivel_de_abertura_1,
  categoria_de_abertura_1,
  varivel_de_abertura_2,
  categoria_de_abertura_2,
  SAFE_CAST(outubro AS FLOAT64)
FROM `techchallenge3-481414.techchallenge_pnad.202010_itens_de_limpeza`

UNION ALL
SELECT
  DATE '2020-11-01',
  'itens_limpeza',
  indicador,
  nvel_territorial,
  abertura_territorial,
  varivel_de_abertura_1,
  categoria_de_abertura_1,
  varivel_de_abertura_2,
  categoria_de_abertura_2,
  SAFE_CAST(novembro AS FLOAT64)
FROM `techchallenge3-481414.techchallenge_pnad.202011_itens_de_limpeza`;


SELECT
  tema,
  COUNT(*) AS linhas
FROM `techchallenge3-481414.techchallenge_pnad.fact_pnad_indicadores`
GROUP BY tema;



