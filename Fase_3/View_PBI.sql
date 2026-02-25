CREATE OR REPLACE VIEW `techchallenge3-481414.techchallenge_pnad.vw_powerbi` AS
SELECT
  ref_month,
  FORMAT_DATE('%Y-%m', ref_month) AS ref_ym,
  tema,
  indicador,
  nivel_territorial,
  abertura_territorial,
  variavel_abertura_1,
  categoria_abertura_1,
  variavel_abertura_2,
  categoria_abertura_2,
  valor
FROM `techchallenge3-481414.techchallenge_pnad.fact_pnad_indicadores`
WHERE ref_month IN (DATE '2020-09-01', DATE '2020-10-01', DATE '2020-11-01')
  AND valor IS NOT NULL;


CREATE OR REPLACE VIEW `techchallenge3-481414.techchallenge_pnad.vw_uf` AS
SELECT *
FROM `techchallenge3-481414.techchallenge_pnad.vw_powerbi`
WHERE nivel_territorial = 'UF';



CREATE OR REPLACE VIEW `techchallenge3-481414.techchallenge_pnad.vw_br_gr` AS
SELECT *
FROM `techchallenge3-481414.techchallenge_pnad.vw_powerbi`
WHERE nivel_territorial IN ('Brasil', 'Grande Região');
