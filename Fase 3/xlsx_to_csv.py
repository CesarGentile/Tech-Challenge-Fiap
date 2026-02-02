import os
import re
from pathlib import Path
import pandas as pd


def sanitize_name(text: str, max_len: int = 120) -> str:
    """
    Normaliza nomes de arquivos/abas para algo seguro:
    - minúsculo
    - sem espaços e caracteres estranhos
    - limita tamanho
    """
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "_", text)                 # espaços -> _
    text = re.sub(r"[^a-z0-9_]+", "", text)          # remove caracteres
    text = re.sub(r"_+", "_", text).strip("_")       # colapsa underscores
    return text[:max_len] if len(text) > max_len else text


def convert_xlsx_to_csvs(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(list(input_path.glob("*.xlsx")))
    if not xlsx_files:
        raise FileNotFoundError(f"Nenhum .xlsx encontrado em: {input_path.resolve()}")

    print(f"Encontrados {len(xlsx_files)} arquivo(s) .xlsx em {input_path.resolve()}")
    print(f"Saída CSV em: {out_path.resolve()}\n")

    for xlsx_file in xlsx_files:
        base = sanitize_name(xlsx_file.stem)
        print(f"-> Lendo: {xlsx_file.name}")

        try:
            xl = pd.ExcelFile(xlsx_file, engine="openpyxl")
        except Exception as e:
            print(f"   [ERRO] Não consegui abrir {xlsx_file.name}: {e}")
            continue

        for sheet in xl.sheet_names:
            sheet_safe = sanitize_name(sheet)
            csv_name = f"{base}__{sheet_safe}.csv"
            csv_path = out_path / csv_name

            try:
                df = xl.parse(sheet_name=sheet, dtype=str)  # lê tudo como texto p/ não quebrar tipos
                # Remove colunas totalmente vazias
                df = df.dropna(axis=1, how="all")
                # Remove linhas totalmente vazias
                df = df.dropna(axis=0, how="all")

                # Padroniza nomes de colunas
                df.columns = [sanitize_name(c) for c in df.columns]

                # Salva CSV UTF-8
                df.to_csv(csv_path, index=False, encoding="utf-8")
                print(f"   OK: {csv_name}  ({len(df)} linhas, {len(df.columns)} colunas)")
            except Exception as e:
                print(f"   [ERRO] Aba '{sheet}' falhou: {e}")

        print()

    print("Concluído ✅")


if __name__ == "__main__":
    # Ajuste os caminhos aqui:
    INPUT_DIR = "."            # pasta onde estão os XLSX
    OUTPUT_DIR = "./csv_out"   # pasta de saída

    convert_xlsx_to_csvs(INPUT_DIR, OUTPUT_DIR)