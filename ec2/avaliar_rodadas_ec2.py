"""Calcula as métricas das rodadas do emissor fotovoltaico."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ["normal", "soiling", "sombreamento"]
MIN_POA_AVALIACAO = 100  # W/m², mesmo recorte do teste no Colab.


def matriz_de_confusao(estados, previsoes):
    return pd.crosstab(estados, previsoes).reindex(
        index=CLASSES, columns=CLASSES, fill_value=0
    )


def avaliar(path):
    dados = pd.read_csv(path)
    if dados.empty:
        raise ValueError(f"{path}: o arquivo ainda não contém leituras.")

    horarios = pd.to_datetime(dados["timestamp"], utc=True)
    if horarios.isna().any() or not horarios.diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all():
        raise ValueError(f"{path}: a sequência precisa ter uma leitura a cada cinco minutos.")

    meta = dados[["cenario", "semente"]].drop_duplicates()
    if len(meta) != 1:
        raise ValueError(f"{path}: cada CSV precisa conter um único cenário e uma única semente.")

    diurnas = dados.loc[dados["poa_ceu_limpo"] >= MIN_POA_AVALIACAO]
    avaliadas = diurnas.loc[diurnas["status"].eq("classificado")]
    if avaliadas.empty:
        raise ValueError(f"{path}: ainda não há classificações no recorte diurno.")
    if not avaliadas["estado"].isin(CLASSES).all() or not avaliadas["previsao"].isin(CLASSES).all():
        raise ValueError(f"{path}: há classes ausentes ou inválidas em leituras classificadas.")

    matriz = matriz_de_confusao(avaliadas["estado"], avaliadas["previsao"])
    acertos = pd.Series(np.diag(matriz), index=CLASSES)
    suporte = matriz.sum(axis=1)
    previstas = matriz.sum(axis=0)
    metricas = pd.DataFrame({
        "Precisão": acertos.div(previstas.where(previstas > 0)).fillna(0),
        "Revocação": acertos.div(suporte.where(suporte > 0)).fillna(0),
        "F1": (2 * acertos).div((suporte + previstas).where(suporte + previstas > 0)).fillna(0),
        "Amostras": suporte,
    })

    cenario, semente = meta.iloc[0]
    print(f"\n{path.name} | cenário: {cenario} | semente: {semente}")
    print(f"Leituras registradas: {len(dados)} | ciclos no arquivo: {dados['ciclo'].nunique()}")
    print(f"Período registrado: {dados['timestamp'].iloc[0]} a {dados['timestamp'].iloc[-1]}")
    print(f"Leituras no recorte: {len(diurnas)} | sem classificação: {len(diurnas) - len(avaliadas)}")
    print(f"F1 macro: {metricas['F1'].mean():.4f} | acurácia: {acertos.sum() / len(avaliadas):.2%}")
    print(f"Poeira prevista como normal: {matriz.loc['soiling', 'normal']} de {suporte['soiling']}")
    if (suporte == 0).any():
        print("Há classes sem exemplos de referência. O F1 macro atribui zero ao F1 indefinido.")
    print("\nMétricas por classe")
    print(metricas.round(4).to_string())
    print("\nMatriz de confusão (linhas: classe simulada, colunas: previsão)")
    print(matriz.to_string())
    return metricas, matriz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arquivos", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.arquivos:
        try:
            avaliar(path)
        except (OSError, ValueError, KeyError) as erro:
            parser.exit(1, f"Erro ao avaliar: {erro}\n")


if __name__ == "__main__":
    main()
