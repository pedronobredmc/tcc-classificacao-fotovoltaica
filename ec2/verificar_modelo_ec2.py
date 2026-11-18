"""Confere os atributos e as previsões da EC2 com exemplos exportados do Colab."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


GRANDEZAS = ["tensao", "corrente", "temp_celula"]


def preparar_entrada(leituras, artefato):
    serie = pd.DataFrame(leituras)
    serie["timestamp"] = pd.to_datetime(serie["timestamp"])
    serie = serie.set_index("timestamp").sort_index()
    frequencia = pd.Timedelta(artefato["frequencia"])
    janelas = artefato["janelas_minutos_e_leituras"]
    tamanho = max(n for _, n in janelas) + 1
    esperados = pd.date_range(end=serie.index[-1], periods=tamanho, freq=frequencia)
    if not serie.index.equals(esperados):
        raise ValueError(f"São necessárias {tamanho} leituras consecutivas.")
    if artefato["reinicio_diario"] and serie.index.normalize().nunique() != 1:
        raise ValueError("As leituras da janela precisam pertencer ao mesmo dia.")
    if not np.isfinite(serie[GRANDEZAS].to_numpy(dtype=float)).all():
        raise ValueError("A janela contém leituras ausentes ou não finitas.")

    atual = serie.iloc[-1]
    atributos = {coluna: atual[coluna] for coluna in GRANDEZAS}
    for coluna in GRANDEZAS:
        for minutos, n in janelas:
            anteriores = serie[coluna].iloc[-n - 1 : -1]
            atributos[f"{coluna}_media_{minutos}min"] = anteriores.mean()
            atributos[f"{coluna}_variacao_{minutos}min"] = atual[coluna] - anteriores.iloc[0]

    return pd.DataFrame([atributos]).loc[:, artefato["atributos"]]


def verificar(artefato, referencia):
    modelo = artefato["modelo"]
    exemplos = referencia["exemplos"]
    if referencia["atributos"] != artefato["atributos"]:
        raise ValueError("A lista de atributos do Colab difere da lista do modelo salvo.")
    if {exemplo["classe_referencia"] for exemplo in exemplos} != set(modelo.classes_):
        raise ValueError("A referência precisa incluir exemplos das três classes simuladas.")

    for exemplo in exemplos:
        instante = exemplo["leituras"][-1]["timestamp"]
        entrada = preparar_entrada(exemplo["leituras"], artefato)
        np.testing.assert_allclose(
            entrada.iloc[0].to_numpy(),
            exemplo["atributos_colab"],
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Atributos divergentes em {instante}",
        )
        previsao = modelo.predict(entrada)[0]
        if previsao != exemplo["previsao_colab"]:
            raise ValueError(
                f"{instante}: Colab previu {exemplo['previsao_colab']}, "
                f"EC2 previu {previsao}."
            )
        print(f"{instante} | referência: {exemplo['classe_referencia']} | "
              f"previsão em ambos: {previsao}")
    print(f"Colab e EC2 produziram os mesmos atributos e previsões em {len(exemplos)} exemplos.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modelo", type=Path)
    parser.add_argument("referencia", type=Path)
    args = parser.parse_args()
    artefato = joblib.load(args.modelo)
    referencia = json.loads(args.referencia.read_text(encoding="utf-8"))
    verificar(artefato, referencia)


if __name__ == "__main__":
    main()
