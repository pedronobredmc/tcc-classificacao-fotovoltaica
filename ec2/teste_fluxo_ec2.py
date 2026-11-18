"""Serve o Random Forest localmente e testa o envio de leituras por HTTP."""

import argparse
import json
import math
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import pandas as pd

from avaliar_rodadas_ec2 import CLASSES, MIN_POA_AVALIACAO, matriz_de_confusao
from leituras_csv import ler_recentes
from verificar_modelo_ec2 import GRANDEZAS, preparar_entrada


LEITURAS_RECENTES = 288
NOMES_CLASSES = {
    "normal": "Operação normal",
    "soiling": "Poeira",
    "sombreamento": "Sombreamento",
}


def metricas_recentes(linhas):
    avaliadas = pd.DataFrame.from_records(
        (
            (linha["estado"], linha["previsao"])
            for linha in linhas
            if linha["status"] == "classificado"
            and float(linha["poa_ceu_limpo"]) >= MIN_POA_AVALIACAO
        ),
        columns=["estado", "previsao"],
    )
    matriz = matriz_de_confusao(avaliadas["estado"], avaliadas["previsao"])
    por_classe = []
    for classe in CLASSES:
        previstas = int(matriz[classe].sum())
        acertos = int(matriz.at[classe, classe])
        por_classe.append({
            "classe": NOMES_CLASSES[classe],
            "precisao": acertos / previstas if previstas else None,
            "previsoes": previstas,
            "acertos": acertos,
        })
    total = sum(item["previsoes"] for item in por_classe)
    acertos = sum(item["acertos"] for item in por_classe)
    resumo = {
        "acuracia": acertos / total if total else None,
        "previsoes": total,
        "acertos": acertos,
    }
    return por_classe, resumo


class Classificador:
    def __init__(self, artefato):
        self.artefato = artefato
        self.frequencia = pd.Timedelta(artefato["frequencia"])
        self.tamanho = max(n for _, n in artefato["janelas_minutos_e_leituras"]) + 1
        self.historico = deque(maxlen=self.tamanho)
        self.ultimo_instante = None
        self.ultima_resposta = None

    def receber(self, dados):
        if not isinstance(dados, dict) or not isinstance(dados.get("timestamp"), str):
            raise ValueError("Envie um timestamp no formato ISO 8601 com fuso horário.")
        instante = pd.Timestamp(dados["timestamp"])
        if pd.isna(instante) or instante.tzinfo is None:
            raise ValueError("Envie um timestamp válido com fuso horário.")
        timestamp = instante.isoformat()
        leitura = {"timestamp": timestamp}
        for campo in GRANDEZAS:
            valor = dados.get(campo)
            if type(valor) not in (int, float) or not math.isfinite(valor):
                raise ValueError(f"{campo} precisa ser um número finito.")
            leitura[campo] = valor

        ultimo = self.ultimo_instante
        if ultimo is not None:
            if instante < ultimo:
                raise ValueError("Envie as leituras em ordem cronológica.")
            if instante == ultimo:
                if leitura != self.historico[-1]:
                    raise ValueError("Já existe uma leitura diferente nesse instante.")
                return self.ultima_resposta
            mudou_dia = self.artefato["reinicio_diario"] and instante.date() != ultimo.date()
            if mudou_dia or instante - ultimo != self.frequencia:
                self.historico.clear()

        self.historico.append(leitura)
        faltam = self.tamanho - len(self.historico)
        if faltam:
            resposta = {"status": "aguardando_historico", "faltam": faltam}
        else:
            entrada = preparar_entrada(list(self.historico), self.artefato)
            classe = self.artefato["modelo"].predict(entrada)[0]
            resposta = {"status": "classificado", "classe": str(classe)}
        self.ultimo_instante = instante
        self.ultima_resposta = {"timestamp": timestamp, **resposta}
        return self.ultima_resposta


class Handler(BaseHTTPRequestHandler):
    def responder(self, codigo, dados):
        corpo = json.dumps(dados, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        if self.path not in ("/previsoes", "/metricas", "/metricas/resumo") or self.server.csv_monitorado is None:
            self.responder(404, {"erro": "Histórico de previsões indisponível."})
            return
        linhas = ler_recentes(self.server.csv_monitorado, LEITURAS_RECENTES)
        if self.path != "/previsoes":
            por_classe, resumo = metricas_recentes(linhas)
            self.responder(200, por_classe if self.path == "/metricas" else [resumo])
            return
        dados = []
        for linha in reversed(linhas):
            previsao = linha["previsao"] or "aguardando_historico"
            if float(linha["poa_ceu_limpo"]) < MIN_POA_AVALIACAO:
                previsao = "fora da faixa de avaliação"
            dados.append({
                "timestamp": linha["timestamp"],
                **{campo: float(linha[campo]) for campo in GRANDEZAS},
                "classe_simulada": linha["estado"],
                "previsao": previsao,
            })
        self.responder(200, dados)

    def do_POST(self):
        if self.path != "/leituras":
            self.responder(404, {"erro": "Use POST /leituras."})
            return
        try:
            tamanho = int(self.headers.get("Content-Length", "0"))
            if not 0 < tamanho <= 4096:
                raise ValueError("Tamanho inválido para uma leitura.")
            dados = json.loads(self.rfile.read(tamanho))
            resposta = self.server.classificador.receber(dados)
        except (ValueError, TypeError, OverflowError) as erro:
            self.responder(400, {"erro": str(erro)})
            return
        self.responder(200, resposta)


def testar(referencia, porta):
    esperadas = {}
    leituras = {}
    for exemplo in referencia["exemplos"]:
        janela = exemplo["leituras"]
        esperadas[pd.Timestamp(janela[-1]["timestamp"])] = exemplo["previsao_colab"]
        for leitura in janela:
            instante = pd.Timestamp(leitura["timestamp"])
            if instante in leituras and leituras[instante] != leitura:
                raise ValueError("A referência contém leituras conflitantes.")
            leituras[instante] = leitura
    if not esperadas:
        raise ValueError("A referência precisa conter exemplos do Colab.")

    for instante, leitura in sorted(leituras.items()):
        requisicao = Request(
            f"http://127.0.0.1:{porta}/leituras",
            data=json.dumps(leitura).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(requisicao, timeout=10) as resposta:
            resultado = json.load(resposta)
        if instante in esperadas:
            previsto = resultado.get("classe")
            if previsto != esperadas[instante]:
                raise ValueError(f"{instante}: Colab esperava {esperadas[instante]}, API devolveu {resultado}.")
            print(f"{instante.isoformat()} | previsão igual à do Colab: {previsto}")
    print(f"Fluxo HTTP conferido: {len(esperadas)} previsões coincidem com o Colab.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modo", choices=["servir", "testar"])
    parser.add_argument("arquivo", type=Path, help="Modelo joblib ou referência JSON.")
    parser.add_argument("--porta", type=int, default=8000)
    parser.add_argument("--monitorar-csv", type=Path, help="CSV do emissor para recuperar e consultar leituras recentes.")
    args = parser.parse_args()
    if args.modo == "testar":
        testar(json.loads(args.arquivo.read_text(encoding="utf-8")), args.porta)
        return

    artefato = joblib.load(args.arquivo)
    with HTTPServer(("127.0.0.1", args.porta), Handler) as servidor:
        servidor.classificador = Classificador(artefato)
        servidor.csv_monitorado = args.monitorar_csv
        if args.monitorar_csv is not None:
            linhas = ler_recentes(args.monitorar_csv, servidor.classificador.tamanho)
            for linha in linhas:
                servidor.classificador.receber({
                    "timestamp": linha["timestamp"],
                    **{campo: float(linha[campo]) for campo in GRANDEZAS},
                })
        print(f"Recebendo leituras em http://127.0.0.1:{args.porta}/leituras", flush=True)
        try:
            servidor.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
