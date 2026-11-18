"""Lê o fim do CSV de um emissor em execução."""

import csv


def ler_recentes(path, quantidade):
    try:
        arquivo = path.open("rb")
    except FileNotFoundError:
        return []

    with arquivo:
        cabecalho = arquivo.readline()
        inicio = arquivo.tell()
        arquivo.seek(0, 2)
        posicao = arquivo.tell()
        blocos = []
        quebras = 0
        while posicao > inicio and quebras < quantidade + 1:
            tamanho = min(65536, posicao - inicio)
            posicao -= tamanho
            arquivo.seek(posicao)
            bloco = arquivo.read(tamanho)
            blocos.append(bloco)
            quebras += bloco.count(b"\n")

    trecho = b"".join(reversed(blocos))
    if posicao > inicio:
        trecho = trecho.split(b"\n", 1)[-1]
    linhas = trecho.split(b"\n")[:-1]  # Lê somente linhas completas.
    nomes = next(csv.reader([cabecalho.decode("utf-8")]), [])
    return list(csv.DictReader(
        (linha.decode("utf-8") for linha in linhas[-quantidade:]), fieldnames=nomes
    ))
