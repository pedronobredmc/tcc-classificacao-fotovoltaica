"""Gera e envia leituras sintéticas em ordem cronológica para a API na EC2."""

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from leituras_csv import ler_recentes

import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from pvlib.pvsystem import retrieve_sam, calcparams_cec, singlediode
from pvlib.temperature import faiman


FREQ = "5min"
TZ = "America/Fortaleza"
TILT, AZIMUTH = 20, 0
DURACOES_SOMBRA = {"nominal": (5, 40), "sombra_prolongada": (45, 60)}
U0, U1 = 25.0, 6.84


def carregar_modulo():
    cec_modules = retrieve_sam("CECMod")
    MODULO_CEC = "Canadian_Solar_Inc__CS6X_300M"
    if MODULO_CEC not in cec_modules.columns:
        MODULO_CEC = [c for c in cec_modules.columns if "Canadian_Solar" in c][0]

    return cec_modules[MODULO_CEC]


def simular_ciclo(inicio, rng, site, modulo, duracao_sombra):
    """Gera um ciclo completo, do início do acúmulo à próxima limpeza."""
    dias = int(rng.integers(6, 15))
    times = pd.date_range(inicio, periods=dias * 24 * 12, freq=FREQ)
    n = len(times)

    clearsky = site.get_clearsky(times, model="ineichen")

    solpos = site.get_solarposition(times)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=TILT,
        surface_azimuth=AZIMUTH,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=clearsky["dni"],
        ghi=clearsky["ghi"],
        dhi=clearsky["dhi"],
    )
    poa_global = poa["poa_global"].clip(lower=0)

    ruido_nuvem = np.clip(rng.normal(loc=1.0, scale=0.05, size=len(times)), 0.6, 1.05)
    poa_global = poa_global * ruido_nuvem

    hora_do_dia = times.hour + times.minute / 60
    temp_base = 26 + 6 * np.sin((hora_do_dia - 9) / 24 * 2 * np.pi)
    temp_ambiente = temp_base + rng.normal(0, 0.8, size=len(times))

    vento = np.clip(rng.normal(2.5, 1.0, size=len(times)), 0.2, None)  # m/s

    perda_maxima = rng.uniform(0.15, 0.40)
    sujeira_fraction = perda_maxima * np.arange(n) / n

    sujeira_fraction_label = sujeira_fraction.copy()   # tendência "limpa", usada só para o rótulo
    sujeira_fraction = np.clip(sujeira_fraction + rng.normal(0, 0.008, size=n), 0, None)  # com ruído, usada na física
    LIMIAR_SOILING_VISIVEL = 0.12   # só rotula como falha quando a perda já é inequívoca (>12%), bem acima do ruído de nuvem

    sombra_fator = np.ones(n)
    sombra_ativo = np.zeros(n, dtype=bool)

    # Isola o ruído da sombra para manter os outros sorteios iguais entre cenários.
    rng_ruido_sombra = np.random.default_rng(rng.integers(0, 2**32))
    media_eventos_por_dia = 2.5
    n_eventos = rng.poisson(media_eventos_por_dia * dias)

    for _ in range(n_eventos):
        dia = rng.integers(0, dias)
        # horário do evento: bimodal, concentrado perto do nascer/pôr do sol (baixo ângulo solar)
        if rng.random() < 0.5:
            hora = np.clip(rng.normal(6.5, 1.0), 5.0, 9.5)
        else:
            hora = np.clip(rng.normal(16.5, 1.2), 13.5, 18.5)

        inicio = times.normalize()[0] + pd.Timedelta(days=int(dia), hours=float(hora))
        duracao_evento = pd.Timedelta(minutes=rng.uniform(*duracao_sombra))
        fim = inicio + duracao_evento

        fator_base = rng.uniform(0.30, 0.65)
        mask_evento = (times >= inicio) & (times < fim)
        if not mask_evento.any():
            continue

        ruido_sombra = rng_ruido_sombra.normal(1.0, 0.08, size=mask_evento.sum())
        sombra_fator[mask_evento] = np.minimum(sombra_fator[mask_evento], fator_base * ruido_sombra)
        sombra_ativo[mask_evento] = True

    sombra_fator = np.clip(sombra_fator, 0, 1)

    irradiancia_efetiva = poa_global.values * (1 - sujeira_fraction) * sombra_fator
    irradiancia_efetiva = np.clip(irradiancia_efetiva, 0, None)

    estados = np.full(n, "normal", dtype=object)
    estados[sujeira_fraction_label > LIMIAR_SOILING_VISIVEL] = "soiling"
    estados[sombra_ativo] = "sombreamento"   # evento de sombreamento tem prioridade sobre o rótulo de soiling de fundo


    temp_celula = faiman(
        poa_global=irradiancia_efetiva,
        temp_air=temp_ambiente.values,
        wind_speed=vento,
        u0=U0, u1=U1,
    )


    # Resolve o modelo elétrico apenas onde há luz suficiente.
    sem_luz = irradiancia_efetiva < 5
    params = calcparams_cec(
        effective_irradiance=irradiancia_efetiva[~sem_luz],
        temp_cell=temp_celula[~sem_luz],
        alpha_sc=modulo["alpha_sc"],
        a_ref=modulo["a_ref"],
        I_L_ref=modulo["I_L_ref"],
        I_o_ref=modulo["I_o_ref"],
        R_sh_ref=modulo["R_sh_ref"],
        R_s=modulo["R_s"],
        Adjust=modulo["Adjust"],
    )

    iv = singlediode(*params, method="lambertw")
    tensao = np.zeros(n)
    corrente = np.zeros(n)
    tensao[~sem_luz] = iv["v_mp"]
    corrente[~sem_luz] = iv["i_mp"]
    estados[sem_luz] = "normal"

    tensao_sensor = np.clip(tensao + rng.normal(0, 0.15, size=n), 0, None)
    corrente_sensor = np.clip(corrente + rng.normal(0, 0.05, size=n), 0, None)
    temp_celula_sensor = temp_celula + rng.normal(0, 0.3, size=n)

    return pd.DataFrame({
        "timestamp": times,
        "tensao": tensao_sensor,
        "corrente": corrente_sensor,
        "temp_celula": temp_celula_sensor,
        "estado": estados,
        "poa_ceu_limpo": poa["poa_global"].clip(lower=0).to_numpy(),
    })


def enviar(leitura, porta):
    requisicao = Request(
        f"http://127.0.0.1:{porta}/leituras",
        data=json.dumps(leitura).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(requisicao, timeout=15) as resposta:
        resultado = json.load(resposta)
    if resultado.get("timestamp") != leitura["timestamp"]:
        raise ValueError("O serviço retornou um horário diferente do enviado.")
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inicio", default="2025-05-01", help="Data do início da série (AAAA-MM-DD).")
    parser.add_argument("--semente", type=int, default=2026)
    parser.add_argument("--intervalo", type=float, default=1, help="Segundos reais entre envios.")
    parser.add_argument("--ciclos", type=int, help="Ciclos a enviar. Sem este argumento, gera continuamente.")
    parser.add_argument("--cenario", choices=DURACOES_SOMBRA, default="nominal")
    parser.add_argument("--porta", type=int, default=8000)
    parser.add_argument("--saida", type=Path, required=True, help="CSV para registrar a rodada.")
    parser.add_argument("--retomar", action="store_true", help="Continua a partir da última leitura registrada no CSV.")
    args = parser.parse_args()
    if not np.isfinite(args.intervalo) or args.intervalo < 0:
        parser.error("--intervalo precisa ser um número finito maior ou igual a zero.")
    if args.ciclos is not None and args.ciclos < 1:
        parser.error("--ciclos precisa ser maior ou igual a um.")

    inicio = pd.Timestamp(args.inicio, tz=TZ)
    if inicio != inicio.normalize():
        parser.error("--inicio precisa ser uma data à meia-noite.")
    rng = np.random.default_rng(args.semente)
    site = Location(-4.97, -39.02, tz=TZ, altitude=200)
    modulo = carregar_modulo()
    duracao_sombra = DURACOES_SOMBRA[args.cenario]

    campos = [
        "cenario", "semente", "ciclo", "timestamp", "tensao", "corrente",
        "temp_celula", "poa_ceu_limpo", "estado", "status", "previsao",
    ]
    ciclo = 1
    try:
        existe = args.retomar and args.saida.exists() and args.saida.stat().st_size > 0
        ultimos = ler_recentes(args.saida, 1) if existe else []
        ultimo_ciclo, ultimo_instante = 0, None
        if existe:
            with args.saida.open(encoding="utf-8", newline="") as anterior:
                leitor = csv.DictReader(anterior)
                primeiro = next(leitor, None)
                if leitor.fieldnames != campos:
                    raise ValueError("O cabeçalho do CSV não corresponde ao emissor atual.")
            with args.saida.open("rb") as anterior:
                anterior.seek(-1, 2)
                if anterior.read(1) != b"\n":
                    raise ValueError("A última linha do CSV está incompleta. Preserve o arquivo antes de corrigir a linha.")
            if primeiro:
                if (primeiro["cenario"] != args.cenario or
                        int(primeiro["semente"]) != args.semente or
                        pd.Timestamp(primeiro["timestamp"]) != inicio):
                    raise ValueError("Use a mesma data inicial, semente e cenário do CSV para retomar.")
                ultimo_ciclo = int(ultimos[0]["ciclo"])
                ultimo_instante = pd.Timestamp(ultimos[0]["timestamp"])

        modo = "a" if args.retomar else "x"
        with args.saida.open(modo, newline="", encoding="utf-8") as arquivo:
            registro = csv.writer(arquivo)
            if not existe:
                registro.writerow(campos)
                arquivo.flush()
            # Reproduz os sorteios anteriores para preservar o estado do gerador.
            for _ in range(1, ultimo_ciclo):
                dados = simular_ciclo(inicio, rng, site, modulo, duracao_sombra)
                inicio = dados["timestamp"].iloc[-1] + pd.Timedelta(FREQ)
            ciclo = max(1, ultimo_ciclo)
            while args.ciclos is None or ciclo <= args.ciclos:
                dados = simular_ciclo(inicio, rng, site, modulo, duracao_sombra)
                pendentes = dados
                if ciclo == ultimo_ciclo:
                    if not dados["timestamp"].eq(ultimo_instante).any():
                        raise ValueError("O último horário registrado não pertence ao ciclo reproduzido.")
                    pendentes = dados.loc[dados["timestamp"] > ultimo_instante]
                print(f"Ciclo {ciclo} | {inicio.date()} | {len(pendentes)} leituras a enviar", flush=True)
                for linha in pendentes.itertuples(index=False):
                    leitura = {
                        "timestamp": linha.timestamp.isoformat(),
                        "tensao": linha.tensao,
                        "corrente": linha.corrente,
                        "temp_celula": linha.temp_celula,
                    }
                    resultado = enviar(leitura, args.porta)
                    registro.writerow([
                        args.cenario, args.semente, ciclo, leitura["timestamp"],
                        leitura["tensao"], leitura["corrente"], leitura["temp_celula"],
                        linha.poa_ceu_limpo, linha.estado, resultado["status"],
                        resultado.get("classe", ""),
                    ])
                    arquivo.flush()
                    time.sleep(args.intervalo)
                inicio = dados["timestamp"].iloc[-1] + pd.Timedelta(FREQ)
                ciclo += 1
    except KeyboardInterrupt:
        print(f"\nEmissor interrompido. Leituras registradas em {args.saida}.")
    except URLError as erro:
        raise SystemExit(f"Falha no envio: {erro}. Registros anteriores em {args.saida}.") from erro
    else:
        print(f"Rodada concluída. Leituras registradas em {args.saida}.")


if __name__ == "__main__":
    main()
