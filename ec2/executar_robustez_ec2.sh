#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/tcc"
python=.venv/bin/python
modelo=modelo/random_forest_com_historico.joblib
porta=8001
inicio=2025-05-01
semente=2026
ciclos=3

if [[ ! -x "$python" ]]; then
    printf 'Ambiente Python não encontrado em %s/.venv.\n' "$PWD" >&2
    exit 1
fi
for arquivo in "$modelo" teste_fluxo_ec2.py emissor_painel_ec2.py avaliar_rodadas_ec2.py; do
    if [[ ! -r "$arquivo" ]]; then
        printf 'Arquivo necessário não encontrado: %s/%s\n' "$PWD" "$arquivo" >&2
        exit 1
    fi
done
"$python" -c 'import joblib, numpy, pandas, pvlib, sklearn'

url="http://127.0.0.1:$porta/previsoes"
if curl --silent --max-time 1 --output /dev/null "$url"; then
    printf 'A porta %s já está em uso.\n' "$porta" >&2
    exit 1
fi
saida=$(mktemp -d "$PWD/robustez_$(date -u +%Y%m%d_%H%M%S)_XXXXXX")
printf 'Arquivos do ensaio: %s\n' "$saida"
pid_classificador=

parar_classificador() {
    if [[ -n "$pid_classificador" ]]; then
        if kill -0 "$pid_classificador" 2>/dev/null; then
            kill "$pid_classificador" 2>/dev/null || true
        fi
        wait "$pid_classificador" 2>/dev/null || true
        pid_classificador=
    fi
}

trap parar_classificador EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for cenario in nominal sombra_prolongada; do
    printf '\nIniciando cenário %s com %s ciclos.\n' "$cenario" "$ciclos"
    log="$saida/classificador_$cenario.log"
    "$python" -u teste_fluxo_ec2.py servir "$modelo" --porta "$porta" > "$log" 2>&1 &
    pid_classificador=$!
    for tentativa in {1..30}; do
        if ! kill -0 "$pid_classificador" 2>/dev/null; then
            break
        fi
        if grep -Fq "Recebendo leituras em http://127.0.0.1:$porta/leituras" "$log"; then
            break
        fi
        sleep 1
    done
    if ! kill -0 "$pid_classificador" 2>/dev/null ||
       ! grep -Fq "Recebendo leituras em http://127.0.0.1:$porta/leituras" "$log"; then
        cat "$log" >&2
        printf 'O classificador do cenário %s não ficou disponível.\n' "$cenario" >&2
        exit 1
    fi

    "$python" -u emissor_painel_ec2.py \
        --cenario "$cenario" --inicio "$inicio" --semente "$semente" \
        --ciclos "$ciclos" --intervalo 0 --porta "$porta" \
        --saida "$saida/$cenario.csv"
    parar_classificador
done

nominal="$saida/nominal.csv"
prolongado="$saida/sombra_prolongada.csv"
"$python" - "$nominal" "$prolongado" "$ciclos" <<'PY'
import sys
import pandas as pd

nominal, prolongado = (pd.read_csv(path) for path in sys.argv[1:3])
ciclos_esperados = list(range(1, int(sys.argv[3]) + 1))
for cenario, dados in (("nominal", nominal), ("sombra_prolongada", prolongado)):
    if dados["cenario"].unique().tolist() != [cenario]:
        raise SystemExit(f"Cenário inesperado no CSV de {cenario}.")
    if dados["ciclo"].unique().tolist() != ciclos_esperados:
        raise SystemExit(f"Ciclos ausentes no CSV de {cenario}.")
campos = ["semente", "ciclo", "timestamp", "poa_ceu_limpo"]
pd.testing.assert_frame_equal(nominal[campos], prolongado[campos])
print("Rodadas pareadas: mesmos instantes simulados, ciclos, semente e irradiância de céu limpo.")
PY

"$python" avaliar_rodadas_ec2.py "$nominal" "$prolongado" | tee "$saida/relatorio.txt"
printf '\nEnsaio concluído. Envie os dois CSVs e o relatorio.txt de %s.\n' "$saida"
