# Classificação da condição de operação de um módulo fotovoltaico simulado

Este repositório reúne os notebooks e scripts do Trabalho de Conclusão de Curso de Pedro Nobrega Damacena, do curso de Engenharia de Computação da Universidade Federal do Ceará (UFC), Campus Quixadá.

O trabalho avalia um classificador para identificar três condições de operação de um módulo fotovoltaico simulado: operação normal, acúmulo de poeira (*soiling*) e sombreamento. Os modelos utilizam as leituras atuais de tensão, corrente e temperatura e atributos derivados do histórico de 15 e 30 minutos.

## Apêndice A — Notebooks dos experimentos

| Notebook | Finalidade |
| --- | --- |
| [Comparação dos classificadores](notebooks/TCC2_Comparacao_Classificadores_Com_Historico_Colab.ipynb) | Compara árvore de decisão, SVM-RBF e Random Forest com os mesmos atributos temporais e as mesmas partições cronológicas. |
| [Treinamento e avaliação do Random Forest](notebooks/TCC2_Random_Forest_Com_Historico_Colab.ipynb) | Gera os dados simulados, treina e avalia o Random Forest e exporta o modelo utilizado na Amazon EC2. |

Baixe o arquivo `.ipynb` desejado, carregue-o no Google Colab e execute as células na ordem apresentada. Os notebooks geram os próprios dados simulados com a biblioteca `pvlib`. Ao final da execução, baixe os arquivos de saída. O modelo `random_forest_com_historico.joblib` está dentro de `resultados_tcc_janelas.zip`, produzido pelo segundo notebook.

## Apêndice B — Execução na EC2 e ensaio de robustez

A pasta [`ec2/`](ec2/) reúne os scripts do classificador, do emissor de leituras e da avaliação das rodadas, além das unidades do `systemd` usadas no monitoramento contínuo.

### Reproduzir o ensaio

1. Em um ambiente Linux, copie os arquivos da pasta `ec2/` para `~/tcc`. O script do ensaio espera encontrar o ambiente Python em `~/tcc/.venv`.
2. Extraia o modelo exportado pelo notebook e coloque-o em `~/tcc/modelo/random_forest_com_historico.joblib`.
3. Com o Python e o `curl` disponíveis, prepare o ambiente e execute o ensaio. Para repetir as versões do ambiente de treinamento, consulte o arquivo `versoes.txt` exportado junto com o modelo.

```bash
cd ~/tcc
python3 -m venv .venv
.venv/bin/python -m pip install numpy pandas scipy joblib pvlib==0.16.1 scikit-learn==1.6.1
bash executar_robustez_ec2.sh
```

O script usa a porta local `8001` e executa três ciclos por cenário, com a semente 2026. O caminho da pasta de saída aparece no terminal. Nessa pasta, `nominal.csv` e `sombra_prolongada.csv` registram as rodadas, e `relatorio.txt` reúne as métricas e as matrizes de confusão.

### Monitoramento no Grafana

No monitoramento contínuo, o emissor envia as leituras ao classificador e registra os resultados em `monitoramento_continuo.csv`. O Grafana consulta as rotas `GET /previsoes` para exibir as leituras recentes e `GET /metricas/resumo` para mostrar os acertos, a quantidade de previsões válidas e a acurácia da janela monitorada. O ensaio de robustez pode ser executado de forma independente do painel.

## Escopo da avaliação

O desempenho foi avaliado com dados simulados. Os resultados descrevem as condições representadas pelo simulador. A validação com medições rotuladas de módulos reais permanece como trabalho futuro.
