
![pytest](https://img.shields.io/badge/tests-passing-brightgreen?style=flat-square&logo=pytest)
# wine_prediction

## Estrutura Recomendada

- `src/` — Código-fonte principal do projeto
- `tests/` — Testes automatizados
- `dados/` — Dados de entrada (csv)
- `graficos/` — Saída de gráficos
- `previsoes/` — Saída de previsões
- `requirements.txt` — Dependências
- `.gitignore` — Arquivos/pastas ignorados
- `README.md` — Documentação

## Como rodar

Veja as instruções no início deste arquivo para preparar o ambiente, rodar testes e executar o pipeline principal.
# Projeto: Previsão da Qualidade de Vinhos com Machine Learning

## 🍇 Sobre o Projeto

Este projeto aplica **Machine Learning** para prever a **qualidade de vinhos tintos** com base em atributos físico-químicos como acidez, pH, álcool, entre outros. Ele segue um pipeline completo de ciência de dados, desde o tratamento de dados até a avaliação final do modelo.

---

## 🧪 Objetivo

Desenvolver e validar um modelo de regressão capaz de prever com boa precisão a nota de qualidade de vinhos tintos, a partir de dados brutos de composição química.

---

## 🧠 Etapas do Pipeline

1. **Leitura e análise dos dados (EDA):** visualizações, correlações e insights.
2. **Engenharia de atributos:** criação de atributos não lineares (polinomiais, cúbicos, logarítmicos e de interação).
3. **Normalização:** padronização via Z-score.
4. **Treinamento:** regressão linear otimizada via BFGS.
5. **Validação:** k-fold cross-validation.
6. **Avaliação externa:** uso de conjunto de teste separado.
7. **Armazenamento do modelo final:** salvando `theta`, `mean`, `std`.
8. **Salvamento das previsões:** gera um arquivo CSV com as previsões do modelo no conjunto de teste.

---

## 📁 Estrutura de Pastas

```bash
/
├── dados/
│   ├── winequality-red_treino.csv      # Dados de treino (usado no wine_prediction.py)
│   ├── winequality-red_teste.csv       # Dados de teste externo (usado no wine_prediction.py)
│   ├── winequality-red.csv             # Cópia do treino ou união, usado no correlational.py para EDA
│   ├── winequality-white.csv           # (Opcional) Dados de vinho branco
│   └── winequality.names               # Dicionário dos atributos e informações da fonte
│
├── graficos/
│   └── correlation_matrix_features_target.png  # ⬅️ IMAGEM GERADA automaticamente pelo correlational.py
│   └── dist_features.png               # ⬅️ IMAGEM GERADA automaticamente pelo model_tools.py (main)
│   └── real_vs_predito_R4.png          # ⬅️ IMAGEM GERADA automaticamente pelo model_tools.py (main)
│   └── theta_diff.png                  # ⬅️ IMAGEM GERADA automaticamente pelo model_tools.py (main)
│   └── pred_diff_R{R}.png              # ⬅️ IMAGEM(NS) GERADA(S) automaticamente pelo model_tools.py (main)
│
├── previsoes/                           # ⬅️ NOVA PASTA: Contém os resultados das predições em CSV
│   └── Y_pred_winequality_red_BFGS.csv # ⬅️ ARQUIVO GERADO pelo wine_prediction.py
│   └── Y_pred_BFGS_R{R}.csv            # ⬅️ ARQUIVOS GERADOS pelo model_tools.py (main)
│   └── Y_pred_GD_R{R}.csv              # ⬅️ ARQUIVOS GERADOS pelo model_tools.py (main)
│
├── src/
│   ├── model_tools.py                  # Funções para normalização, treino, validação e seleção de variáveis
│   ├── wine_prediction.py              # Script principal (treina e avalia o modelo final)
│   ├── correlational.py                # Gera matriz de correlação dos dados para EDA
│   ├── logger_config.py                # Configuração de logging
│   └── __init__.py                     # Torna src um pacote Python
├── tests/
│   ├── test_model_tools.py             # Testes automatizados
│   └── __init__.py                     # Torna tests um pacote Python
├── modelo_winequality_red.npz          # Modelo salvo (pode ser removido para zerar)
├── requirements.txt                    # Dependências do projeto
└── README.md                           # Este arquivo
```

---

## 📈 Resultados Obtidos

- **MSE (validação cruzada - 5 folds):** ~0.48
- **MSE (teste externo):** ~0.53

Esses valores mostram que o modelo generaliza bem e possui baixo erro quadrático.

---

## 📊 correlational.py: Análise Exploratória de Dados (EDA)

Este script gera a matriz de correlação entre os atributos (incluindo as features engenheiradas) e a qualidade:

- **Arquivo gerado:** `graficos/correlation_matrix_features_target.png`
- Usa `seaborn` para visualização.
- Mostra as relações mais fortes com a variável alvo.

**Observação Importante:** Este script tenta carregar `dados/winequality-red.csv`. Se este arquivo não existir, ele usará `dados/winequality-red_treino.csv` e emitirá um aviso. Recomenda-se ter `dados/winequality-red.csv` (pode ser uma cópia ou união dos dados de treino/teste) para a análise completa.

---

## ⚙️ model_tools.py: Funções Auxiliares e Pipeline de Seleção de Variáveis

Este módulo contém as principais funções reutilizadas no pipeline, além de um pipeline `main()` secundário focado na seleção de variáveis e comparação entre otimizadores (BFGS e GD).

| Função                      | Descrição                                                             |
|-----------------------------|----------------------------------------------------------------------|
| `load_csv(path)`            | Lê arquivos CSV no formato NumPy (para uso geral)                    |
| `create_nonlinear_features(X)` | Gera atributos polinomiais (x², x³, log(x), x1x2...)            |
| `normalize(X)`              | Aplica Z-score e retorna mean e std para reuso                       |
| `add_bias(X)`               | Adiciona termo de intercepto (coluna de 1s)                          |
| `fit_bfgs(Xb, y)`           | Ajusta modelo via minimização BFGS (scipy)                           |
| `fit_gd(Xb, y, ...)`        | Ajusta modelo via Gradient Descent (implementação própria)            |
| `kfold_mse(X, y, k, fit)`   | Realiza validação cruzada (k-fold)                                   |
| `best_subset_by_R(...)`     | Seleciona o melhor subconjunto de R features via CV                  |

**Exportar para as Planilhas:**  
O `main()` dentro de `model_tools.py` pode ser executado para explorar a seleção de variáveis e gerar gráficos adicionais (`dist_features.png`, `theta_diff.png`, `pred_diff_R{R}.png`, `real_vs_predito_R4.png`) e salvar previsões para cada subconjunto (`Y_pred_BFGS_R{R}.csv`, `Y_pred_GD_R{R}.csv`).

---

## 🧪 wine_prediction.py: Pipeline de Treinamento e Avaliação Principal

Este é o script principal para treinar o modelo de regressão com todas as features engenheiradas e avaliar seu desempenho no conjunto de teste externo. Ele é responsável por:

- Carregar os dados de treino e teste.
- Gerar os atributos polinomiais (e outros).
- Normalizar os dados usando as estatísticas do conjunto de treino.
- Treinar o modelo com BFGS.
- Realizar 5-fold CV.
- Avaliar o modelo no conjunto de teste externo.
- Salvar o modelo como `modelo_winequality_red.npz`.
- Salvar as previsões do modelo treinado no arquivo `Y_pred_winequality_red_BFGS.csv` dentro da pasta `previsoes/`.

---

## 🛠️ Como Executar

Para rodar o projeto, siga estes passos no seu terminal:

**1. Clone o repositório:**
```bash
git clone https://github.com/RaFeltrim/wine_prediction.git
cd wine_prediction
```

**2. Crie e ative um ambiente virtual (recomendado):**
```bash
python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate
```

**3. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**4. Prepare o arquivo de dados para a análise de correlação (se winequality-red.csv não existir):**  
Para evitar `FileNotFoundError` no correlational.py, você pode copiar o arquivo de treino:
```bash
copy dados\winequality-red_treino.csv dados\winequality-red.csv  # No Windows
# ou
cp dados/winequality-red_treino.csv dados/winequality-red.csv    # No macOS/Linux
```

**5. Execute os scripts principais:**

- Para gerar a matriz de correlação (EDA):
  ```bash
  python -m src.correlational
  ```
  *(A imagem será salva em graficos/)*

- Para treinar o modelo principal, avaliar e salvar suas previsões:
  ```bash
  python -m src.wine_prediction
  ```
  *(Os parâmetros do modelo serão salvos em modelo_winequality_red.npz e as previsões em previsoes/)*

- Opcional: Para executar o pipeline de seleção de variáveis e gerar gráficos e previsões adicionais (para cada R):
  ```bash
  python -m src.model_tools
  ```
  *(Gráficos em graficos/ e previsões detalhadas em previsoes/Y_pred_BFGS_R{R}.csv e Y_pred_GD_R{R}.csv)*

---

## 📌 Requisitos

- Python 3.8+
- Bibliotecas:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scipy

✅ Todas estão listadas em `requirements.txt`.

---

## 📚 Base de Dados

- **Fonte:** UCI Machine Learning Repository, conforme descrito em [Cortez et al., 2009].
- **Citação:**  
  Se for usar este banco de dados, por favor, inclua esta citação:

  > P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.  
  > Modeling wine preferences by data mining from physicochemical properties.  
  > In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.  
  > Disponível em: [@Elsevier](http://dx.doi.org/10.1016/j.dss.2009.05.016)

- **Notas de qualidade:** 0 a 10 (valores reais entre 3 e 8)
- **Atributos:** 11 colunas físico-químicas de entrada + 1 coluna de saída (`quality`)
- **Valores Ausentes:** Nenhuma variável possui valores ausentes.
- **Correlação entre Atributos:** Vários atributos podem ser correlacionados, o que torna a engenharia e seleção de features uma etapa importante.

---

## ✨ Melhorias Futuras

- Incluir modelos mais robustos (Random Forest, XGBoost, SVR)
- Adicionar regularização (Ridge, Lasso)
- Explicabilidade com SHAP/Permutation Importance para entender a contribuição de cada feature
- Interface Web para uso interativo
- Refinar a seleção de variáveis (`best_subset_by_R`) para ser mais eficiente e talvez dinâmica, não dependendo de índices hardcoded
- Considerar otimização de hiperparâmetros

---

## 🎯 Motivação Pessoal

> “Escolhi esse projeto por causa do meu amor por vinhos e pela diversidade que eles representam. Queria unir essa paixão com meu interesse por tecnologia, criando uma IA capaz de avaliar produtos sensoriais com base em dados.”  
> — Rafael Feltrim

---

## 🙋‍♂️ Autores

- **Rafael Feltrim** — [@RaFeltrim](https://github.com/RaFeltrim)
- **Gustavo** — Menção Honrosa 

---

## 💡 Licença

Este projeto está sob a Licença MIT. Consulte o arquivo `LICENSE`.

> "A melhor forma de aprender é ensinar. Compartilhe este projeto se ele te ajudou!"