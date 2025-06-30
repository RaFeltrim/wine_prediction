# Projeto: Previsão da Qualidade de Vinhos com Machine Learning

![Ilustração de vinho com gráficos](https://i.imgur.com/0Pz0O2Q.png) <!-- Substituir por imagem do projeto -->

## 🍇 Sobre o Projeto

Este projeto usa algoritmos de regressão para prever a **qualidade de vinhos tintos** com base em seus atributos físico-químicos, como acidez, álcool, pH, entre outros. Ele segue todo o pipeline de um projeto de Machine Learning:

1. Leitura e análise de dados (EDA)
2. Engenharia de atributos (expansão polinomial)
3. Normalização
4. Treinamento com Regressão Linear (BFGS)
5. Validação cruzada (k-fold)
6. Avaliação com dados externos
7. Armazenamento do modelo final

## 🔍 Documentação Técnica Detalhada

### 📁 Estrutura de Pastas

```
/
├── dados/                         # Arquivos CSV com dados de treino e teste
│   ├── winequality-red_treino.csv
│   ├── winequality-red_teste.csv
│   └── winequality-white.csv
├── graficos/                      # Gráficos salvos durante o EDA
│   ├── hist_red.png
│   ├── hist_white.png
│   ├── scatter_red.png
│   └── heatmap_red.png
├── model_tools.py                 # Funções auxiliares para modelagem
├── wine_prediction.py             # Pipeline principal de treinamento e avaliação
├── correlational.py               # Geração de gráficos exploratórios
├── modelo_winequality_red.npz    # Modelo salvo com parâmetros aprendidos
└── req.txt                        # Lista de dependências
```

### 📊 Arquivo: `correlational.py`

* Gera os principais gráficos de EDA:

  * Histograma da distribuição de notas (tinto e branco)
  * Dispersão entre teor alcoólico e qualidade
  * Mapa de calor da matriz de correlação

> **Objetivo:** Compreender a distribuição e relações entre variáveis.

### ⚙️ Arquivo: `model_tools.py`

Contém as principais funções utilizadas no pipeline de Machine Learning:

* `load_csv(path)`: lê os dados em formato NumPy.
* `create_nonlinear_features(X)`: adiciona features polinomiais x² e x³.
* `normalize(X_train, X_test)`: normaliza os dados pelo z-score.
* `add_bias(X)`: adiciona uma coluna de 1s (intercepto).
* `fit_bfgs(Xb, y)`: ajusta o modelo de regressão com BFGS.
* `kfold_mse(Xb, y, k)`: executa validação cruzada k-fold e retorna erro médio.

### 🧠 Arquivo: `wine_prediction.py`

Pipeline principal:

1. Carrega os dados de treino e teste externo.
2. Aplica engenharia de features (x, x², x³).
3. Normaliza os dados.
4. Adiciona intercepto.
5. Treina o modelo com BFGS.
6. Calcula e imprime MSE de treino, validação e teste externo.
7. Salva os parâmetros do modelo (`theta`, `mean`, `std`) em `.npz`.

## 📈 Resultados Obtidos

* **MSE no treino:** 0.52
* **MSE com validação cruzada (5-fold):** 0.58
* **MSE no teste externo:** 0.60

Esses resultados indicam que o modelo é capaz de generalizar bem para novos dados.

## 🎯 Motivação Pessoal

> “Escolhi esse projeto por causa do meu amor por vinhos e pela diversidade que eles representam. Queria unir essa paixão com meu interesse por tecnologia, criando uma IA capaz de avaliar produtos sensoriais com base em dados.” — Rafael Feltrim

## 🛠️ Como Executar

### 1. Clone o repositório:

```bash
git clone https://github.com/seuusuario/wine-quality-ml.git
cd wine-quality-ml
```

### 2. Instale as dependências:

```bash
pip install -r req.txt
```

### 3. Execute os scripts:

```bash
python correlational.py     # Gera os gráficos
python wine_prediction.py   # Treina e avalia o modelo
```

## 📌 Requisitos

* Python 3.8+
* Bibliotecas:

  * numpy
  * pandas
  * matplotlib
  * scipy

## 📚 Base de Dados

* Disponível em: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

## ✨ Próximas Melhorias

* Implementar outros algoritmos como Random Forest, SVR, XGBoost.
* Aplicar regularização (Ridge/Lasso).
* Usar explicabilidade de modelo (ex: SHAP).
* Criar uma interface interativa para usuários finais.

## 🙋‍♂️ Autores

* Rafael Feltrim — [@RaFeltrim](https://github.com/RaFeltrim)
* Gustavo — Didática e apresentação do projeto

## 💡 Licença

Este projeto está sob a Licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

> *"A melhor forma de aprender é ensinar. Compartilhe este projeto se ele te ajudou!"*