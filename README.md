# Redes Neurais na Prática: do Perceptron ao Classificador de Imagens com Python

Mini curso prático que apresenta os fundamentos de redes neurais de forma progressiva.
Começamos com um **Perceptron** simples de pesos fixos, passamos pelo **treinamento
automático de pesos** com Scikit-learn e chegamos a um **classificador de imagens** de
dígitos manuscritos usando uma rede neural (MLP), atingindo uma **acurácia de 0.98 no
treinamento**.

## Objetivo

Entender, na prática, como uma rede neural funciona por dentro:

- Como as entradas são combinadas com pesos e um bias.
- Qual o papel da função de ativação.
- A diferença entre **definir pesos manualmente** e **deixar o modelo aprender** os pesos.
- Como aplicar esses conceitos a um problema real de **classificação de imagens**.

## Requisitos

```bash
pip install numpy scikit-learn matplotlib
```

## Estrutura do curso (arquivos Python)

### 1. `Analogia_Perceptron_PesoFixo.py` — O Perceptron na mão

Primeiro contato com o neurônio artificial. Usamos uma analogia simples: **aprovar ou
reprovar um aluno** a partir de três características (frequência, atividades e prova).

O que fizemos neste arquivo:

- Implementamos a **função de ativação degrau** (`funcao_degrau`), que devolve `1`
  (aprovado) ou `0` (reprovado).
- **Normalizamos as entradas** para uma escala comparável (dividindo por 100 e por 10).
- Definimos os **pesos manualmente**, incluindo o **bias**, atribuindo a importância de
  cada característica.
- Calculamos a **soma ponderada** (`np.dot`) e aplicamos a função de ativação para obter
  o resultado.

**Ideia central:** mostrar que o "cérebro" de um neurônio é apenas uma soma ponderada
seguida de uma função de ativação.

### 2. `Analogia_Perceptron_PesoFixoAlunos.py` — Aplicando o Perceptron a vários casos

Reaproveitamos o mesmo Perceptron de pesos fixos, agora avaliando **uma turma inteira**.

O que fizemos neste arquivo:

- Reescrevemos a lógica na função `prever_aprovacao`, que retorna a **saída** e a **soma
  ponderada**.
- Criamos uma **lista de alunos** com seus dados.
- Percorremos a turma em um **laço**, exibindo para cada aluno a frequência, as
  atividades, a prova, a soma ponderada e a situação final (APROVADO / REPROVADO).

**Ideia central:** aplicar o mesmo conjunto de pesos a vários exemplos e comparar os
resultados.

### 3. `Analogia_Perceptron_Treinamento com Biblioteca.py` — Deixando o modelo aprender os pesos

Aqui damos o salto conceitual: em vez de escolher os pesos na mão, o **modelo aprende os
pesos sozinho** a partir dos dados.

O que fizemos neste arquivo:

- Montamos a **matriz de características** `X` (dados dos alunos) e o **vetor de classes**
  `y` (aprovado/reprovado).
- Criamos um **pipeline** com `StandardScaler` (padronização dos dados) + `Perceptron`
  do Scikit-learn.
- **Treinamos o modelo** com `modelo.fit(X, y)`.
- **Inspecionamos os pesos e o bias aprendidos** (`coef_` e `intercept_`), comparando com
  a versão de pesos fixos.
- Fizemos **previsões para novos alunos**, inclusive com **entrada interativa** e um laço
  que permite testar vários casos.

**Ideia central:** a transição de pesos definidos manualmente para pesos **aprendidos**
automaticamente pelo algoritmo.

### 4. `CriaDataset.py` — Conhecendo o conjunto de imagens

Preparação para o problema de imagens: exploramos o dataset de **dígitos manuscritos**
(`load_digits`) do Scikit-learn.

O que fizemos neste arquivo:

- Carregamos o conjunto de dados e **exploramos sua estrutura** (`images`, `data`,
  `target`, dimensões e descrição).
- Verificamos que cada imagem é uma matriz **8×8** (64 valores) com classes de **0 a 9**.
- Usamos o **Matplotlib** para exibir uma amostra de **10 imagens** com suas respectivas
  classes.

**Ideia central:** entender os dados **antes** de treinar — o que são as imagens, como
estão representadas e quais são os rótulos.

### 5. `TreinamentoDaImagem.py` — Classificador de imagens com rede neural (MLP)

O ponto alto do curso: treinamos uma **rede neural** para reconhecer dígitos manuscritos.

O que fizemos neste arquivo:

- **Preparamos os dados:** transformamos as imagens em vetores (`X`), separamos as classes
  (`y`) e **normalizamos** os pixels para o intervalo `[0, 1]`.
- **Dividimos os dados** em treino e teste com `train_test_split` (20% para teste,
  `stratify=y` para preservar a proporção das classes).
- Criamos e treinamos um **`MLPClassifier`** (rede neural com uma camada oculta de 64
  neurônios, ativação `relu`, solver `adam`).
- Plotamos a **curva de treinamento** (erro por iteração) para acompanhar o aprendizado.
- **Avaliamos o modelo** com `accuracy_score`, `classification_report` (precisão, recall,
  F1-score) e **matriz de confusão**.
- **Visualizamos os erros:** exibimos as imagens que o modelo classificou incorretamente,
  comparando classe real x classe prevista.

**Resultado:** obtivemos uma **acurácia de 0.98 no treinamento**, mostrando que a rede
neural aprendeu a reconhecer os dígitos com alta precisão.

## Conclusão

Ao longo do curso, saímos de um único neurônio com pesos definidos na mão e chegamos a uma
rede neural capaz de **classificar imagens** com **0.98 de acurácia**. O percurso deixa
claro cada peça do quebra-cabeça: entradas, pesos, bias, função de ativação, treinamento,
avaliação e visualização dos resultados.

## Como executar

```bash
python Analogia_Perceptron_PesoFixo.py
python Analogia_Perceptron_PesoFixoAlunos.py
python "Analogia_Perceptron_Treinamento com Biblioteca.py"
python CriaDataset.py
python TreinamentoDaImagem.py
```
