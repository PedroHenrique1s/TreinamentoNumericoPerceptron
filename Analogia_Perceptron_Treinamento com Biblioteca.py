# Importa a biblioteca NumPy
import numpy as np

# Importa o algoritmo Perceptron
from sklearn.linear_model import Perceptron

# Importa o padronizador de dados
from sklearn.preprocessing import StandardScaler

# Importa a função para criar um pipeline
from sklearn.pipeline import make_pipeline

# Cria a matriz de características dos alunos
# Cada linha representa um aluno
# Cada coluna representa uma característica
X = np.array([
    [90, 8, 9],
    [85, 7, 8],
    [80, 8, 7],
    [75, 6, 6],
    [95, 9, 8],
    [60, 4, 3],
    [65, 5, 4],
    [70, 4, 5],
    [55, 6, 4],
    [68, 5, 5]
])

# Cria o vetor com as classes corretas
# 1 representa aprovado
# 0 representa reprovado
y = np.array([
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0
])

# Cria um pipeline com duas etapas:
# 1. Padronização dos dados
# 2. Classificação utilizando o Perceptron
modelo = make_pipeline(
    StandardScaler(),
    Perceptron(
        max_iter=1000,
        random_state=42
    )
)

# Treina o modelo utilizando os dados e as classes corretas
modelo.fit(X, y)

# Visualizando os pesos aprendidos
perceptron = modelo.named_steps["perceptron"]

print("Pesos aprendidos:")
print(perceptron.coef_)

print("Bias aprendido:")
print(perceptron.intercept_)

# Cria os dados de um novo aluno
# A matriz possui uma linha e três características
novo_aluno = np.array([
    [82, 7, 6]
])

# Realiza a previsão para o novo aluno
previsao = modelo.predict(novo_aluno)

# Verifica a classe prevista
if previsao[0] == 1:
    print("Resultado previsto: APROVADO")
else:
    print("Resultado previsto: REPROVADO")
   
print()
   
# Entrada dos dados do novo aluno
print("\n INFORMAÇÃO DOS DADOS DE NOVO ALUNO ")

frequencia = float(input("Frequência do aluno: "))
atividades = float(input("Desempenho nas atividades/trabalho: "))
nota = float(input("Nota de prova do aluno: "))

# Criação da matriz com os dados do novo aluno
novo_aluno = np.array([
    [frequencia, atividades, nota]
])


# Realização da previsão
previsao = modelo.predict(novo_aluno)


# Exibição do resultado
if previsao[0] == 1:
    print("Resultado previsto: APROVADO")
else:
    print("Resultado previsto: REPROVADO")


# Laço para permitir vários testes
while True:

    print("\n===================================")
    print("Previsão de aprovação de aluno")
    print("===================================")

    # Entrada dos dados
    frequencia = float(input("Frequência do aluno: "))
    atividades = float(input("Desempenho nas atividades/trabalho: "))
    nota = float(input("Nota de prova do aluno: "))

    # Criação da matriz do novo aluno
    novo_aluno = np.array([
        [frequencia, atividades, nota]
    ])

    # Realização da previsão
    previsao = modelo.predict(novo_aluno)

    # Exibição do resultado
    if previsao[0] == 1:
        print("\nResultado previsto: APROVADO")
    else:
        print("\nResultado previsto: REPROVADO")

    # Pergunta se deseja realizar outro teste
    resposta = input("\nDeseja testar outro aluno? (S/N): ").upper()

    if resposta != "S":
        print("\nPrograma encerrado.")
        break

