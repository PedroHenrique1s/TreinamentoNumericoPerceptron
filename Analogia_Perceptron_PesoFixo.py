import numpy as np

def funcao_degrau(valor):
    if valor >= 0:
        return 1
    else:
        return 0

def avaliar_aluno(frequencia, atividades, prova):
    # Normalização das entradas
    entradas = np.array([
        1,                 # entrada do bias
        frequencia / 100,  # frequência
        atividades / 10,   # atividades
        prova / 10         # prova
    ])
    # Pesos definidos manualmente
    pesos = np.array([
       -0.60,  # peso do bias
        0.30,  # importância da frequência
        0.25,  # importância das atividades
        0.45   # importância da prova
    ])
    # Soma ponderada
    soma = np.dot(entradas, pesos) 
    # Aplicação da função de ativação
    saida = funcao_degrau(soma)
    print("Entradas normalizadas:", entradas)
    print("Pesos:", pesos)
    print("Soma ponderada:", round(soma, 3))
    if saida == 1:
        print("Resultado: APROVADO")
    else:
        print("Resultado: REPROVADO")

avaliar_aluno(
    frequencia=80,
    atividades=7,
    prova=6
)
