import numpy as np

def funcao_degrau(valor):
    if valor >= 0:
        return 1
    else:
        return 0

def prever_aprovacao(frequencia, atividades, prova):
    entradas = np.array([
        1,
        frequencia / 100,
        atividades / 10,
        prova / 10
    ])

    pesos = np.array([-0.6, 0.30, 0.25, 0.45])
    
    soma = np.dot(entradas, pesos) 
    saida = funcao_degrau(soma)

    return saida, soma

alunos = [
    ["Ana", 90, 8, 9],
    ["Bruno", 75, 6, 5],
    ["Carlos", 60, 9, 8],
    ["Daniela", 85, 5, 4],
    ["Eduardo", 95, 9, 7]
]

for aluno in alunos:
    nome = aluno[0]
    frequencia = aluno[1]
    atividades = aluno[2]
    prova = aluno[3]

    resultado, soma = prever_aprovacao(
        frequencia,
        atividades,
        prova
    )

    situacao = "APROVADO" if resultado == 1 else "REPROVADO"

    print(
        f"{nome:10s} | "
        f"Frequência: {frequencia:3d}% | "
        f"Atividades: {atividades:2d} | "
        f"Prova: {prova:2d} | "
        f"Soma: {soma:.3f} | "
        f"{situacao}"
    )