 # Importa a função que carrega o conjunto de dados digits
from sklearn.datasets import load_digits

# Carrega o conjunto de imagens de dígitos manuscritos
digits = load_digits()

# Armazena as imagens no formato de vetores
# Cada imagem possui 64 valores
X = digits.data

# Armazena as classes corretas das imagens
# As classes variam de 0 a 9
y = digits.target

# Normalização dos dados
# Divide todos os valores de X pelo maior valor possível dos pixels
# Os valores passam do intervalo [0, 16] para o intervalo [0, 1]
X_normalizado = X / 16.0

# Divisão dos Dados
# Importa a função responsável por dividir os dados
from sklearn.model_selection import train_test_split

# Divide as entradas e as classes em conjuntos de treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    # Dados de entrada já normalizados
    X_normalizado,
    # Classes corretas das imagens
    y,
    # Reserva 20% dos exemplos para teste
    test_size=0.20,
    # Mantém a mesma divisão em todas as execuções
    random_state=42,
    # Preserva aproximadamente a proporção das classes
    # nos conjuntos de treinamento e teste
    stratify=y
)

# Importa o classificador baseado em uma rede neural MLP
from sklearn.neural_network import MLPClassifier

# Cria o modelo de rede neural
modelo = MLPClassifier(

    # Uma camada oculta contendo 64 neurônios
    hidden_layer_sizes=(64,),

    # Função de ativação utilizada na camada oculta
    activation="relu",

    # Algoritmo utilizado para atualizar pesos e biases
    solver="adam",

    # Número máximo de épocas de treinamento
    max_iter=350,

    # Fixa a aleatoriedade para reproduzir os resultados
    random_state=42
)

# Treina a rede usando as imagens e as classes de treinamento
modelo.fit(X_treino, y_treino)

# Verificação de quantas iterações foram usadas
print(modelo.n_iter_)

# Informa que o treinamento terminou
print("Treinamento concluído.")


# Importa o módulo de criação de gráficos
import matplotlib.pyplot as plt

# Desenha um gráfico utilizando os valores do erro
# armazenados durante o treinamento da rede neural
plt.plot(modelo.loss_curve_)

# Escreve o nome do eixo horizontal
plt.xlabel("Iteração")

# Escreve o nome do eixo vertical
plt.ylabel("Erro")

# Adiciona um título ao gráfico
plt.title("Curva de treinamento")

# Exibe uma grade para facilitar a leitura dos valores
plt.grid()

# Mostra o gráfico na tela
plt.show()

# ETAPA DE AVALIAÇÃO DO MODELO

# Importa a função que calcula a acurácia
from sklearn.metrics import accuracy_score

# Importa a função que gera um relatório detalhado
from sklearn.metrics import classification_report

# Faz previsões para as imagens do conjunto de teste
y_previsto = modelo.predict(X_teste)

# Compara as previsões com as respostas corretas
# e calcula a acurácia do modelo
acuracia = accuracy_score(y_teste, y_previsto)

# Mostra a acurácia com quatro casas decimais
print(f"Acurácia: {acuracia:.4f}")

# Apenas deixa uma linha em branco
print()

# Exibe um relatório completo contendo:
# - Precisão
# - Recall
# - F1-score
# - Quantidade de exemplos de cada classe
print(classification_report(y_teste, y_previsto))
    

# Importa a função que calcula a matriz de confusão
from sklearn.metrics import confusion_matrix

# Importa a classe responsável por desenhar a matriz de confusão
from sklearn.metrics import ConfusionMatrixDisplay

# Calcula a matriz de confusão comparando
# as classes reais (y_teste) com as classes previstas (y_previsto)
matriz = confusion_matrix(y_teste, y_previsto)

# Cria um objeto de visualização utilizando a matriz calculada
disp = ConfusionMatrixDisplay(confusion_matrix=matriz)

# Desenha a matriz de confusão utilizando uma escala de cores em tons de azul
disp.plot(cmap="Blues")

# Adiciona um título ao gráfico
plt.title("Matriz de confusão")

# Exibe a figura na tela
plt.show()


import numpy as np
import matplotlib.pyplot as plt


# Compara, posição por posição, as classes reais com as classes previstas.

indices_errados = np.where(y_teste != y_previsto)[0]

# Verifica se o modelo não cometeu nenhum erro.
if len(indices_errados) == 0:
    print("O modelo classificou corretamente todas as imagens de teste.")
else:
    # Define quantas imagens erradas serão mostradas.
    # Caso haja menos de 10 erros, mostra apenas a quantidade disponível.
    # Exemplo: min(10, 7)  retorna 7
    quantidade = min(10, len(indices_errados))
    # Cria uma figura com 2 linhas e 5 colunas = espaço para até 10 imagens.
    # largura da figura = 10 polegadas
    # altura da figura = 5 polegadas
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    # axes é originalmente uma matriz com 2 linhas e 5 colunas.
    # axes.ravel() transforma essa matriz em um vetor com 10 eixos.
    # [:quantidade] seleciona apenas a quantidade de espaços necessária.
    # enumerate(..) fornece posição atual e espaço onde a imagem será desenhada
    for posicao, eixo in enumerate(axes.ravel()[:quantidade]):
        # Obtém o índice de uma imagem que foi classificada incorretamente.
        # Na primeira repetição, posicao = 0 e  indice = 12
        indice = indices_errados[posicao]
        # Recupera a imagem correspondente no conjunto de teste.
        # Cada imagem está armazenada como um vetor com 64 valores.
        # reshape(8, 8) transforma esse vetor novamente
        # em uma matriz de 8 linhas e 8 colunas.
        imagem = X_teste[indice].reshape(8, 8)
        # Obtém a classe correta da imagem.
        classe_real = y_teste[indice]
        # Obtém a classe que foi prevista pelo modelo.
        classe_prevista = y_previsto[indice]
        # Exibe a imagem no espaço atual da figura, em tons de cinza.
        eixo.imshow(imagem, cmap="gray")
        # Adiciona um título acima da imagem.
        eixo.set_title(f"Real: {classe_real}\nPrevisto: {classe_prevista}")
        # Remove os números, marcas e linhas dos eixos.
        eixo.axis("off")

    # Seleciona os espaços da figura que não foram utilizados.
    # Este laço percorre apenas esses espaços restantes.
    for eixo in axes.ravel()[quantidade:]:
        # Esconde os eixos dos espaços que não receberam imagens.
        eixo.axis("off")

    # Ajusta automaticamente os espaços entre as imagens e os títulos.
    plt.tight_layout()

    # Exibe a figura completa na tela.
    plt.show()
    
