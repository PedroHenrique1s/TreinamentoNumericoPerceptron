# Importa o módulo de criação de gráficos
import matplotlib.pyplot as plt

# Importa a função que carrega o conjunto de imagens de dígitos
from sklearn.datasets import load_digits

# Carrega o conjunto de dados
digits = load_digits()

#Exploração dos dados
print(f" Images: {digits.images.shape}")
print(f" Images: {digits.images}")
print(f" Target: {digits.target.shape}")
print(f" Target: {digits.target}")
print(f" Data: {digits.data.shape}")
print(f" Data: {digits.data}")
print(digits.keys())
print(digits.DESCR)
print("\n\n Número de exemplos no conjunto de dados: ", len(digits.data))
print("\n Conjunto de dados carregado com SUCESSO!")    


# Cria uma figura com 2 linhas e 5 colunas
# Ao todo, serão exibidas 10 imagens
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

# Percorre os 10 espaços da figura
for indice, eixo in enumerate(axes.ravel()):

    # Exibe a imagem correspondente ao índice atual
    eixo.imshow(digits.images[indice], cmap="gray")

    # Mostra a classe correta da imagem no título
    eixo.set_title(f"Classe: {digits.target[indice]}")

    # Oculta os eixos e as marcações numéricas
    eixo.axis("off")

# Ajusta o espaçamento entre as imagens
plt.tight_layout()

# Exibe a figura
plt.show()

