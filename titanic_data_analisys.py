import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


TITANIC_DATASET_URL = "https://raw.githubusercontent.com/stevillis/titanic-data-viz/master/titanic.csv"


def preprocessing(df):
    # Remove null values from Age column
    df = df[df["Age"].notna()]

    # Filter columns of interest
    df = df.loc[:, ["PassengerId", "Survived", "Pclass", "Sex", "Age", "Embarked"]]

    # Convert Sex column from object to numeric
    df["Sex"].replace("male", 0, inplace=True)
    df["Sex"].replace("female", 1, inplace=True)

    # Convert Embarked column form object to numeric
    df["Embarked"].replace("C", 0, inplace=True)
    df["Embarked"].replace("Q", 1, inplace=True)
    df["Embarked"].replace("S", 2, inplace=True)

    return df


st.title("Visualização de Dados do Titanic")
st.write(
    """Em uma noite tranquila e estrelada em 1912, o lendário navio Titanic partiu em sua jornada inaugural com destino a Nova York.
    A bordo estavam mais de 1.300 passageiros e uma tripulação de aproximadamente 900 pessoas, todos com histórias, sonhos e destinos diferentes."""
)
st.write(
    """
    Mas, como muitos de nós sabemos, a tragédia se abateu sobre o Titanic quando ele colidiu com um iceberg.
    No entanto, além do terrível destino que aguardava o navio, os dados nos contam uma história fascinante sobre os passageiros a bordo."""
)
st.write(
    """
    Ao analisar os dados do Titanic, pude descobrir informações valiosas sobre a composição demográfica dos passageiros e como isso influenciou suas chances de sobrevivência.
    A manipulação dos dados e a visualização são apresentados nos tópicos a seguir."""
)

st.header("Pré-Processamento")
st.write(
    "O Pré-processamento consistiu em carregar o dataset do titanic e fazer manipulações de dados para remover dados faltantes e alterar tipos de dados de algumas colunas."
)
st.write("As 5 primeiras linhas do dataset original, que contém 891 registros, é mostrada a seguir:")

titanic_df = pd.read_csv(TITANIC_DATASET_URL)
st.write(titanic_df.head())

titanic_df = preprocessing(titanic_df.copy())

st.subheader("Remoção de Dados nulos e Filtro de Colunas")
st.write("As linhas com valores nulos na coluna `Age` foram removidas.")
st.write(
    "Além disso, apenas as colunas `PassengerId`,  `Survived`, `Pclass`, `Sex`, `Age` e `Embarked` foram mantidas para a visualização dos dados."
)

st.subheader("Conversão de Tipos de Dados")
st.write(
    "A coluna `Sex` teve seu tipo de dados alterado de `object` para `numeric`, mapeando os valores `male` para `0` e `female` para `1`."
)
st.write(
    "Esta conversão também foi aplicada à coluna `Embarked`, que teve os valores mapeados de `C` para `0`, `Q` para `1` e `S` para `2`."
)

st.header("Visualização de Dados")
st.subheader("Histograma da Idade dos Passageiros")
st.write("O Histograma da Idade dos Passageiros mostra que a maior parte destes tinham entre 20 e 30 anos.")

st.code(
    """
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.hist(titanic_df["Age"])

ax.set_title("Histograma da Idade dos Passageiros")
ax.set_xlabel("Idade")
ax.set_ylabel("Quantidade de Passageiros")
plt.show()
"""
)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.hist(titanic_df["Age"])

ax.set_title("Histograma da Idade dos Passageiros")
ax.set_xlabel("Idade")
ax.set_ylabel("Quantidade de Passageiros")
st.pyplot(fig)

st.write("O que confirmamos analisando a mediana das idades.")
st.code(
    """
titanic_df["Age"].median()  # 28
"""
)
