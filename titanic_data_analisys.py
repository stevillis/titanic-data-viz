import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

TITANIC_DATASET_URL = "https://raw.githubusercontent.com/stevillis/titanic-data-viz/master/titanic.csv"
COLOR_TAB_GRAY = "tab:gray"
COLOR_TAB_BLUE = "tab:blue"


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


def train_survival(df):
    df.dropna(inplace=True)

    X = df.drop(["Survived", "PassengerId"], axis=1)
    y = df["Survived"]  # trying to predict

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

    log_model = LogisticRegression()

    log_model.fit(X_train, y_train)

    return log_model


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
st.write("Após o Pré-Processamento, temos o dataset neste formato:")
st.write(titanic_df.head())


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


st.subheader("Histograma da Sexo dos Passageiros")
st.write(
    "A partir do Histograma do Sexo dos Passageiros, visualizamos que a maior parte dos tripulantes era do Sexo Masculino."
)

st.code(
    """
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.hist(titanic_df["Sex"])

ax.set_title("Histograma do Sexo dos Passageiros")
ax.set_xlabel("Sexo")
ax.set_ylabel("Quantidade de Passageiros")

qtd_male = titanic_df[titanic_df["Sex"] == 0]["Sex"].count()
qtd_female = titanic_df[titanic_df["Sex"] == 1]["Sex"].count()

ax.text(0.05, qtd_male + 0.05, qtd_male, ha="center", va="bottom")
ax.text(0.95, qtd_female + 0.05, qtd_female, ha="center", va="bottom")

ax.set_xticks([])
ax.set_yticks([])

ax.text(0.05, 206.5, "Masculino", ha="center", va="bottom", rotation=90, color="white")
ax.text(0.95, 110.5, "Feminino", ha="center", va="bottom", rotation=90, color="white")
plt.show()
"""
)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.hist(titanic_df["Sex"])

ax.set_title("Histograma do Sexo dos Passageiros")
ax.set_xlabel("Sexo")
ax.set_ylabel("Quantidade de Passageiros")

qtd_male = titanic_df[titanic_df["Sex"] == 0]["Sex"].count()
qtd_female = titanic_df[titanic_df["Sex"] == 1]["Sex"].count()

ax.text(0.05, qtd_male + 0.05, qtd_male, ha="center", va="bottom")
ax.text(0.95, qtd_female + 0.05, qtd_female, ha="center", va="bottom")

ax.set_xticks([])
ax.set_yticks([])

ax.text(0.05, 206.5, "Masculino", ha="center", va="bottom", rotation=90, color="white")
ax.text(0.95, 110.5, "Feminino", ha="center", va="bottom", rotation=90, color="white")
st.pyplot(fig)


st.subheader("Gráfico de Pizza do Local de Embarque")
st.write("Quase 80% dos passageiros embarcaram no porto de Southampton.")

st.code(
    """
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

embarked_df = titanic_df.groupby(["Embarked"]).sum()
labels = ["Cherbourg", "Queenstown", "Southampton"]
sizes = embarked_df["PassengerId"]

plt.pie(sizes, labels=labels, autopct="%1.1f%%", textprops={"size": "smaller"})

ax.set_title("Local de Embarque dos Passageiros")

plt.show()
"""
)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

embarked_df = titanic_df.groupby(["Embarked"]).sum()
labels = ["Cherbourg", "Queenstown", "Southampton"]
sizes = embarked_df["PassengerId"]

plt.pie(sizes, labels=labels, autopct="%1.1f%%", textprops={"size": "smaller"})

ax.set_title("Local de Embarque dos Passageiros")

st.pyplot(fig)


st.subheader("Gráfico de Pizza de Classe dos Passageiros")
st.write(
    "Este gráfico mostra que quase metade dos passageiros eram da 3ª classe, enquanto pouco menos de 1/4 era da 2ª classe e pouco mais de 1/4 era da 1ª classe."
)

st.code(
    """
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

embarked_df = titanic_df.groupby(["Pclass"]).sum()
labels = ["1ª classe", "2ª classe", "3ª classe"]
sizes = embarked_df["PassengerId"]

plt.pie(sizes, labels=labels, autopct="%1.1f%%")

ax.set_title("Classe dos Passageiros")

plt.show()
"""
)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

embarked_df = titanic_df.groupby(["Pclass"]).sum()
labels = ["1ª classe", "2ª classe", "3ª classe"]
sizes = embarked_df["PassengerId"]

plt.pie(sizes, labels=labels, autopct="%1.1f%%")

ax.set_title("Classe dos Passageiros")

st.pyplot(fig)

st.header("Visualização de Dados Agregados")
st.subheader("Sobreviventes por Sexo")
st.write("Neste gráfico vemos que a maioria dos sobreviventes é do Sexo Feminino.")

survivors_by_sex = titanic_df.loc[(titanic_df["Survived"] == 1)][["Sex", "Survived"]].groupby("Sex")["Survived"].sum()
deaths_by_sex = titanic_df.loc[(titanic_df["Survived"] == 0)][["Sex", "Survived"]].groupby("Sex")["Survived"].count()

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.bar(
    np.array(["Masculino", "Feminino"]),
    survivors_by_sex.values,
    label="Sobreviveu",
    color=COLOR_TAB_BLUE,
    width=0.4,
    alpha=0.8,
    align="edge",
)

ax.bar(
    np.array(["Masculino", "Feminino"]),
    deaths_by_sex.values,
    label="Não sobreviveu",
    color=COLOR_TAB_GRAY,
    width=0.4,
    alpha=0.8,
)


ax.set_title("Sobreviventes por sexo")
ax.set_xlabel("Sexo")
ax.set_ylabel("Quantidade de Passageiros")

ax.text(0.2, survivors_by_sex[0] + 0.05, survivors_by_sex[0], ha="center", va="bottom")
ax.text(1.2, survivors_by_sex[1] + 0.05, survivors_by_sex[1], ha="center", va="bottom")


ax.text(0, deaths_by_sex[0] + 0.05, deaths_by_sex[0], ha="center", va="bottom")
ax.text(1, deaths_by_sex[1] + 0.05, deaths_by_sex[1], ha="center", va="bottom")

plt.legend(loc="upper right")

st.pyplot(fig)


st.subheader("Porcentagem de Homens sobreviventes")
st.write("A maioria dos passageiros do Sexo Masculino não sobreviveu à tragédia.")

men_survived_percentage = (survivors_by_sex[0] / titanic_df.loc[(titanic_df["Sex"] == 0)]["Sex"].count()) * 100
survived_men_serie = pd.Series(
    [survivors_by_sex[0], titanic_df.loc[(titanic_df["Sex"] == 0) & (titanic_df["Survived"] == 0)]["Sex"].count()]
)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

patches, texts, autotexts = ax.pie(
    x=survived_men_serie.values,
    autopct="%1.2f%%",
    colors=[COLOR_TAB_BLUE, COLOR_TAB_GRAY],
    explode=(0, 0.1),
    textprops={"color": "white"},
)

ax.set_title("Sobreviventes do Sexo Masculino")

plt.legend(labels=["Sobreviveu", "Não sobreviveu"], loc="upper right")
st.pyplot(fig)


st.subheader("Porcentagem de Mulheres sobreviventes")
st.write("A maioria dos passageiros do Sexo Feminino sobreviveu ao naufrágio.")

women_survived_percentage = (survivors_by_sex[1] / titanic_df.loc[(titanic_df["Sex"] == 1)]["Sex"].count()) * 100
survived_women_series = pd.Series(
    [survivors_by_sex[1], titanic_df.loc[(titanic_df["Sex"] == 1) & (titanic_df["Survived"] == 0)]["Sex"].count()]
)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

patches, texts, autotexts = ax.pie(
    x=survived_women_series.values,
    autopct="%1.2f%%",
    colors=["tab:pink", "tab:gray"],
    explode=(0.1, 0),
    startangle=90,
    textprops={"color": "white"},
)

ax.set_title("Sobreviventes do Sexo Feminino")

plt.legend(labels=["Sobreviveu", "Não sobreviveu"], loc="upper right")
st.pyplot(fig)


st.subheader("Sobreviventes por Idade")
st.write(
    "Neste gráfico vemos que a maior parte das crianças sobreviveu, enquanto houve mais mortes de pessoas com idades próximas à idade média."
)

survivors_by_age = titanic_df.loc[(titanic_df["Survived"] == 1)][["Age", "Survived"]].groupby("Age").sum().reset_index()
death_by_age = titanic_df.loc[(titanic_df["Survived"] == 0)][["Age", "Survived"]].groupby("Age").count().reset_index()

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.bar(survivors_by_age["Age"], survivors_by_age["Survived"], label="Sobreviveu")
ax.bar(death_by_age["Age"], death_by_age["Survived"], label="Não sobreviveu", color=COLOR_TAB_GRAY, alpha=0.4)

ax.set_title("Sobreviventes por idade")
ax.set_xlabel("Idade")
ax.set_ylabel("Quantidade de Passageiros")

plt.legend()
st.pyplot(fig)


st.subheader("Sobreviventes por Classe")
st.write("Neste gráfico vemos que a maior parte dos sobreviventes era da 1ª Classe.")

survivors_by_class = (
    titanic_df.loc[(titanic_df["Survived"] == 1)][["Pclass", "Survived"]].groupby("Pclass").count().reset_index()
)
death_by_class = (
    titanic_df.loc[(titanic_df["Survived"] == 0)][["Pclass", "Survived"]].groupby("Pclass").count().reset_index()
)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.bar(survivors_by_class["Pclass"], survivors_by_class["Survived"], label="Sobreviveu")
ax.bar(
    death_by_class["Pclass"],
    death_by_class["Survived"],
    label="Não sobreviveu",
    alpha=0.4,
    color=COLOR_TAB_GRAY,
)

ax.set_title("Sobreviventes por Classe")
ax.set_xlabel("Classe")
ax.set_ylabel("Quantidade de Passageiros")

ax.set_xticklabels(["1ª Classe", "2ª Classe", "3ª Classe"])
ax.set_xticks([1, 2, 3])

plt.legend()
st.pyplot(fig)


st.subheader("Mapa de Calor")
st.write("Nesta representação percebemos a maior correlação ocorre entre as `Sex` e `Survived`.")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

sns.heatmap(titanic_df[["Survived", "Pclass", "Sex", "Age", "Embarked"]].corr(), annot=True, fmt=".2f")

st.pyplot(fig)

st.subheader("Será que você teria sobrevivido ao naufrágio do Titanic?")
st.write(
    "Usando a Regressão Logística, podemos determinar a probabilidade de sobrevivência de acordo com a sua `Classe`, `Sexo`, `Idade` e `Porto de Embarque`."
)
st.write("")

if "trained_model" not in st.session_state:
    trained_model = train_survival(titanic_df.copy())
    st.session_state["trained_model"] = trained_model


with st.form("my_form"):
    st.write("Use os campos abaixo e veja se você teria sobrevivido.")
    pclass = st.selectbox("Classe", ("1ª Classe", "2ª Classe", "3ª Classe"))
    sex = st.radio("Sexo", ["Masculino", "Feminino"])
    age = st.number_input("Idade", min_value=1, max_value=150, step=1)
    embarked = st.selectbox("Porto de Embarque", ("Cherbourg", "Queenstown", "Southampton"))

    submitted = st.form_submit_button("Calcular")
    if submitted:
        log_model = st.session_state.trained_model

        if log_model:
            if pclass == "1ª Class":
                pclas_numeric = 1
            elif pclass == "2ª Class":
                pclas_numeric = 2
            else:
                pclas_numeric = 3

            if embarked == "Cherbourg":
                embarked_numeric = 0
            elif embarked == "Queenstown":
                embarked_numeric = 1
            else:
                embarked_numeric = 2

            sex_numeric = 0 if sex == "Masculino" else 1

            y_predicted = log_model.predict(
                pd.DataFrame(
                    {"Pclass": [pclas_numeric], "Sex": [sex_numeric], "Age": [age], "Embarked": [embarked_numeric]}
                )
            )

            if y_predicted[0] == 1:
                st.write(":green[Você teria sobrevivido!]")
            else:
                st.write(":red[Você teria morrido!]")

st.header("Referências")
st.write(
    """
- https://github.com/carlosfab/data_science/blob/master/Titanic.ipynb
- https://www.kaggle.com/competitions/titanic
- https://www.kaggle.com/code/roblexnana/data-viz-tutorial-with-titanic-and-seaborn
- https://predictivelearning.github.io/projects/Project_053_Visualizing_Data_with_Seaborn__Titanic.html
- https://dev.to/shehanat/how-to-create-an-age-distribution-graph-using-python-pandas-and-seaborn-2o5n
- https://medium.com/@melodyyip_/titanic-survival-prediction-using-machine-learning-89a779656113
- https://www.kaggle.com/code/dejavu23/titanic-survival-seaborn-and-ensembles
- https://blog.devgenius.io/analyzing-the-titanic-dataset-a-story-of-tragedy-and-survival-48883b2f2d48
- https://www.kaggle.com/code/punit0811/linear-regression-with-titanic-dataset
"""
)
