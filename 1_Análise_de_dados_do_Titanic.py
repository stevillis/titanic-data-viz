import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from utils import CUSTOM_CSS, EMBARKED_DICT, get_titanic_df, get_titanic_df_preprocessed


def introduction():
    st.title("Análise de Dados do Titanic")
    st.markdown(
        """Em 1912, o lendário navio Titanic partiu em sua jornada inaugural
        com destino a Nova York. A bordo estavam mais de 1.300 passageiros e
        uma tripulação de aproximadamente 900 pessoas. Porém, uma tragédia se
        abateu sobre o navio quando ele colidiu com  um iceberg.
        <br>
        <br>
        Ao analisar um conjunto de dados do Titanic com 891 registros, pude
        descobrir informações valiosas sobre a composição demográfica dos
        passageiros e como isso influenciou suas chances de sobrevivência.
        <br>
        <br>
        Além disso, pude treinar um modelo de predição de sobrevivência
        baseado em algumas variáveis como classe social, sexo e idade. Você
        pode testar o modelo <a href="./Previsão_de_sobrevivência" target="_blank">
        aqui</a> ou no menu lateral.
        <br>
        <br>
        As 5 primeiras linhas do conjunto de dados analisado são mostrada a seguir:
        """,
        unsafe_allow_html=True,
    )

    titanic_df = get_titanic_df()
    st.dataframe(titanic_df.head())


def preprocessing_section():
    st.header("Pré-processamento")
    st.markdown(
        """Essa fase consistiu em remover dados faltantes e alterar tipos de
        dados de algumas colunas.
        """
    )


def removing_null_values():
    st.markdown(
        """
    #### Remoção de dados nulos e filtro de colunas
    As linhas com valores nulos na coluna `Age` foram removidas, assim
    como as linhas cujo local de embarque não foi informado, para não
    prejudicar o treinamento de um modelo de predição de sobrevivência.
    <br>
    <br>
    Além disso, apenas as colunas `Survived`, `Pclass`, `Sex`, `Age` e
    `Embarked` foram mantidas para a visualização dos dados.
    """,
        unsafe_allow_html=True,
    )


def converting_data_types():
    st.markdown(
        """
    #### Conversão de tipos de dados

    A coluna `Sex` teve seu tipo de dados alterado de `object` para `numeric`,
    mapeando os valores `male` para `0` e `female` para `1`.
    <br>
    <br>
    O mesmo procedimento foi aplicado à coluna `Embarked`, que teve os valores
    mapeados de `C` para `0`, `Q` para `1` e `S` para `2`.
    <br>
    <br>
    As cinco primeiras linhas do conjunto de dados após o pré-processamento
    são mostradas abaixo:
    """,
        unsafe_allow_html=True,
    )

    titanic_df = get_titanic_df_preprocessed()
    st.dataframe(titanic_df.head())


def visualizing_data():
    st.header("Visualização de Dados")
    st.subheader("Histograma de idade dos passageiros")
    st.markdown(
        """
        O histograma a seguir mostra a distribuição de idade dos passageiros.
        Nesse cenário, a maior parte tinha entre 20 e 30 anos, como mostram
        as duas barras mais altas do histograma.
        """
    )

    titanic_df = get_titanic_df_preprocessed()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.hist(titanic_df["Age"], color="skyblue", edgecolor="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("", fontsize=16)
    ax.set_xlabel("Idade", fontsize=14)
    ax.set_ylabel("Quantidade de passageiros", fontsize=14)

    st.pyplot(fig)

    st.markdown(
        """A mediana de idade igual a 28 anos confirma a hipótese de que
        a maior parte dos passageiros tinha entre 20 e 30 anos.
    """
    )

    st.code("titanic_df['Age'].median() # 28", language="python")


def visualizing_sex():
    st.subheader("Distribuição de Sexo dos passageiros")
    st.markdown(
        """Ao explorar a distribuição de sexo dos passageiros, verifica-se que
        a maioria era do sexo masculino.
        """
    )

    titanic_df = get_titanic_df_preprocessed()

    counts = titanic_df["Sex"].value_counts().sort_index()
    labels = {0: "Masculino", 1: "Feminino"}

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=[labels[k] for k in counts.index],
        y=counts.values,
        palette=["steelblue", "salmon"],
        ax=ax,
    )

    for i, count in enumerate(counts.values):
        ax.text(i, count + 3, str(count), ha="center", va="bottom", fontsize=12)

        percentage = (count / counts.values.sum()) * 100
        ax.text(
            i,
            count / 2,
            f"{percentage:.2f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
        )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_title("", fontsize=16)
    ax.set_xlabel("Sexo", fontsize=14)
    ax.set_ylabel("Quantidade de Passageiros", fontsize=14)

    st.pyplot(fig)


def visualizing_embarked():
    st.subheader("Distribuição de Local de Embarque")
    st.write("Quase 80% dos passageiros embarcaram no porto de Southampton.")

    titanic_df = get_titanic_df_preprocessed()

    embarked_counts = titanic_df["Embarked"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x=embarked_counts.index.map(EMBARKED_DICT),
        y=embarked_counts.values,
        palette=["lightcoral", "gold", "skyblue"],
        ax=ax,
    )

    for i, count in enumerate(embarked_counts.values):
        ax.text(i, count + 3, str(count), ha="center", va="bottom", fontsize=12)

        percentage = (count / embarked_counts.values.sum()) * 100
        ax.text(
            i,
            count / 2,
            f"{percentage:.2f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title("", fontsize=16)
    ax.set_xlabel("Porto de Embarque", fontsize=14)
    ax.set_ylabel("Número de Passageiros", fontsize=14)

    st.pyplot(fig)


def visualizing_passenger_class():
    st.subheader("Distribuição de Classe Social")
    st.write(
        """A análise da distribuição de passageiros por classe social revela
        que a terceira classe é a mais representativa, enquanto a primeira e a
        segunda têm praticamente a mesma quantidade de passageiros.
        """
    )

    titanic_df = get_titanic_df_preprocessed()

    class_counts = titanic_df["Pclass"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x=["1ª Classe", "2ª Classe", "3ª Classe"],
        y=class_counts.values,
        ax=ax,
    )

    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 5, str(v), ha="center", va="bottom", fontsize=12)
        percentage = (v / class_counts.values.sum()) * 100
        ax.text(
            i,
            v / 2,
            f"{percentage:.2f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylabel("Número de Passageiros", fontsize=12)
    ax.set_xlabel("")

    st.pyplot(fig)


def aggregated_data():
    st.header("Visualização de dados agregados")
    st.subheader("Sobreviventes por Sexo")
    st.markdown(
        """Enquanto 75% das mulheres sobreviveram, apenas 20% dos homens
        conseguiram sobreviver ao desastre."""
    )

    titanic_df = get_titanic_df_preprocessed()

    survival_data = pd.crosstab(titanic_df["Sex"], titanic_df["Survived"])

    survival_rates = (
        survival_data[1] / (survival_data[0] + survival_data[1]) * 100
    ).round(2)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(survival_data.index))
    width = 0.35

    survived_bars = ax.bar(
        x=x - width / 2,
        height=survival_data[1],
        width=width,
        label="Sobreviveram",
        color="#2ecc71",
    )

    death_bars = ax.bar(
        x=x + width / 2,
        height=survival_data[0],
        width=width,
        label="Não Sobreviveram",
        color="#e74c3c",
    )

    genders = survival_data.index.tolist()
    for i, gender in enumerate(genders):
        height_survived = survival_data.loc[gender, 1]
        ax.text(
            x=x[i] - width / 2,
            y=height_survived,
            s=str(height_survived),
            ha="center",
            va="bottom",
        )
        ax.text(
            x=x[i] - width / 2,
            y=height_survived / 2,
            s=f"{survival_rates[gender]:.2f}%",
            ha="center",
            va="bottom",
            color="white",
        )

        height_death = survival_data.loc[gender, 0]
        ax.text(
            x=x[i] + width / 2,
            y=height_death,
            s=str(height_death),
            ha="center",
            va="bottom",
        )
        ax.text(
            x=x[i] + width / 2,
            y=height_death / 2,
            s=f"{100 - survival_rates[gender]:.2f}%",
            ha="center",
            va="bottom",
            color="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Masculino", "Feminino"])

    ax.set_ylabel("Número de Passageiros")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right")

    st.pyplot(fig)


def survivors_by_age():
    st.subheader("Distribuição de Idade dos passageiros por Sobrevivência")
    st.markdown(
        """O gráfico mostra como a idade influenciou a sobrevivência no
        Titanic. Note a maior densidade de sobreviventes entre crianças e
        jovens adultos.
        """
    )

    titanic_df = get_titanic_df_preprocessed()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(
        data=titanic_df[titanic_df["Survived"] == 1]["Age"],
        label="Sobreviveu",
        color="#2ecc71",
        fill=True,
        alpha=0.3,
    )

    sns.kdeplot(
        data=titanic_df[titanic_df["Survived"] == 0]["Age"],
        label="Não Sobreviveu",
        color="#e74c3c",
        fill=True,
        alpha=0.3,
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlabel("Idade")
    ax.set_ylabel("Densidade")

    ax.legend(loc="upper right")

    st.pyplot(fig)


def survivors_by_class():
    st.subheader("Taxa de Sobrevivência por Classe Social")
    st.markdown(
        """A classe social foi um fator determinante para sobrevivência no
        Titanic. Passageiros da primeira classe tiveram maior taxa de
        sobrevivência que as demais. A terceira classe foi a que teve maior
        taxa de mortes.
        """
    )

    titanic_df = get_titanic_df_preprocessed()

    class_survival = pd.crosstab(titanic_df["Pclass"], titanic_df["Survived"])
    survival_rates = (
        class_survival[1] / (class_survival[0] + class_survival[1]) * 100
    ).round(2)

    # st.write(titanic_df.groupby("Pclass")["Survived"].value_counts())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(3)
    width = 0.35

    survived = ax.bar(
        x - width / 2, class_survival[1], width, label="Sobreviveram", color="#2ecc71"
    )

    died = ax.bar(
        x + width / 2,
        class_survival[0],
        width,
        label="Não Sobreviveram",
        color="#e74c3c",
        alpha=0.7,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pclass_labels = class_survival.index.tolist()
    for i, pclass in enumerate(pclass_labels):
        # Sobreviventes
        height_survived = class_survival.loc[pclass, 1]
        ax.text(
            x=x[i] - width / 2,
            y=height_survived,
            s=str(height_survived),
            ha="center",
            va="bottom",
        )
        ax.text(
            x=x[i] - width / 2,
            y=height_survived / 2,
            s=f"{survival_rates[pclass]:.2f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
        )

        # Não sobreviventes
        height_died = class_survival.loc[pclass, 0]
        ax.text(
            x=x[i] + width / 2,
            y=height_died,
            s=str(height_died),
            ha="center",
            va="bottom",
        )
        ax.text(
            x=x[i] + width / 2,
            y=height_died / 2,
            s=f"{100 - survival_rates[pclass]:.2f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
        )

    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(["1ª Classe", "2ª Classe", "3ª Classe"])

    ax.set_ylabel("Número de Passageiros")

    ax.legend(loc="upper left")

    st.pyplot(fig)


def correlation_heatmap():
    st.subheader("Correlações entre variáveis")
    st.markdown(
        """
        Nesta representação é possível observar que a maior correlação positiva
        ocorre entre as variáveis `Sex` e `Survived`. Já a maior correlação
        negativa ocorre entre as variáveis `Pclass` e `Age`.
        """
    )

    titanic_df = get_titanic_df_preprocessed()

    corr_matrix = titanic_df[["Survived", "Pclass", "Sex", "Age", "Embarked"]].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu",
        center=0,
        square=True,
        cbar_kws={"label": "Correlação"},
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    st.pyplot(fig)


def references():
    st.header("Referências")
    st.markdown(
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


if __name__ == "__main__":
    st.set_page_config(
        page_title="Visualização de dados do Titanic", page_icon=":ship:"
    )

    st.markdown(
        CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    introduction()

    preprocessing_section()
    removing_null_values()
    converting_data_types()

    visualizing_data()
    visualizing_sex()
    visualizing_embarked()
    visualizing_passenger_class()

    aggregated_data()
    survivors_by_age()
    survivors_by_class()

    correlation_heatmap()

    references()
