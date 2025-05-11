import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import EMBARKED_DICT, PCLASS_DICT, get_titanic_df_preprocessed


@st.cache_data
def train_survival(df):
    df.dropna(inplace=True)

    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101
    )

    log_model = LogisticRegression()

    log_model.fit(X_train, y_train)

    return log_model


if __name__ == "__main__":
    st.set_page_config(page_title="Previsão de sobrevivência", page_icon=":ship:")

    st.title("Previsão de sobrevivência")
    st.markdown(
        """Utilizando Regressão Logística, é possível se você sobreviveria ao
        naufrágio do Titanic com base em sua **Classe**, **Sexo**, **Idade** e
        **Porto de embarque**. Preencha os dados abaixo e confira.
        """
    )

    titanic_df = get_titanic_df_preprocessed()
    trained_model = train_survival(titanic_df.copy())

    with st.form("survival_prediction_form"):
        st.markdown("### Informe seus dados:")

        pclass = st.selectbox("Classe", ("1ª Classe", "2ª Classe", "3ª Classe"))
        sex = st.radio(label="Sexo", options=["Masculino", "Feminino"])
        age = st.number_input(label="Idade", min_value=1, max_value=150, step=1)
        embarked = st.selectbox(
            label="Porto de embarque",
            options=("Cherbourg", "Queenstown", "Southampton"),
        )
        submitted = st.form_submit_button(label="Calcular probabilidade")

        if submitted:
            if trained_model:
                pclass_numeric = next(
                    (
                        pclass_num
                        for pclass_num, pclass_str in PCLASS_DICT.items()
                        if pclass_str == pclass
                    ),
                    None,
                )

                sex_numeric = 0 if sex == "Masculino" else 1

                embarked_numeric = next(
                    (
                        embarked_num
                        for embarked_num, embarked_str in EMBARKED_DICT.items()
                        if embarked_str == embarked
                    ),
                    None,
                )

                input_data = pd.DataFrame(
                    {
                        "Pclass": [pclass_numeric],
                        "Sex": [sex_numeric],
                        "Age": [age],
                        "Embarked": [embarked_numeric],
                    }
                )

                y_predicted = trained_model.predict(input_data)

                if y_predicted[0] == 1:
                    st.success("Você teria sobrevivido!")
                else:
                    st.error("Você não teria sobrevivido.")
