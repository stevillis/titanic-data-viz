import pandas as pd
import streamlit as st

TITANIC_DATASET_URL = "./titanic.csv"

CUSTOM_CSS = """
    <style>
        /* Text Content */
        [data-testid="stMarkdown"] {
            text-align: justify;
        }
    </style>
"""

COLOR_TAB_GRAY = "tab:gray"
COLOR_TAB_BLUE = "tab:blue"

EMBARKED_DICT = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}
PCLASS_DICT = {1: "1ª Classe", 2: "2ª Classe", 3: "3ª Classe"}


def preprocess_titanic_df(df):
    # Remove null values from Age column
    df = df[df["Age"].notna()]
    df = df[df["Embarked"].notna()]

    # Filter columns of interest
    df = df[["Survived", "Pclass", "Sex", "Age", "Embarked"]]

    # Convert Sex column from object to numeric
    df["Sex"].replace("male", 0, inplace=True)
    df["Sex"].replace("female", 1, inplace=True)

    # Convert Embarked column form object to numeric
    df["Embarked"].replace("C", 0, inplace=True)
    df["Embarked"].replace("Q", 1, inplace=True)
    df["Embarked"].replace("S", 2, inplace=True)

    return df


@st.cache_data
def get_titanic_df():
    titanic_df = pd.read_csv(TITANIC_DATASET_URL)
    return titanic_df


@st.cache_data
def get_titanic_df_preprocessed():
    titanic_df = get_titanic_df()
    titanic_df = preprocess_titanic_df(titanic_df.copy())
    return titanic_df
