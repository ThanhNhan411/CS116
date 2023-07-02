import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import f1_score

def app():
    st.title("Chọn file CSV: ")

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write(df)
        st.write("### Chọn output:")
        all_columns = df.columns.tolist()
        selected_columns_out = st.selectbox("", all_columns)
        st.write("### DataFrame:")
        st.write(df[selected_columns_out])

        st.write("### Chọn input:")
        new_df = df.drop(columns=selected_columns_out)
        all_columns = new_df.columns.tolist()
        selected_columns_in = st.multiselect("", all_columns, default=None)
        st.write("### DataFrame:")
        st.write(df[selected_columns_in])
        st.write("### Chọn model: ")
        if st.button("Logistics Regression"):
            # Add form for hyperparameters

            X = np.asarray(df[selected_columns_in])
            y = np.asarray(df[selected_columns_out])
            X = preprocessing.StandardScaler().fit(X).transform(X)

            # Train logistic regression model
            X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
            model = LogisticRegression(C=0.01, solver="liblinear")
            model.fit(X_train, y_train)

            # Add prediction input and button
            y_hat = model.predict(X_test)
            
            st.write("## F1 Score: ", f1_score(y_test, y_hat, average="macro"))
        if st.button("Linear Regression"):
            msk = np.random.rand(len(df)) < 0.8
            train = df[msk]
            test = df[~msk]
            regr = linear_model.LinearRegression()
            x = np.asanyarray(train[selected_columns_in])
            y = np.asanyarray(train[selected_columns_out])
            regr.fit (x, y)
            y_hat= regr.predict(test[selected_columns_in])
            x = np.asanyarray(test[selected_columns_in])
            y = np.asanyarray(test[selected_columns_out])
            st.write('## Variance score: ', regr.score(x, y))

if __name__ == '__main__':
    app()
