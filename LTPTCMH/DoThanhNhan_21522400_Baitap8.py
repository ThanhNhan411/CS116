import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import array as arr
import matplotlib.pyplot as plt
import plotly.graph_objs as go
wine_data = load_wine()

df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)

df['target'] = wine_data.target

st.table(df)

X = df.iloc[:, 0:-1];
Y = df.iloc[:, -1:];
X_np = X.to_numpy()
Y_np = Y.to_numpy()
std_X_df = pd.DataFrame(X_np)
accuracy_list = []
for i in range(13):
    pca = PCA(i+1)
    pca_std_X_np = pca.fit_transform(X_np)


    X_train, X_test, Y_train, Y_test = train_test_split(pca_std_X_np, Y.to_numpy(), test_size=0.3, random_state= 42)

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracy_list.append(accuracy)

fig = go.Figure()
fig.add_trace(go.Bar(x=list(range(1, 13)), y=accuracy_list))
fig.update_layout(
    xaxis_title='Number of dimensions',
    yaxis_title='Accuracy',
    title='Model accuracy vs. number of dimensions'
)
st.plotly_chart(fig)




    



