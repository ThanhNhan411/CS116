import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.write("Chọn file CSV: ")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.table(df.head())

    X = df.iloc[:, 1:-1];
    Y = df.iloc[:, -1:];   
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state= 21522400)
    scaler = StandardScaler()
    y_train_scaled = LabelEncoder().fit_transform(y_train)
    y_test_scaled = LabelEncoder().fit_transform(y_test)
    #DecisionTree

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy_DecisionTree = accuracy_score(y_test, y_pred)


    #SVM

    clf = SVC()
    clf.fit(X_train,y_train)
    y_predSVM = clf.predict(X_test)
    accuracy_SVM = accuracy_score(y_test, y_predSVM)

    #LogisticRegression

    lgt = LogisticRegression()
    lgt.fit(X_train, y_train)
    y_predLG = lgt.predict(X_test)
    accuracy_LG = accuracy_score(y_test,y_predLG)

    #XGBoost

    bst = XGBClassifier()
    bst.fit(X_train, y_train_scaled)
    y_predXG = bst.predict(X_test)
    accuracy_XG = accuracy_score(y_test_scaled, y_predXG)
    
data = {
    '': ["Accuracy Score"],
    'XG Boost': [accuracy_XG],
    'Logistic Regression': [accuracy_LG],
    'Decision Tree': [accuracy_DecisionTree],
    'SVM': [accuracy_SVM]
}

# Tạo DataFrame từ dữ liệu
tb = pd.DataFrame(data)

# Hiển thị bảng với Streamlit
st.table(tb)