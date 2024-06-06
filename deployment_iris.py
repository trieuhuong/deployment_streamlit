# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:41:32 2024

@author: ADMIN
"""

import streamlit as st
import pickle
import pandas as pd
import os


def predict(x1,x2,x3,x4):    # load mô hình
    loaded_model = pickle.load(open('/mount/src/deployment_streamlit/model.sav','rb'))
    row=[]
    row.append(x1)
    row.append(x2)
    row.append(x3)
    row.append(x4)
    data=[]
    data.append(row)
    df=pd.DataFrame(data)
    y_pred=loaded_model.predict(df)
    return y_pred

st.title("Ứng dụng Streamlit dự đoán")
x1 = st.number_input("sepal_length",value=2)
x2 = st.number_input("sepal_width",value=4)
x3 = st.number_input("petal_length",value=6)
x4 = st.number_input("petal_width",value=3)
if st.button('Phân lớp'):
    output_value = predict(x1,x2,x3,x4)
    st.write("Giá trị đầu ra dự đoán:", output_value)
