# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:56:16 2022

@author: Khuru
"""

import pickle 
import os
import streamlit as st
import numpy as np

MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')

with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)
    
LE_PATH = os.path.join(os.getcwd(),'Label_Encoder.pkl')
with open(LE_PATH,'rb') as file:
    le = pickle.load(file)
    
#new_X = ['Perimeter','MajorAxisLength','MinorAxisLength',
#         'AspectRation', 'Eccentricity','EquivDiameter',
#         'Compactness', 'ShapeFactor3']

le.inverse_transform([5])

#%% To test

with st.form("my_form"):
    Perimeter = st.number_input('Perimeter')
    MajorAxisLength = st.number_input('MajorAxisLength')
    MinorAxisLength = st.number_input('MinorAxisLength')
    AspectRation = st.number_input('AspectRation')
    Eccentricity = st.number_input('Eccentricity')
    EquivDiameter = st.number_input('EquivDiameter')
    Compactness = st.number_input('Compactness')
    ShapeFactor3 = st.number_input('ShapeFactor3')
  

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        X_new = [Perimeter,MajorAxisLength,MinorAxisLength,
                 AspectRation, Eccentricity,EquivDiameter,
                 Compactness, ShapeFactor3]
        outcome = model.predict(np.expand_dims(np.array(X_new),axis=0))
        st.write(le.inverse_transform(outcome))
                      

st.write("Outside the form")