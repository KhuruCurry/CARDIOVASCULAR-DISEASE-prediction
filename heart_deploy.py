# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 16:09:05 2022

@author: Khuru
"""

import pickle
import os
import numpy as np
import streamlit as st

BEST_MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')

with open(BEST_MODEL_PATH,'rb') as file:
    model = pickle.load(file)
    
# new_X = ['age','trtbps','chol','thalachh','oldpeak','cp','thall']

# cat = ['sex','cp','fbs','restecg','exng','slp','caa','output','thall']
# con = ['age','trtbps','chol','thalachh','oldpeak']

# outcome = model.predict((X_new))

#assign each 
age = 70
trtbps = 80
chol = 200
thalachh = 190
oldpeak = 40
cp = 1.5
thall = 1


with st.form("my_form"):
    st.write("This app is to predict the chance of a person having CARDIO VASCULAR DISEASE(CVD)")
    st.write('A short video of CVD (CARDIOVASCULAR DISEASE)')
    #create video input
    data = 'https://www.youtube.com/watch?v=h413NHcx7eo'
    st.video(data, format="video/mp4", start_time=0)

#create form

    st.write("Want to find out more? Here,take a quick info for more!")
    
    age = int(st.number_input('Please key in your age'))
    trtbps = int(st.number_input('Please Key in your resting blood pressure(in mm Hg on admission to the hospital'))
    chol = int(st.number_input('Key in your serum cholestoral in mg/dl'))  
    thalachh = int(st.number_input('Key in your maximum heart rate achieved'))
    oldpeak = int(st.number_input('Key in your ST depression induced by exercise relative to rest'))
    cp = int(st.radio('Click at your chest pain type(1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)',(1,2,3,4)))
    thall = int(st.radio('Click at your thall type,2 = normal; 1 = fixed defect; 3 = reversable defect',(2,1,3)))
    
    # Every form must have a submit button
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write('age',age,'trtbps',trtbps,'chol',chol,'thalachh',thalachh,
                 'oldpeak',oldpeak,'cp',cp,'thall',thall)
        temp = np.expand_dims([age, trtbps, chol, thalachh, oldpeak, cp, thall],axis=0)
        outcome = model.predict(temp)
        
        outcome_dict = {0:'Less risk of getting heart attack',
                        1:'Higher chance getting heart attack'}
        
        st.write(outcome_dict[outcome[0]])
        
        if outcome == 1:
            st.snow()
            st.write("There is high chance of getting heart attack.")
        else:
            st.balloons()
            st.write("You have low risk of getting heart attack")