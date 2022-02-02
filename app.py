# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:24:34 2022

@author: antwi
"""

import streamlit as st
import numpy as np
import string
import pickle
import sklearn
st.set_option('deprecation.showfileUploaderEncoding',False)
model = pickle.load(open('model2.pkl','rb'))


def main():
  st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>Amount Paid Predictor</h1>", unsafe_allow_html=True)
  st.markdown("<h3 style='text-align: center; color: Black;'>Drop in The required Inputs and we will do  the rest.</h3>", unsafe_allow_html=True)
  st.sidebar.header("What is this Project about?")
  st.sidebar.text("It a Web app that would help the user in determining the amount paid .")
  



  patient_age = st.number_input("Enter patient age")
  claim_amount = st.number_input("Enter claim amount")
  National_Provider_Identifier_code = st.number_input("Enter identifier code")
  Diagnosis_Related_Group_code = st.number_input("Enter diagnosis code")
  #is_medicaid = st.selectbox("is medicaid",('True','False'))
  #is_medicare= st.selectbox("is medicare",('True','False'))

  inputs = [[patient_age,claim_amount,National_Provider_Identifier_code,Diagnosis_Related_Group_code]]

  if st.button('Predict'):
    result = model.predict(inputs)
    st.success('The Estimated amount paid is {} dollars'.format(result))


if __name__ =='__main__':
  main()







