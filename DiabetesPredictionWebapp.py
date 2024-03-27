# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 23:46:30 2024

@author: 23531
"""

import numpy as np
import pickle
import streamlit as stl

# Loading saved model
loaded_model = pickle.load(open("myTrained_model.sav","rb"))

def Diabetes_Predition(input_data):
    
    input_as_array=np.asarray(input_data).reshape(1,-1)
    prediction=loaded_model.predict(input_as_array) 
    if prediction[0]==0:
        return "The person has not Diabetes"
    else:
        return "The person has Diabetes"
    
    
def main():
    # Title
    stl.title("Diabetes Predictive System")
    
    # Getting user input
    Pregnancies = stl.text_input(" number of pregnancies")	
    Glucose	= stl.text_input(" Glucose value")
    BloodPressure	= stl.text_input("Blood pressure level ")
    SkinThickness	= stl.text_input("Enter skin thickness")
    Insulin	= stl.text_input("Enter Insulin")
    BMI	= stl.text_input("Enter BMI")
    DiabetesPedigreeFunction= stl.text_input("Diabetes pedigree function")	
    Age	= stl.text_input("How old are you?")
    
    # Prediction code
    diagnosis = ""
    
    if(stl.button("Check status")):
        diagnosis = Diabetes_Predition([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin, BMI,DiabetesPedigreeFunction, Age ])
        
    stl.success(diagnosis)
if __name__ == '__main__':
    main()
    
        
    
    