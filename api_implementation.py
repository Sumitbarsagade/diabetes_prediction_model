# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 19:14:44 2022

@author: siddhardhan
"""


import json
import requests

# 1,146,56,0,0,29.7,0.564,29
# 2,71,70,27,0,28,0.586,22
url = 'http://127.0.0.1:8000/diabetes_prediction'
# url= 'https://diabetes-prediction-3hj8.onrender.com/diabetes_prediction'

input_data_for_model = {
    
    'pregnancies' : 2,
    'Glucose' : 71,
    'BloodPressure' : 70,
    'SkinThickness' : 27,
    'Insulin' : 0,
    'BMI' : 28,
    'DiabetesPedigreeFunction' : 0.586,
    'Age' : 22
    
    }

input_json = json.dumps(input_data_for_model)

response = requests.post(url, data=input_json)

print(response.text)
