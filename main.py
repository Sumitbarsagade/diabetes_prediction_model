
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sklearn.preprocessing import StandardScaler
import json
import uvicorn
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify specific origins instead of "*" if needed
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
scaler = StandardScaler()

class model_input(BaseModel):
    
    pregnancies : int
    Glucose : int
    BloodPressure : int
    SkinThickness : int
    Insulin : int
    BMI : float
    DiabetesPedigreeFunction : float
    Age : int       
        
# loading the saved model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

@app.post('/diabetes_prediction')
def diabetes_pred(input_parameters : model_input):
    
    input_data = input_parameters.model_dump_json()
    input_dictionary = json.loads(input_data)
    preg = input_dictionary['pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']
    
    
    # input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    data = np.array([[preg, glu, bp, skin, insulin, bmi, dpf,age]])
    input_data_reshaped = data.reshape(1,-1)
    scaler.fit(input_data_reshaped)
    std_data = scaler.transform(input_data_reshaped)

    
    # arr = np.array([[data1, data2, data3, data4]])
    prediction = diabetes_model.predict(std_data)
    # prediction = diabetes_model.predict([input_list])
    
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'




    



