from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib


app = FastAPI(
    title="Deploy lung cancer model",
    version="0.0.1"
)

# --------------------------------
# CARGAR MODELO AI
# --------------------------------
model = joblib.load("model/lung_cancer_model.pkl")


@app.post("/api/v1/predict-lung-cancer", tags=["lung-cancer"])
async def predict(
    index: float,
    age: float,
    gender: float,
    air_pollution: float,
    alcohol_use: float,
    dust_allergy: float,
    occupational_hazards: float,
    genetic_risk: float,
    chronic_lung_disease: float,
    balanced_diet: float,
    obesity: float,
    smoking: float,
    passive_smoker: float,
    chest_pain: float,
    coughing_of_blood: float,
    fatigue: float,
    weight_loss: float,
    shortness_of_breath: float,
    wheezing: float,
    swallowing_difficulty: float,
    clubbing_of_finger_nails: float,
    frequent_cold: float,
    dry_cough: float,
    snoring: float
):
    dictionary = {
        'index': index,
        'Age': age,
        'Gender': gender,
        'Air Pollution': air_pollution,
        'Alcohol use': alcohol_use,
        'Dust Allergy': dust_allergy,
        'OccuPational Hazards': occupational_hazards,
        'Genetic Risk': genetic_risk,
        'chronic Lung Disease': chronic_lung_disease,
        'Balanced Diet': balanced_diet,
        'Obesity': obesity,
        'Smoking': smoking,
        'Passive Smoker': passive_smoker,
        'Chest Pain': chest_pain,
        'Coughing of Blood': coughing_of_blood,
        'Fatigue': fatigue,
        'Weight Loss': weight_loss,
        'Shortness of Breath': shortness_of_breath,
        'Wheezing': wheezing,
        'Swallowing Difficulty': swallowing_difficulty,
        'Clubbing of Finger Nails': clubbing_of_finger_nails,
        'Frequent Cold': frequent_cold,
        'Dry Cough': dry_cough,
        'Snoring': snoring
    }

    try:
        df = pd.DataFrame(dictionary, index=[0])
        prediction = model.predict(df)
        prediction = int(prediction[0])
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"prediction":prediction}
        )
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_400_BAD_REQUEST
        )
