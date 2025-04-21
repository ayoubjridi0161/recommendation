from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random

app = FastAPI()

# CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and preprocess data
class RecommendationModel:
    def __init__(self):
        self.data = pd.read_excel("C:/Users/jridi/Desktop/model/api/gymRecommendation.xlsx")
        self.data.drop(columns=['ID'], inplace=True)
        
        # Label encoding for categorical columns
        label_enc = LabelEncoder()
        for col in ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type']:
            self.data[col] = label_enc.fit_transform(self.data[col])
        
        # Standard scaling for numerical features
        self.scaler = StandardScaler()
        self.data[['Age', 'Height', 'Weight', 'BMI']] = self.scaler.fit_transform(
            self.data[['Age', 'Height', 'Weight', 'BMI']])

model = RecommendationModel()

# Request model for recommendation input
class RecommendationRequest(BaseModel):
    Sex: int
    Age: int
    Height: float
    Weight: float
    Hypertension: int
    Diabetes: int
    BMI: float
    Level: int
    FitnessGoal: int
    FitnessType: int

# Response model for recommendation output
class RecommendationResponse(BaseModel):
    exercises: str
    equipment: str
    diet: str
    recommendation: str

@app.post("/recommend", response_model=list[RecommendationResponse])
def get_recommendation(request: RecommendationRequest):
    """
    Get fitness recommendations based on user input
    """
    user_input = {
        'Sex': request.Sex,
        'Age': request.Age,
        'Height': request.Height,
        'Weight': request.Weight,
        'Hypertension': request.Hypertension,
        'Diabetes': request.Diabetes,
        'BMI': request.BMI,
        'Level': request.Level,
        'Fitness Goal': request.FitnessGoal,
        'Fitness Type': request.FitnessType
    }

    # Normalize numerical features
    num_features = ['Age', 'Height', 'Weight', 'BMI']
    user_df = pd.DataFrame([user_input], columns=num_features)
    user_df[num_features] = model.scaler.transform(user_df[num_features])
    user_input.update(user_df.iloc[0].to_dict())
    user_df = pd.DataFrame([user_input])

    # Calculate similarity scores
    user_features = model.data[['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']]
    similarity_scores = cosine_similarity(user_features, user_df).flatten()
    similar_user_indices = similarity_scores.argsort()[-5:][::-1]
    similar_users = model.data.iloc[similar_user_indices]
    recommendation_1 = similar_users[['Exercises', 'Diet', 'Equipment', 'Recommendation']].mode().iloc[0]

    # Generate variations
    simulated_recommendations = []
    for _ in range(2):
        modified_input = user_input.copy()
        modified_input['Age'] += random.randint(-5, 5)
        modified_input['Weight'] += random.uniform(-5, 5)
        modified_input['BMI'] += random.uniform(-1, 1)

        modified_user_df = pd.DataFrame([modified_input], columns=num_features)
        modified_user_df[num_features] = model.scaler.transform(modified_user_df[num_features])
        modified_input.update(modified_user_df.iloc[0].to_dict())

        modified_similarity_scores = cosine_similarity(user_features, pd.DataFrame([modified_input])).flatten()
        modified_similar_user_indices = modified_similarity_scores.argsort()[-5:][::-1]
        modified_similar_users = model.data.iloc[modified_similar_user_indices]
        recommendation = modified_similar_users[['Exercises', 'Diet', 'Equipment', 'Recommendation']].mode().iloc[0]

        if not any(
            rec['Exercises'] == recommendation['Exercises'] and 
            rec['Diet'] == recommendation['Diet'] and 
            rec['Equipment'] == recommendation['Equipment'] and 
            rec['Recommendation'] == recommendation['Recommendation'] 
            for rec in simulated_recommendations
        ):
            simulated_recommendations.append(recommendation)

    # Format response
    recommendations = [recommendation_1] + simulated_recommendations
    return [
        RecommendationResponse(
            exercises=rec['Exercises'],
            equipment=rec['Equipment'],
            diet=rec['Diet'],
            recommendation=rec['Recommendation']
        ) for rec in recommendations
    ]

@app.get("/health")
def health_check():
    return {"status": "healthy"}