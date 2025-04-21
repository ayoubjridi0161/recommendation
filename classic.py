import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random  # Importing the random module for generating random variations

data = pd.read_excel("C:/Users/jridi/Desktop/model/api/gymRecommendation.xlsx")
data.drop(columns=['ID'], inplace = True)

label_enc = LabelEncoder()
for col in ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type']:
    data[col] = label_enc.fit_transform(data[col])
    
scaler=StandardScaler()
data[['Age', 'Height', 'Weight', 'BMI',]]= scaler.fit_transform(data[['Age', 'Height', 'Weight', 'BMI',]])

print(data.head())
print(data.columns)



def get_recommendation(top_n=3):
    # Start the recommendation process with user inputs
    print("Please enter your details for a personalized workout and diet recommendation.")
    
    # Collect user inputs for their details and health information
    user_input = {
        'Sex': 1,  # User's gender
        'Age': 35,  # User's age
        'Height': 1.88,  # User's height
        'Weight': 90,  # User's weight
        'Hypertension': 0,  # Hypertension status
        'Diabetes': 0,  # Diabetes status
        'BMI': 22.3,  # User's BMI
        'Level': 3,  # Fitness level
        'Fitness Goal': 1,  # Fitness goal
        'Fitness Type': 0  # Fitness type
    }

    # Normalize numerical features for consistency
    num_features = ['Age', 'Height', 'Weight', 'BMI']  # Columns to normalize
    user_df = pd.DataFrame([user_input], columns=num_features)  # Create a DataFrame for user input
    user_df[num_features] = scaler.transform(user_df[num_features])  # Normalize numerical features
    user_input.update(user_df.iloc[0].to_dict())  # Update the normalized values in the user input dictionary
    user_df = pd.DataFrame([user_input])  # Create a new DataFrame with updated user input

    # Calculate similarity scores between user input and dataset
    user_features = data[['Sex', 'Age', 'Height', 'Weight', 'Hypertension', 'Diabetes', 'BMI', 'Level', 'Fitness Goal', 'Fitness Type']]  # Features for similarity
    similarity_scores = cosine_similarity(user_features, user_df).flatten()  # Calculate similarity scores

    # Retrieve the top 5 similar users
    similar_user_indices = similarity_scores.argsort()[-5:][::-1]  # Get indices of the most similar users
    similar_users = data.iloc[similar_user_indices]  # Extract similar user data
    recommendation_1 = similar_users[['Exercises', 'Diet', 'Equipment','Recommendation']].mode().iloc[0]  # Get the most common recommendation

    # Generate two additional recommendations by slightly varying the input
    simulated_recommendations = []

    for _ in range(2):  # Loop for generating two additional recommendations
        modified_input = user_input.copy()  # Create a copy of the original input

        # Randomly adjust Age, Weight, and BMI with larger variations
        modified_input['Age'] += random.randint(-5, 5)  # Randomly vary age
        modified_input['Weight'] += random.uniform(-5, 5)  # Randomly vary weight
        modified_input['BMI'] += random.uniform(-1, 1)  # Randomly vary BMI

        # Normalize the modified input values
        modified_user_df = pd.DataFrame([modified_input], columns=num_features)  # Create a DataFrame
        modified_user_df[num_features] = scaler.transform(modified_user_df[num_features])  # Normalize numerical features
        modified_input.update(modified_user_df.iloc[0].to_dict())  # Update normalized values in modified input

        # Calculate similarity scores for the modified input
        modified_similarity_scores = cosine_similarity(user_features, pd.DataFrame([modified_input])).flatten()  # Calculate similarity
        modified_similar_user_indices = modified_similarity_scores.argsort()[-5:][::-1]  # Get indices of similar users
        modified_similar_users = data.iloc[modified_similar_user_indices]  # Extract similar user data
        recommendation = modified_similar_users[['Exercises', 'Diet', 'Equipment','Recommendation']].mode().iloc[0]  # Get the most common recommendation

        # Ensure the new recommendation is unique
        if not any(rec['Exercises'] == recommendation['Exercises'] and rec['Diet'] == recommendation['Diet'] and rec['Equipment'] == recommendation['Equipment'] and rec['Recommendation'] == recommendation['Recommendation'] for rec in simulated_recommendations):
            simulated_recommendations.append(recommendation)  # Add unique recommendation

    # Display recommendations to the user
    print("\nRecommended Workout and Diet Plans based on your input:")
    print("\nRecommendation 1 (Exact match):")
    print("-"*50)
    print("EXERCISES:\n", recommendation_1['Exercises'])  # Added newline
    print("\nEQUIPMENTS:\n", recommendation_1['Equipment'])  # Added newline
    print("\nDIET:\n", recommendation_1['Diet'])  # Added newline*
    print("\nRECOMMENDATION:\n", recommendation_1['Recommendation'])  # Added newline
    
    print("-"*50)

    # Display additional recommendations
    for idx, rec in enumerate(simulated_recommendations, start=2):
        print(f"\nRecommendation {idx} (Slight variation):")
        print("-"*50)
        print("EXERCISES:\n", rec['Exercises'])
        print("\nEQUIPMENTS:\n", rec['Equipment'])
        print("\nDIET:\n", rec['Diet'])
        print("\nRECOMMENDATION:\n", rec['Recommendation'])
        print("-"*50)

    # Collect feedback from the user
    feedback_matrix = []
    for i in range(len(simulated_recommendations) + 1):  # Loop through all recommendations
        feedback = int(input(f"Was recommendation {i+1} relevant? (Yes: 1, No: 0): "))  # Get feedback
        feedback_matrix.append(feedback)  # Store feedback

    # Calculate Mean Reciprocal Rank (MRR)
    relevant_indices = [i + 1 for i, feedback in enumerate(feedback_matrix) if feedback == 1]  # Get ranks of relevant recommendations
    if relevant_indices:  # Check if there are any relevant recommendations
        mrr = np.mean([1 / rank for rank in relevant_indices])  # Calculate MRR
    else:
        mrr = 0.0  # Set MRR to 0 if no recommendations were relevant

    # Display MRR score
    print(f"\nMean Reciprocal Rank (MRR): {mrr:.2f}")

    return [recommendation_1] + simulated_recommendations  # Return all recommendations

# Get and display recommendations
recommendations = get_recommendation(top_n=3)  # Call the function to generate recommendations


