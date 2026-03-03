import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Load dataset containing movie details
movies_df = pd.read_csv("BollywoodMovieDetail.csv")

# Clean and preprocess the data
# Extracting the first genre and handling missing actor data
movies_df['genre'] = movies_df['genre'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else x)
movies_df['actors'] = movies_df['actors'].fillna('')
movies_df['description'] = movies_df['title']
movies_df['releaseYear'] = pd.to_numeric(movies_df['releaseYear'], errors='coerce').fillna(0).astype(int)

# Vectorizing the movie descriptions using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(movies_df['description']).toarray()

# Encoding the movie genres
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(movies_df['genre'])
y = to_categorical(y)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# Function to create the movie recommendation model
def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train the model
model = build_model(X_train.shape[1], y.shape[1])
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Movie recommendation function based on user inputs
def recommend_movie(user_inputs, actor_pref, actress_pref, start_year, end_year, vectorizer, label_encoder, model, movies_df):
    user_vectors = vectorizer.transform(user_inputs).toarray()
    predictions = model.predict(user_vectors)
    predicted_genres = [label_encoder.inverse_transform([np.argmax(pred)])[0] for pred in predictions]

    unique_genres = list(set(predicted_genres))
    recommended_movies = movies_df[movies_df['genre'].isin(unique_genres)]

    # Filter by actor preference
    if actor_pref:
        actor_pref = actor_pref.lower()
        recommended_movies = recommended_movies[recommended_movies['actors'].str.contains(actor_pref, case=False, na=False)]

    # Filter by actress preference
    if actress_pref:
        actress_pref = actress_pref.lower()
        recommended_movies = recommended_movies[recommended_movies['actors'].str.contains(actress_pref, case=False, na=False)]

    # Filter by year range
    recommended_movies = recommended_movies[(recommended_movies['releaseYear'] >= start_year) & (recommended_movies['releaseYear'] <= end_year)]

    # Return a sample of 20 recommended movies
    num_movies_to_return = min(20, len(recommended_movies))
    recommended_movies = recommended_movies.sample(n=num_movies_to_return, random_state=np.random.randint(0, 1000))

    return unique_genres, recommended_movies[['title', 'releaseYear']].values.tolist()

# Streamlit UI setup
st.title('Bollywood Movie Recommender')

st.write("""
    Enter 2-3 movie descriptions you like, your actor/actress preferences, and your release year range.
    The model will recommend movies based on your input.
""")

# Sidebar for additional info
st.sidebar.header("Project Info")
st.sidebar.write("""
    This project is a Bollywood movie recommender based on genre prediction using a machine learning model.
    The model is trained using movie descriptions and generates genre-based recommendations.
""")

# Input for movie descriptions
user_inputs = []
for i in range(3):
    desc = st.text_input(f"Movie {i+1} description (press enter to skip):", "")
    if desc:
        user_inputs.append(desc.lower())

# Input for actor and actress preferences
actor_pref = st.text_input("Enter your favorite actor (press enter to skip):", "").strip().lower()
actress_pref = st.text_input("Enter your favorite actress (press enter to skip):", "").strip().lower()

# Input for release year range
start_year = st.number_input("Enter the earliest release year you prefer (e.g., 2000):", min_value=1900, max_value=2025, value=1900)
end_year = st.number_input("Enter the latest release year you prefer (e.g., 2023):", min_value=1900, max_value=2025, value=2025)

# Display recommendations when the button is pressed
if st.button('Get Recommendations'):
    if user_inputs:
        predicted_genres, recommendations = recommend_movie(user_inputs, actor_pref, actress_pref, start_year, end_year, vectorizer, label_encoder, model, movies_df)

        st.write("### Predicted Genres Based on Your Input:")
        st.write(", ".join(predicted_genres))

        st.write("### Recommended Movies:")
        for title, year in recommendations:
            st.write(f"- {title} ({year})")
    else:
        st.warning("Please provide at least one movie description.")
