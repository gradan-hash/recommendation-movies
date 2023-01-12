import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# data collection and preprocessing

movies_data = pd.read_csv("movies.csv")

# print(movies_data.columns)

# feature selection
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
# print(selected_features)

# replace the null values with null string

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna(" ")

# combining all the selected feature
combined_features = movies_data['genres'] + " " + movies_data['keywords'] + " " + movies_data['tagline'] + " " + \
                    movies_data['cast'] + " " + movies_data['director']
# print(combined_features)


# converting the text(combined data) data to feature vectors
vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

# print(feature_vectors)


# getting cosine score using cosine cosine_similarity
similarity = cosine_similarity(feature_vectors)
# print(similarity)

# GETTING THE MOVIE NAME FROM THE USER
movie_name = input("Enter Your Favourite Movie Name: ")

# creating a list with all the movies with all the movies given in the dataset
list_of_all_titles = movies_data['title'].tolist()
# print(list_of_all_titles)

# finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
# print(find_close_match)

close_match = find_close_match[0]
print(close_match)

# finding the index of the movie with tittle
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
# print(index_of_the_movie)

# getting a list of similar movies by index easy way
similarity_score = list(enumerate(similarity[index_of_the_movie]))
# print(similarity_score)

# sorting the movies based on similarity score
sorted_similar_list = sorted(similarity_score, key=lambda x: x[1], reverse=True)
# print(sorted_similar_list)

# print the name of similar movies based on the index
print("Movies suggested for you : \n ")

i = 1
for movie in sorted_similar_list:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if i < 30:
        print(i, " .", title_from_index)
        i += 1
