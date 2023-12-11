import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
import recmetrics

# load datasetrecipes = treat(recipes)
recipes = pd.read_csv('data/recipes.csv')


def treat_missing_values(dataframe):
    '''
    Identifies and treats missing values depending on column name
    '''
    # replaces all empty string with Null values
    dataframe.replace(" ", None, inplace=True)

    # tests for missing values
    if dataframe.isna().sum().sum() > 0:

        # finds where null values are in the dataframe
        for column in dataframe.columns:
            # checks column to make appropriate changes
            if column == "cuisine":
                dataframe[column].fillna("Unknown cuisine", inplace=True)

            if column == "category":
                dataframe[column].fillna("Unknown category", inplace=True)

    # Find duplicates in title column
    duplicates = dataframe['title'].duplicated(keep=False)

    # Add suffix to duplicate titles
    counter = 1
    for title in dataframe.loc[duplicates, 'title'].unique():
        for index in dataframe[dataframe['title'] == title].index:
            dataframe.loc[index, 'title'] = title + ' ' + str(counter)
            counter += 1

    return dataframe


def show_summary():
    '''
    shows the number of recipes in the dataset, the smallest and the highest rating, and the average rating
    '''

    print("Summary Statistics: ")
    print("Total recipes in the dataset:", statistics.loc['count'][1])
    print("Smallest Rating:", statistics.loc['min'][2])
    print("Best Rating:", statistics.loc['max'][2])
    print("Average Rating:", round(statistics.loc['mean'][2], 2))


def show_top10():
    '''
    shows the 10 highest rated recipes (rating_val) and their average rating (rating_avg)
    '''

    print("Top 10 Recipes with the highest rating value: ")
    top_10 = recipes.sort_values(by=['rating_val'], ascending=False).head(10)
    for i in range(10):
        print("Recipe", i+1, ":", "Title:", top_10.iloc[i]['title'], ": Rating Average:",
              top_10.iloc[i]['rating_avg'], ": Rating Value:", top_10.iloc[i]['rating_val'])


def show_rating_distribution():
    '''
    distribution of the number of ratings (rating_val) and the average rating (rating_avg) for each recipe.
    '''

    grouped_data = recipes.groupby(['rating_val']).mean(
        numeric_only=True)['rating_avg']
    average_rating = recipes['rating_avg']
    number_of_ratings = recipes['rating_val']

    plt.figure(figsize=(12, 6))
    plt.scatter(number_of_ratings, average_rating, alpha=0.5)
    plt.xlabel('Number of Ratings')
    plt.ylabel('Average Rating Value')
    plt.title('Average Rating for each Number of Ratings (Not Grouped)')
    plt.show()

    plt.scatter(grouped_data.index, grouped_data.values)
    plt.xlabel('Number of Ratings')
    plt.ylabel('Average Rating Value')
    plt.title('Average Rating for each Number of Ratings (Grouped)')
    plt.show()

    # Based on the scatter plot we can see a weak negative relationship between the number of ratings
    # and the Rating average this means that generally recipes with a higher amount of ratings tend to
    # have a lower average rating. However this is hard to determine since many of the recipes have below
    # 100 reviews. I suggest that the threshhold for the number of ratings should be above 10 as in the
    # grouped data graph most data is above the 10 threshold.


def combine_features_as_strings(row):
    """
    adds a column with a list of features to be used in the recommendation engine
    """
    return row['title']+" "+str(row['rating_avg'])+" "+str(row['rating_val'])+" "+str(row['total_time'])+" "+row['category']+" "+row['cuisine']+" "+row['ingredients']


def vec_space_method(recipe_title, k):
    # Combine features into a single string
    recipes["combine_features"] = recipes.apply(
        combine_features_as_strings, axis=1)

    # Create a TfidfVectorizer with support for both words and numbers using a regular expression
    # The regular expression will match any word or number of length 1 or more
    vectorizer = TfidfVectorizer(token_pattern=r'([a-zA-Z0-9-/]{1,})')

    # Create a matrix of vectors that take into account the words and numbers
    tfidf_matrix = vectorizer.fit_transform(recipes["combine_features"])

    # Compute the cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)

    # Get the index of the recipe that matches the title
    try:
        recipe_index = recipes[recipes.title == recipe_title].index[0]
    except:
        # If the recipe is not found, find the closest match
        recipe_index = recipes[recipes.title.str.contains(
            recipe_title)].index[0]
        print(f"Did you mean {recipes.iloc[recipe_index]['title']}?")

    # Create a list of enumerations for the similarity scores
    similar_recipes = list(enumerate(cosine_sim_matrix[recipe_index]))

    # Sort the list of similar recipes in descending order
    sorted_similar_recipes = sorted(
        similar_recipes, key=lambda x: x[1], reverse=True)

    # Print the top k similar recipes
    similar_recipes = []
    print(f"Top {k} similar recipes to {recipe_title}:")
    for i in range(1, k+1):
        # Print the recipe title and similarity score
        print(
            f"Recipe {i}: {recipes.iloc[sorted_similar_recipes[i][0]]['title']} (Score: {sorted_similar_recipes[i][1]:.2f})")
        similar_recipes.append(recipes.iloc[sorted_similar_recipes[i][0]]['title'])

    return similar_recipes


def knn_similarity(query, k):
    '''
    Uses KNN similarity to find the similar recipes to the target
    '''
    # Create a TfidfVectorizer with support for both words and numbers using a regular expression
    # The regular expression will match any word or number of length 1 or more

    vectorizer = TfidfVectorizer(token_pattern=r'([a-zA-Z0-9-/]{1,})')

    # Create a matrix of vectors that take into account the words and numbers
    tfidf_matrix = vectorizer.fit_transform(recipes["combine_features"])

    # Get the index of the recipe that matches the title
    try:
        recipe_index = recipes[recipes.title == query].index[0]  # ERROR
    except:
        # If the recipe is not found, find the closest match
        recipe_index = recipes[recipes.title.str.contains(query)].index[0]
        print(f"Did you mean {recipes.iloc[recipe_index]['title']}?")

    # Get the feature vector of the recipe
    recipe_vector = vectorizer.transform([query]).reshape(1, -1)

    # Compute KNN similarity based on the feature matrix
    knnmodel = NearestNeighbors()
    knnmodel.fit(tfidf_matrix)
    dist, ind = knnmodel.kneighbors(recipe_vector, n_neighbors=20)

    # Print the top k similar recipes and their distances
    similar_recipes = []
    print(f"Top {k} similar recipes to {query}:")
    for i in range(1, k+1):
        print(
            f"Recipe {i}: {recipes.iloc[ind[0][i]]['title']}, Distance: {dist[0][i]:.2f}")

        similar_recipes.append(recipes.iloc[ind[0][i]]['title'])

    return similar_recipes


def predictive_model(query):
    # Preprocess the data

    # Only uses recipes with 10 or more ratings as they are considered significant
    significant_ratings = recipes[recipes['rating_val'] >= 10]

    # Sets the inputs and outputs of the predictive model
    x = significant_ratings['title']
    y = significant_ratings['tasty']

    # Creates a vectorizer and assigns numerical value to title
    vectorizer = CountVectorizer()
    X_vect = vectorizer.fit_transform(x)

    # Trains KNN classifier with k=5 (default value)
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_vect, y)

    # Uses the vectorizer to assign numerical value to input string
    query_vect = vectorizer.transform([query])

    # Uses the model to make a prediction
    prediction = knn_model.predict(query_vect)[0]

    # Prints the accuracy of the model
    print(f"Accuracy: {knn_model.score(X_vect, y)*100:.2f}%")

    # returns wether the query dish is tasty
    if prediction == 1:
        return (query, "Tasty")
    else:
        return (query, "Not Tasty")


def print_top_10(sorted_similar_recipes, recipe_index):
    """ 
    print the titles of the first 10 recipes#
    """
    print("Top 10 similar recipes to " + recipes.title[recipe_index] + ": ")
    for i in range(1, 11):
        print(
            f"Recipe {i}: {recipes.iloc[sorted_similar_recipes[i][0]].title} (Score: {sorted_similar_recipes[i][1]:.2f})")


def tasty_detector(Avg_rating):
    if Avg_rating > 4.2:
        return 1
    else:
        return -1


if __name__ == "__main__":
    # show the summary statistics
    print("Part 1 (Q1): ")
    recipes = treat_missing_values(recipes)
    statistics = recipes.describe()
    show_summary()
    print("")
    show_top10()
    print("")

    print("Part 1 (Q2) - Graphing Ratings: ")
    show_rating_distribution()
    print("")

    print("Part 1 (Q3) - Calculating cosine similarity using a string of all features: \n")
    # creates a column for every row with a str of all features
    recipes["combine_features"] = recipes.apply(
        combine_features_as_strings, axis=1)

    # create count matrix from the new combined column
    count_vectorizer = CountVectorizer()

    cv = count_vectorizer.fit_transform(recipes["combine_features"])

    # compute the Cosine Similarity based on the count_matrix
    cosine_sim_matrix = cosine_similarity(cv, cv)

    # get the index of the recipe that matches the title
    recipe_index = recipes[recipes.title ==
                           "Chicken and coconut curry"].index[0]

    # create a list of enumerations for the similarity scores
    similar_recipes = list(enumerate(cosine_sim_matrix[recipe_index]))

    # sort the list of similar recipes in descending order
    sorted_similar_recipes = sorted(
        similar_recipes, key=lambda x: x[1], reverse=True)
    print_top_10(sorted_similar_recipes, recipe_index)
    print("")

    print("Part 2 (Q4) - Calculating cosine similarity using integers and strings: \n")
    # returns recipe recommendations for the recipe "chicken and coconut curry" using the vector space method
    vec_space_method("Chicken and coconut curry", 10)
    print("")

    print("Part 2 (Q5) - Calculating KNN similairty: \n")
    # returns recipe recommendations for the recipe "Fluffy American pancakes" using KNN
    knn_similarity("Chicken and coconut curry", 10)
    print("")

    print("Part 2 (Q6) - Testing and evaluations")
    test_array = ["Chicken tikka masala",
                  "Albanian baked lamb with rice",
                  "Baked salmon with chorizo rice",
                  "Almond lentil stew"]
    
    recommendations_vec_space = []
    recommendations_knn = []

    for item in test_array:
        print("User", test_array.index(item)+1, "\n")
        print("Cosine_similarity:")
        recommendations_vec_space.append(vec_space_method(item, 10))
        print("\nKNN similarity:")
        recommendations_knn.append(knn_similarity(item, 10))
        print("--------------------------")

    print("")

    # use recmetrics to evaluate the recommendations
    print("Evaluating the recommendations...")
    print("")
    catalog = recipes.title.unique().tolist()

    print(f"Personalisation (Cosine similarity): {recmetrics.personalization(recommendations_vec_space)*100}%")
    print(f"Coverage (Cosine similarity): {recmetrics.prediction_coverage(recommendations_vec_space, catalog)}\n")
    print(f"Personalisation (KNN): {recmetrics.personalization(recommendations_knn)*100}%")
    print(f"Coverage (KNN): {recmetrics.prediction_coverage(recommendations_knn, catalog)}\n")

    # EVALUATION OF THE RECOMMENDER SYSTEMS 
    """
    KNN and Cosine similarity both have a high personalisation score, which means that the recommendations are personalised to the user.
    The coverage score for both is also high, which means that the recommendations are diverse and cover a wide range of recipes. 
    The test data provided was not very diverse, so the coverage score is not very high. This can be said for both KNN and Cosine similarity.
    This leads to similar scores for both KNN and Cosine similarity. 

    We have tested this with a few different test cases. For instance, if 2 users have similar 
    interests and tastes, the recommendations will be similar.

    If we replace one of the tests with Chicken and coconut curry, 
    the personalisation score for KNN is 98.3 and the coverage is 1.18.
    """

    print("Part 2 (Q7) - Predictive model")
    print("Computing the average rating for each recipe...")
    recipes["tasty"] = recipes["rating_avg"].apply(tasty_detector)
    prediction = predictive_model("Vegan goat curry")
    print(f"\"{prediction[0]}\" is {prediction[1].lower()}")
