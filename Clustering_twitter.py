#Using Sentiment140 Dataset for clustering twitter tweets. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score      
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

#Step1: Load the dataset
#So we have to specify the columns here because there is no header 
columns = ['target', 'id', 'date', 'flag', 'user', 'text']
data = pd.read_csv("twitter.csv", encoding= 'latin-1', header= None, names= columns)
#print(data.head())
#print(data.shape)

#Now we need to cluster the tweets to have to focus on the text data
text_data = data['text']

# Step 2: Preprocessing the Text Data
# Remove stopwords, punctuation, and convert to lowercase as it 
# helps reduce noise, standardize the text, and improve the quality of the features for better model performance.
#This imports the predefined list of English stopwords from scikit-learn and the regular expression library re, 
# which are used for text preprocessing tasks like removing common words and cleaning the text data.

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

#re.sub() used to replace parts of a string that match a regular expression pattern with a specified replacement string
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    #that is first the text is split and all the words are then joined that are not in English stop words like is,etc, those does not help
    text =  text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])  # Remove stopwords
    return text

text_data_cleaned = text_data.apply(preprocess_text)
sampled_data = text_data_cleaned.sample(n=10000, random_state=42)

#Step3: Convert text to numerical data using TF-IDF (TF-IDF converts text data into numerical vectors where each word in the text is 
# represented by its TF-IDF score, capturing both its frequency and importance)
'''TF-IDF is better: It provides a weighted representation of the words, capturing their importance within each 
document, which is useful for tasks like clustering, where you need to evaluate the relevance of each word 
Purpose: It helps to identify important words in a document and reduces the weight of common, unimportant words like "the", "and", etc. 
It is used in text-based tasks like text clustering, classification, and information retrieval.
Advantage: Unlike word count, which might give high weight to common words across documents, TF-IDF assigns higher importance to words 
that are specific to a few documents. Term Frequency (TF): Measures how frequently a word appears in a document. 
Inverse Document Frequency (IDF): Measures how important a word is in the entire corpus. Words that appear in many documents are 
less important.'''

# Limit features for simplicity (limits the number of features (words) in the TF-IDF representation to the top 1000 most important ones)
vectorizer = TfidfVectorizer(max_features=1000) 
#It is needed to transform the textual data into numerical features (TF-IDF scores) that can 
# be used by machine learning algorithms for clustering 
#This line converts the cleaned text data into a sparse matrix of TF-IDF features and then converts it into a 
# dense numpy array for further processing, We need a dense NumPy array to store the TF-IDF features in a compact, efficient format 
#fit_transform is used here to both fit the TF-IDF model to the data (learning the word frequencies) and then transform the 
# text data into numerical feature vectors simultaneously.
text_vectors = vectorizer.fit_transform(sampled_data).toarray()

# Step 4: KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42) # Adjust 'n_clusters' as needed
#fit_predict is used to both fit the clustering model (e.g., KMeans) on the data and 
# simultaneously predict the cluster labels for each data point
kmeans_labels = kmeans.fit_predict(text_vectors)

# Evaluate KMeans using Silhouette Score
kmeans_silhouette = silhouette_score(text_vectors, kmeans_labels)
print(f"KMeans Silhouette Score: {kmeans_silhouette:.2f}") # :.2f is used to format the silhouette score to display only two decimal places for better readability.

# Step 5: DBSCAN Clustering (# Adjust parameters)
dbscan= DBSCAN(eps=0.5, min_samples=10, metric='euclidean') #The metric='euclidean' specifies that the Euclidean distance should be used t
dbscan_labels = dbscan.fit_predict(text_vectors)

# Filter noise (-1 indicates noise points)
'''The step dbscan_core_samples = sum(dbscan_labels != -1) counts the number of core samples in DBSCAN, where -1 indicates noise points. 
This step is important for evaluating how many points belong to clusters, as DBSCAN can mark some points as noise, 
and the number of core samples helps understand the density of the formed clusters. Unlike K-means, DBSCAN does not require 
the user to specify the number of clusters, and this step is useful to assess the clustering quality.'''

dbscan_core_samples = sum(dbscan_labels != -1)
print(f"DBSCAN Core Samples: {dbscan_core_samples}")
'''the silhouette score wasn't calculated for DBSCAN because it is typically used for algorithms like K-means, where each point 
belongs to a single cluster. In DBSCAN, some points are marked as noise (-1), and the silhouette score may not be meaningful for those points'''

# Step 6: Agglomerative Clustering
#linkage='ward' refers to the method used in hierarchical clustering to minimize the variance within clusters as 
# they are merged, aiming to create compact and homogeneous clusters.
agg = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
agg_labels = agg.fit_predict(text_vectors)

#Evaluate Agglomerative Clustering using Silhouette Score
agg_silhouette = silhouette_score(text_vectors, agg_labels)
print(f"Agglomerative Silhouette Score: {agg_silhouette:.2f}")

# Step 7: Visualize Clustering Results (Using PCA for Dimensionality Reduction)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
text_vectors_2d = pca.fit_transform(text_vectors)

plt.figure(figsize=(15, 5))
#The numbers in plt.subplot(1, 3, 1) indicate the grid layout for the subplots: 1 row, 3 columns, and this is the 1st subplot in that grid.
# KMeans
plt.subplot(1, 3, 1)
plt.scatter(text_vectors_2d[:, 0], text_vectors_2d[:, 1], c=kmeans_labels, cmap='viridis', s=10)
plt.title("KMeans Clustering")

# DBSCAN
plt.subplot(1, 3, 2)
plt.scatter(text_vectors_2d[:, 0], text_vectors_2d[:, 1], c=dbscan_labels, cmap='viridis', s=10)
plt.title("DBSCAN Clustering")

# Agglomerative Clustering
plt.subplot(1, 3, 3)
plt.scatter(text_vectors_2d[:, 0], text_vectors_2d[:, 1], c=agg_labels, cmap='viridis', s=10)
plt.title("Agglomerative Clustering")

plt.show()

# Step 8: Conclusion
print("Clustering Complete!")