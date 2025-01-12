#another implementation
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load dataset
file_path = "twitter.csv"
data = pd.read_csv(file_path, encoding="latin-1", header=None)
data.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Preprocessing
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS])  # Remove stopwords
    return text.lower()

data['cleaned_text'] = data['text'].apply(clean_text)

# Sampling the data to reduce size
sampled_data = data['cleaned_text'].sample(n=10000, random_state=42)

# Vectorization using TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
text_vectors = tfidf.fit_transform(sampled_data)

# Dimensionality reduction with TruncatedSVD
svd = TruncatedSVD(n_components=100, random_state=42)
reduced_vectors = svd.fit_transform(text_vectors)

# Clustering using K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(reduced_vectors)

# Evaluate K-Means clustering
kmeans_silhouette = silhouette_score(reduced_vectors, kmeans_labels)
print(f"K-Means Silhouette Score: {kmeans_silhouette:.2f}")

# Clustering using DBSCAN
dbscan = DBSCAN(eps=2.5, min_samples=5, metric='euclidean')
dbscan_labels = dbscan.fit_predict(reduced_vectors)

# Evaluate DBSCAN clustering
# Silhouette Score calculation for DBSCAN (ignoring noise points)
dbscan_silhouette = silhouette_score(
    reduced_vectors[dbscan_labels != -1],
    dbscan_labels[dbscan_labels != -1]
) if len(set(dbscan_labels)) > 1 else -1
print(f"DBSCAN Silhouette Score: {dbscan_silhouette:.2f}")

# Display cluster sizes
kmeans_cluster_sizes = pd.Series(kmeans_labels).value_counts()
dbscan_cluster_sizes = pd.Series(dbscan_labels).value_counts()

print("\nK-Means Cluster Sizes:\n", kmeans_cluster_sizes)
print("\nDBSCAN Cluster Sizes:\n", dbscan_cluster_sizes)
