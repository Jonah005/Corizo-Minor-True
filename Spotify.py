# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the dataset
spotify_data = pd.read_csv('spotify_dataset.csv')

# Data preprocessing
# Drop irrelevant columns
spotify_data.drop(columns=['track_id', 'track_name', 'track_artist', 'track_album_id', 'track_album_name', 'track_album_release_date'], inplace=True)

# Drop duplicate rows
spotify_data.drop_duplicates(inplace=True)

# Convert categorical variables to numerical
spotify_data = pd.get_dummies(spotify_data, columns=['playlist_genre', 'playlist_subgenre'])

# Splitting features and target variable
X = spotify_data.drop(columns=['playlist_name', 'playlist_id'])
y = spotify_data['playlist_name']

# Data analysis and visualization
# Pairplot for visualizing relationships between variables
sns.pairplot(spotify_data, hue='playlist_name', diag_kind='kde')
plt.show()

# Correlation matrix
correlation_matrix = spotify_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
plt.title('Correlation Matrix')
plt.show()

# Clustering based on playlist genres
genre_clusters = KMeans(n_clusters=5, random_state=42)
genre_clusters.fit(X)
spotify_data['genre_cluster'] = genre_clusters.labels_

# Visualizing clusters with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=spotify_data['genre_cluster'], palette='Set1')
plt.title('Clustering based on Playlist Genres (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Clustering based on playlist names
name_clusters = KMeans(n_clusters=5, random_state=42)
name_clusters.fit(X)
spotify_data['name_cluster'] = name_clusters.labels_

# Visualizing clusters with t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=spotify_data['name_cluster'], palette='Set2')
plt.title('Clustering based on Playlist Names (t-SNE)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

# Model building - This can be your recommendation system based on user input
# For simplicity, let's assume a user provides a playlist name and we recommend similar playlists based on clustering

def recommend_playlist(playlist_name):
    similar_playlists = spotify_data[spotify_data['playlist_name'] == playlist_name]['name_cluster'].iloc[0]
    recommendations = spotify_data[spotify_data['name_cluster'] == similar_playlists]['playlist_name'].unique()
    print("You might like playlists similar to", playlist_name, ":")
    for playlist in recommendations:
        if playlist != playlist_name:
            print("-", playlist)

# Example usage of recommendation system
recommend_playlist('Pop Remix')
