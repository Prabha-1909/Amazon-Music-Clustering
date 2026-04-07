Amazon Music Clustering using Machine Learning
Project Overview

Music streaming platforms contain millions of songs, making it difficult to manually categorize them into genres or moods. This project uses unsupervised machine learning (K-Means clustering) to automatically group songs based on their audio characteristics.

By analyzing features such as danceability, energy, loudness, tempo, and valence, the model identifies patterns and clusters songs with similar musical properties.

These clusters can represent music moods, styles, or genres, helping improve recommendation systems and playlist generation.

Project Objectives
Perform Exploratory Data Analysis (EDA) on Amazon Music dataset
Clean and preprocess song features
Normalize numerical features using StandardScaler
Apply K-Means Clustering to group similar songs
Determine optimal clusters using the Elbow Method
Evaluate cluster quality using Silhouette Score
Visualize clusters using PCA (Principal Component Analysis)
Interpret clusters as different music types

Dataset Information

The dataset contains audio features of songs that describe their musical characteristics.

Main Audio Features
Feature	Description
danceability	How suitable a track is for dancing
energy	Intensity and activity level
loudness	Overall loudness of a track
speechiness	Presence of spoken words
acousticness	Likelihood of acoustic sound
instrumentalness	Probability of instrumental music
liveness	Detects live audience presence
valence	Musical positivity / mood
tempo	Beats per minute
duration_ms	Length of the track
Metadata Columns

These columns are used only for reference and not for clustering.

track_id
track_name
artist_name

Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
PCA
K-Means Clustering

Project Workflow

Data Exploration

Initial analysis of dataset:

Dataset shape
Column names
Data types
Missing values
Duplicate rows
Statistical summary

Example code:

print(df.shape)
print(df.head())
print(df.isnull().sum())
print(df.describe())

Data Preprocessing

Steps performed:

Load dataset
Select numeric features
Remove irrelevant columns

Removed columns:

popularity_songs
popularity_artists
followers
Apply StandardScaler for feature normalization.

Why scaling?

Clustering algorithms use distance calculations, so features must be on the same scale.

Clustering (K-Means)

K-Means clustering was applied to group songs based on their audio features.

Elbow Method

Used to determine the optimal number of clusters (K).

The algorithm calculates inertia (SSE) for multiple values of K.

The "elbow point" suggests the best number of clusters.

Output saved:

outputs/elbow_plot.png
4️⃣ Cluster Evaluation

Cluster quality was evaluated using:

Silhouette Score

Measures how similar a song is to its own cluster compared to other clusters.

Higher score → Better clustering.

5️⃣ Cluster Interpretation

Clusters were interpreted based on average feature values.

Cluster labels assigned:

Cluster	Music Type
0	Chill Songs
1	Party Tracks
2	Melody Songs
3	Energetic Songs

Cluster summaries were saved as:

cluster_summary.csv
clustered_music_dataset.csv
📈 Visualization
PCA Cluster Visualization

Principal Component Analysis (PCA) was used to reduce dimensions to 2 components for visualization.

Output saved:

outputs/cluster_visualization.png

This plot shows how songs are grouped into clusters.

Cluster Feature Heatmap

A heatmap was created to compare feature averages across clusters.

Helps understand how clusters differ in terms of:

energy
danceability
tempo
acousticness
valence
📂 Project Structure
Amazon-Music-Clustering
│
├── data
│   └── raw
│       └── single_genre_artists.csv
│
├── outputs
│   ├── elbow_plot.png
│   └── cluster_visualization.png
│
├── data_preprocessing.py
├── clustering.py
├── visualization.py
│
├── cluster_summary.csv
├── clustered_music_dataset.csv
│
└── README.md

How to Run the Project
1️⃣ Clone Repository
git clone https://github.com/yourusername/amazon-music-clustering.git
2️⃣ Install Required Libraries
pip install pandas numpy matplotlib seaborn scikit-learn
3️⃣ Run Data Preprocessing
python data_preprocessing.py
4️⃣ Run Clustering
python clustering.py
5️⃣ Run Visualization
python visualization.py

Business Applications
Personalized Playlist Creation

Automatically group songs with similar sound characteristics.

Music Recommendation Systems

Suggest songs similar to the user's listening preferences.

Artist Market Analysis

Identify songs with similar musical patterns.

Streaming Platform Insights

Analyze listening trends and segment music catalogs.

Key Learnings
Data preprocessing for machine learning
Feature scaling techniques
Unsupervised learning with K-Means
Cluster evaluation using Silhouette Score
Dimensionality reduction using PCA
Data visualization for cluster interpretation