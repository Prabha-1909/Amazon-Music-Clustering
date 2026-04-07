## Amazon Music Clustering using Machine Learning
## Project Overview

- Music streaming platforms contain millions of songs, making it difficult to manually categorize them into genres or moods. This project uses unsupervised machine learning (K-Means clustering) to automatically group songs based on their audio characteristics.

- By analyzing features such as danceability, energy, loudness, tempo, and valence, the model identifies patterns and clusters songs with similar musical properties.

- These clusters can represent music moods, styles, or genres, helping improve recommendation systems and playlist generation.

## Project Objectives
- Perform Exploratory Data Analysis (EDA) on Amazon Music dataset
- Clean and preprocess song features
- Normalize numerical features using StandardScaler
- Apply K-Means Clustering to group similar songs
- Determine optimal clusters using the Elbow Method
- Evaluate cluster quality using Silhouette Score
- Visualize clusters using PCA (Principal Component Analysis)
- Interpret clusters as different music types

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PCA
- K-Means Clustering

## Project Structure
```
Amazon-Music-Clustering
│
├── data
│ ├── raw
│ │ └── single_genre_artists.csv 
│ │
│ └── processed
│ └── clustered_music_dataset.csv 
│
├── outputs
│ ├── cluster_visualization.png 
│ ├── elbow_plot.png 
│ └── heatmap.png 
│
├── src
│ ├── eda.py
│ ├── data_preprocessing.py 
│ ├── clustering.py 
│ └── visualization.py 
├── cluster_summary.csv 
│
├── requirements.txt 
├── README.md 
└── .gitignore
```

## How to Run the Project
- 1. Clone Repository
git clone https://github.com/yourusername/amazon-music-clustering.git
- 2. Install Required Libraries
pip install pandas numpy matplotlib seaborn scikit-learn
- 3. Run Data Preprocessing
python data_preprocessing.py
- 4. Run Clustering
python clustering.py
- 5. Run Visualization
python visualization.py

## Business Applications

- Personalized Playlist Creation

- Automatically group songs with similar sound characteristics.

- Music Recommendation Systems

- Suggest songs similar to the user's listening preferences.

- Artist Market Analysis

- Identify songs with similar musical patterns.

- Streaming Platform Insights

- Analyze listening trends and segment music catalogs.

## Key Learnings
- Data preprocessing for machine learning
- Feature scaling techniques
- Unsupervised learning with K-Means
- Cluster evaluation using Silhouette Score
- Dimensionality reduction using PCA
- Data visualization for cluster interpretation
