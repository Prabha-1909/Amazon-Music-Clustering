import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():

    df = pd.read_csv("data/raw/single_genre_artists.csv")
    
    numeric_df = df.select_dtypes(include=['int64','float64'])
    
    drop_cols = ["popularity_songs", "popularity_artists", "followers"]
    numeric_df = numeric_df.drop(columns=drop_cols)
   
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    print("Data loaded successfully")
    print("Shape:", numeric_df.shape)

    return numeric_df, scaled_data

if __name__ == "__main__":
    load_and_preprocess()