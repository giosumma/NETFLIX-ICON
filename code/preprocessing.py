import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

pd.options.mode.chained_assignment = None

def preprocessing_for_classification(dataframe):
    # Valori mancanti
    dataframe['imdbAverageRating'] = dataframe['imdbAverageRating'].fillna(dataframe['imdbAverageRating'].median())
    dataframe['imdbNumVotes'] = dataframe['imdbNumVotes'].fillna(0)
    dataframe['releaseYear'] = dataframe['releaseYear'].fillna(dataframe['releaseYear'].median())
    dataframe['type'] = dataframe['type'].fillna('Unknown')
    dataframe['genres'] = dataframe['genres'].fillna('Unknown')

    # Nuova definizione di successo: alta valutazione e abbastanza voti
    dataframe['successful'] = ((dataframe['imdbAverageRating'] >= 7.0) & (dataframe['imdbNumVotes'] >= 10000)).astype(int)

    # encoding
    encoder = OrdinalEncoder()
    dataframe[['type', 'genres']] = encoder.fit_transform(dataframe[['type', 'genres']].astype(str))

    return dataframe[['releaseYear', 'imdbAverageRating', 'type', 'genres', 'imdbNumVotes', 'successful']]

def preprocessing_for_clustering(dataframe):
    indx = dataframe[['title']]
    attributes = dataframe.copy()

    # Preprocessing base
    attributes['imdbAverageRating'] = attributes['imdbAverageRating'].fillna(attributes['imdbAverageRating'].median())
    attributes['imdbNumVotes'] = attributes['imdbNumVotes'].fillna(0)
    attributes['releaseYear'] = attributes['releaseYear'].fillna(attributes['releaseYear'].median())
    attributes['type'] = attributes['type'].fillna('Unknown')

    # Espansione dei generi multipli
    attributes['genres'] = attributes['genres'].fillna('Unknown')

    # Crea colonna 'genres_list'
    attributes['genres_list'] = attributes['genres'].apply(lambda x: [g.strip() for g in str(x).split(',')])

    # Estrarre tutti i generi unici
    all_genres = sorted({g for sublist in attributes['genres_list'] for g in sublist})

    # Colonne binarie per ciascun genere
    for genre in all_genres:
        attributes[f'genre_{genre}'] = attributes['genres_list'].apply(lambda g: int(genre in g))

    attributes = attributes.drop(columns=['genres_list'])

    # Encoding su 'type'
    encoder = OrdinalEncoder()
    attributes[['type']] = encoder.fit_transform(attributes[['type']].astype(str))

    # Se manca 'title', lo aggiunge
    if 'title' not in attributes.columns:
        attributes.insert(0, 'title', indx['title'])

    # Rimuovi duplicati e scala le feature numeriche
    genre_columns = [col for col in attributes.columns if col.startswith('genre_')]
    genres = attributes.groupby(['title'])[genre_columns].max().reset_index()
    attributes = attributes.drop(columns=genre_columns)

    atts_cols = attributes.drop(['title'], axis=1).select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    attributes[atts_cols] = scaler.fit_transform(attributes[atts_cols])

    # Merge finale
    movies = pd.merge(genres, attributes, on='title')
    movies = movies.drop_duplicates(['title']).reset_index(drop=True)

    return dataframe, movies

