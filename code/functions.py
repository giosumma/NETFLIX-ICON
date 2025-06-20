import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from termcolor import colored

# Funzione che dato un film, genera una lista di film simili
def recommend_movies(title, preferred_genre, movies, n_movies=10):
    similar_movies = find_similar_movies(title, preferred_genre, movies, n_movies)

    if similar_movies is not None and not similar_movies.empty:
        print(colored(f'🎬 Raccomandazioni basate su "{title}" e genere "{preferred_genre}"', "green"))
        print()
        for i in range(len(similar_movies)):
            name = similar_movies.title[i]
            score = similar_movies.similarity[i]
            print(colored(f"{name} — Similarità: {score:.4f}", "green"))
        print()

        # 🔽 GRAFICO
        #plt.figure(figsize=(8, 5))
        #plt.bar(similar_movies.title, similar_movies.similarity, color='skyblue')
        #plt.xticks(rotation=45, ha='right')
        #plt.xlabel("Titoli consigliati")
        #plt.ylabel("Punteggio di similarità")
        #plt.title(f"Similarità con '{title}'")
        #plt.tight_layout()
        #plt.show()

    else:
        print("❌ Nessun film trovato che corrisponda al titolo e al genere preferito.")



# Funzione che dato un film, ne trova simili
def find_similar_movies(title, preferred_genre, movies, top_n=5):
    genre_col = f"genre_{preferred_genre}"

    if genre_col not in movies.columns:
        print(f"⚠️ Genere '{preferred_genre}' non trovato nel database.")
        return None

    # Preparare il dataset
    database = movies.copy()
    database['title_clean'] = database['title'].str.strip().str.lower()
    title_clean = title.strip().lower()

    selected_movie = database[database['title_clean'] == title_clean]
    if selected_movie.empty:
        suggestions = database[database['title_clean'].str.contains(title_clean[:5])]['title'].unique()
        print(f"⚠️ Film '{title}' non trovato. Forse cercavi uno di questi?")
        for s in suggestions[:5]:
            print(f" - {s}")
        return None

    index = selected_movie.index[0]

    # Similarità
    indx_names = database[['title', 'Cluster']].copy()
    indx_names['title_clean'] = database['title_clean']
    movies_features = database.drop(columns=['title', 'Cluster']).select_dtypes(include=['number'])

    cos_dists = cosine_similarity(movies_features, movies_features)
    indx_names['similarity'] = cos_dists[index]

    # Filtra per genere preferito
    filtered = indx_names[database[genre_col] == 1].copy()

    # Esclude il film stesso con confronto pulito
    filtered = filtered[filtered['title_clean'] != title_clean]

    # Ordina per similarità
    filtered = filtered.sort_values(by='similarity', ascending=False)
    return filtered.head(top_n).reset_index(drop=True)
