import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def clustering(dataFrame, movies):
    DF = movies.drop(['title'], axis=1).select_dtypes(include=['number'])
    #Divido i film in 5 gruppi, perche da quanto visto con l'Elbow Method, è un numero equilibrato
    kmeans = KMeans(n_clusters=5, random_state=42)
    movies['Cluster'] = kmeans.fit_predict(DF)

def elbow_method(dataFrame, movies):
    sse = {}
    DF = movies.drop(['title'], axis=1).select_dtypes(include=['number'])

    for k in range(1, 30, 3):
        kmeans = KMeans(n_clusters=k, max_iter=100, random_state=42).fit(DF)
        sse[k] = kmeans.inertia_

    plt.figure(figsize=(8, 5))
    plt.plot(list(sse.keys()), list(sse.values()), marker='o')
    plt.title("Elbow Method per Clustering Film")
    plt.xlabel("Numero di cluster")
    plt.ylabel("SSE")
    plt.grid(True)
    plt.show()
