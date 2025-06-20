import pandas as pd
import graphics
import functions
import clustering
import prediction
import preprocessing
import os
from termcolor import colored

def main():
    graphics.print_logo()
    #first_time = True

    while True:
        graphics.print_menu()
        choice = input().strip()

        if choice == '1':
            graphics.movie_request()
            movie_title = input().strip()

            graphics.genre_request()
            preferred_genre = input().strip().capitalize()

            #n_movies = int(input())
            
            graphics.number_request()
            n_movies = input().strip()
            if not n_movies.isdigit():
                print("⚠️ Inserisci un numero valido.")
                return
            n_movies = int(n_movies)


            data = pd.read_csv('../dataset/movies.csv', low_memory=False)
            _, movies = preprocessing.preprocessing_for_clustering(data)
            clustering.clustering(_, movies)

            print(colored(f" RACCOMANDAZIONI BASATE SU '{movie_title}' (Genere: {preferred_genre}) ", "green", attrs=["bold"]))
            functions.recommend_movies(movie_title, preferred_genre, movies, n_movies)

            print("\n")
            #os.system("pause")
            first_time = False
            print("\n")


        elif choice == '2':
            graphics.movie_request_for_prediction()
            movie_title = input().strip()

            data = pd.read_csv('../dataset/movies.csv', low_memory=False)
            row = data.loc[data['title'] == movie_title]

            if row.empty:
                graphics.no_movie_matched()
            else:
                graphics.movie_matched()
                graphics.choose_classifier()
                model_choice = int(input())

                result = False

                match model_choice:
                    case 1:
                        if prediction.rfc_prediction(movie_title) == 0:
                            graphics.movie_not_popular()
                        else:
                            graphics.movie_is_popular()
                    case 2:
                        if prediction.knn_prediction(movie_title) == 0:
                            graphics.movie_not_popular()
                        else:
                            graphics.movie_is_popular()
                    case 3:
                        if prediction.dt_prediction(movie_title) == 0:
                            graphics.movie_not_popular()
                        else:
                            graphics.movie_is_popular()
                    case 4:
                        if prediction.lr_prediction(movie_title) == 0:
                            graphics.movie_not_popular()
                        else:
                            graphics.movie_is_popular()
                    

            print("\n")
            #os.system("pause")
            first_time = False
            print("\n")

    
        elif choice == '3':
            graphics.print_goodbye()
            break

        elif choice >= '4':
            print("Scelta non valida")

            
if __name__ == '__main__':
    main()