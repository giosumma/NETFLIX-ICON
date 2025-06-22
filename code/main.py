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
            while True:
                n_movies = input().strip()
                if n_movies.isdigit():
                    n_movies = int(n_movies)
                    break
                else:
                    print(colored("Inserisci un numero valido.","red"))

            data = pd.read_csv('../dataset/movies.csv', low_memory=False)
            _, movies = preprocessing.preprocessing_for_clustering(data)
            clustering.clustering(_, movies)

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

                while True:
                    model_input = input().strip()
                    if model_input.isdigit():
                        model_choice = int(model_input)
                        if model_choice in [1, 2, 3, 4]:
                            break
                        else:
                            print(colored("Scelta non valida. Inserisci un numero tra 1 e 4.","red"))
                    else:
                        print(colored("Inserisci un numero valido.","red"))

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
            print(colored("Scelta non valida","red"))

            
if __name__ == '__main__':
    main()