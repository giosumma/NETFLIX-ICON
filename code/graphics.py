from termcolor import colored

def print_logo():
    print(colored("\nSISTEMA DI ANALISI E RACCOMANDAZIONE FILM\n", "cyan"))

def print_menu():
    """Stampa il menu principale delle operazioni disponibili."""
    print("\n" + "-" * 70)
    print(colored(" MENU PRINCIPALE ", "green", attrs=["bold"]))
    print("-" * 70)
    
    print("1. Generazione di raccomandazioni film basate su similarità")
    print("2. Previsione della popolarità di un film")
    print("3. Uscita")
    print("-" * 70)
    print("Selezionare l'opzione desiderata [1-3]:")

def print_goodbye():
    print(colored("Grazie per aver usato il sistema! Arrivederci!", "cyan"))

def movie_request():
    print(colored("Inserisci il nome di un film o una serie TV che ti è piaciuto/a:", "cyan"))

def genre_request():
    print(colored("Inserisci il tuo genere preferito (es. Action, Comedy, Drama):", "cyan"))

def number_request():
    print(colored("Quanti film simili vuoi vedere?:", "cyan"))

def movie_request_for_prediction():
    print(colored("Inserisci il titolo del film di cui vuoi sapere la popolarità:", "cyan"))


def choose_classifier():
    print(colored("Scegli un modello di classificazione:\n"
                  + "1 - Random Forest\n"
                  + "2 - K-Nearest Neighbors\n"
                  + "3 - Decision Tree\n"
                  + "4 - Logistic Regression", "cyan"))

def no_movie_matched():
    print(colored("[ERRORE] Il film specificato non è stato trovato nel dataset.", "red"))

def movie_matched():
    print(colored("[INFO] Film identificato nel dataset. Procedere con la selezione del classificatore", "green"))

def movie_is_popular():
    print(colored("[RISULTATO PREVISIONE] Il film è classificato come POPOLARE", "green"))

def movie_not_popular():
    print(colored("[RISULTATO PREVISIONE] Il film è classificato come NON POPOLARE", "red"))