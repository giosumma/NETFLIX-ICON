import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from termcolor import colored
import preprocessing

def print_classification_results(name, y_valid, predict, probabilities):
    print(colored(f"\n {name}", "cyan", attrs=["bold"]))
    print(colored(" Accuracy:", "yellow"), f"{accuracy_score(y_valid, predict):.3f}")
    print(colored(" AUC:", "yellow"), f"{roc_auc_score(y_valid, predict):.3f}")
    print(colored(" Confusion Matrix:", "yellow"))
    print(confusion_matrix(y_valid, predict))
    print(colored("\n Esempio di previsioni:", "green"))
    print("Reali     →", y_valid.reset_index(drop=True).iloc[:10].tolist())
    print("Predette  →", predict[:10].tolist())
    print("Probabilità →", probabilities[:10].round(3).tolist())

def various_classification():
    dataframe = pd.read_csv('../dataset/movies.csv')
    dataframe = preprocessing.preprocessing_for_classification(dataframe)

    features = ["releaseYear", "imdbAverageRating", "type", "genres", "imdbNumVotes"]
    training = dataframe.sample(frac=0.8, random_state=420)
    X_train = training[features]
    y_train = training['successful']
    X_test = dataframe.drop(training.index)[features]
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=420)

    logistic_regression(X_train, y_train, X_valid, y_valid)
    random_forest_classifier(X_train, y_train, X_valid, y_valid)
    k_neighbors_classifier(X_train, y_train, X_valid, y_valid)
    decision_tree_classifier(X_train, y_train, X_valid, y_valid)

def logistic_regression(X_train, y_train, X_valid, y_valid):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predict = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]
    print_classification_results("Logistic Regression", y_valid, predict, probabilities)

def random_forest_classifier(X_train, y_train, X_valid, y_valid):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predict = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]
    print_classification_results("Random Forest Classifier", y_valid, predict, probabilities)

def k_neighbors_classifier(X_train, y_train, X_valid, y_valid):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    predict = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]
    print_classification_results("K Neighbors Classifier", y_valid, predict, probabilities)

def decision_tree_classifier(X_train, y_train, X_valid, y_valid):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predict = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]
    print_classification_results("Decision Tree Classifier", y_valid, predict, probabilities)
