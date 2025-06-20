import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import preprocessing

def rfc_prediction(title):
    df_raw = pd.read_csv('../dataset/movies.csv')
    row_raw = df_raw.loc[df_raw['title'] == title]
    if row_raw.empty:
        print(" Film non trovato.")
        return False

    df = preprocessing.preprocessing_for_classification(df_raw)
    features = ["releaseYear", "imdbAverageRating", "type", "genres"]
    row = df.iloc[row_raw.index][features]

    X = df[features]
    y = df['successful']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=420)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    valid_pred = model.predict(X_valid)
    valid_proba = model.predict_proba(X_valid)[:, 1]
    acc = accuracy_score(y_valid, valid_pred)
    auc = roc_auc_score(y_valid, valid_proba)
    print(f"\n Accuracy: {acc:.3f}\n AUC: {auc:.3f}")

    prediction = model.predict(row)
    return bool(prediction[0])

def lr_prediction(title):
    df_raw = pd.read_csv('../dataset/movies.csv')
    row_raw = df_raw.loc[df_raw['title'] == title]
    if row_raw.empty:
        print(" Film non trovato.")
        return False

    df = preprocessing.preprocessing_for_classification(df_raw)
    features = ["releaseYear", "imdbAverageRating", "type", "genres"]
    row = df.iloc[row_raw.index][features]

    X = df[features]
    y = df['successful']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=420)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    valid_pred = model.predict(X_valid)
    valid_proba = model.predict_proba(X_valid)[:, 1]
    acc = accuracy_score(y_valid, valid_pred)
    auc = roc_auc_score(y_valid, valid_proba)
    print(f"\n Accuracy: {acc:.3f}\n AUC: {auc:.3f}")

    prediction = model.predict(row)
    return bool(prediction[0])

def knn_prediction(title):
    df_raw = pd.read_csv('../dataset/movies.csv')
    row_raw = df_raw.loc[df_raw['title'] == title]
    if row_raw.empty:
        print(" Film non trovato.")
        return False

    df = preprocessing.preprocessing_for_classification(df_raw)
    features = ["releaseYear", "imdbAverageRating", "type", "genres"]
    row = df.iloc[row_raw.index][features]

    X = df[features]
    y = df['successful']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=420)

    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    valid_pred = model.predict(X_valid)
    valid_proba = model.predict_proba(X_valid)[:, 1]
    acc = accuracy_score(y_valid, valid_pred)
    auc = roc_auc_score(y_valid, valid_proba)
    print(f"\n Accuracy: {acc:.3f}\n AUC: {auc:.3f}")

    prediction = model.predict(row)
    return bool(prediction[0])

def dt_prediction(title):
    df_raw = pd.read_csv('../dataset/movies.csv')
    row_raw = df_raw.loc[df_raw['title'] == title]
    if row_raw.empty:
        print(" Film non trovato.")
        return False

    df = preprocessing.preprocessing_for_classification(df_raw)
    features = ["releaseYear", "imdbAverageRating", "type", "genres"]
    row = df.iloc[row_raw.index][features]

    X = df[features]
    y = df['successful']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=420)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    valid_pred = model.predict(X_valid)
    valid_proba = model.predict_proba(X_valid)[:, 1]
    acc = accuracy_score(y_valid, valid_pred)
    auc = roc_auc_score(y_valid, valid_proba)
    print(f"\n Accuracy: {acc:.3f}\n AUC: {auc:.3f}")

    prediction = model.predict(row)
    return bool(prediction[0])