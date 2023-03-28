"""Functions used across notebooks in one centralized source
of truth for improved maintenance"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay)


def preprocess(path):
    """Preprocess raw csv"""
    df = pd.read_csv(path).drop("Unnamed: 0", axis=1)
    return df


def get_features_labels(df, features):
    """Get feature matrix and labels"""
    X = df[features]
    y = df['cut'].values.flatten()
    return X, y


def fit_model(features, X_train, y_train, estimator, param_grid=None):
    """Fit classifier model"""

    cat_features = ['color', 'clarity']
    num_features = [f for f in features if f not in cat_features]

    ohe = ColumnTransformer([
        ("ohe_features", OneHotEncoder(), cat_features),
        ("scaled_num", StandardScaler(), num_features),
    ])

    pipe = Pipeline([
        ("ohe", ohe),
        ("est", estimator)
    ])

    if param_grid:
        model_gs = GridSearchCV(pipe, cv=3, param_grid=param_grid,
                                verbose=True)
        model_gs.fit(X_train, y_train)
        return model_gs
    else:
        pipe.fit(X_train, y_train)
        return pipe


def model_evaluation(model, X, y):
    print(classification_report(y, model.predict(X)))
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y, model.predict(X))
    cmp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    cmp.plot(ax=ax)
    plt.show()
    return


def run_model(path, features, estimator, param_grid=None):
    """Run classifier model starting from raw csv"""
    # Preprocess
    df = preprocess(path)
    X, y = get_features_labels(df, features)

    # Test train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if estimator.__class__.__name__ == 'XGBClassifier':
        y_train = LabelBinarizer().fit_transform(y_train)
        y_test = LabelBinarizer().fit_transform(y_test)

    model = fit_model(features, X_train, y_train, estimator, param_grid)

    # Evaluate
    if not estimator.__class__.__name__ == 'XGBClassifier':
        print('Results for test set:')
        model_evaluation(model, X_test, y_test)
    else:
        print(classification_report(y_test, model.predict(X_test)))

    return model


if __name__ == "__main__":
    pass
