"""Functions used across notebooks in one centralized source
of truth for improved maintenance"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics


def preprocess(path):
    """Preprocess raw csv"""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df


def get_features_labels(df, features):
    """Get feature matrix and labels"""
    X = df[features]
    y = LabelEncoder().fit_transform(df["Churn"])
    return X, y


def fit_model(features, X_train, y_train, estimator, param_grid=None):
    """Fit classifier model"""

    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_features = [f for f in features if f not in num_features]

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


def model_evaluation(model, X, y, test_train):
    """Compute and display model's score metrics"""
    y_pred = model.predict(X)
    scores = {}
    scores['accuracy'] = round(metrics.accuracy_score(y, y_pred), 4)
    scores['precision'] = round(metrics.precision_score(y, y_pred), 4)
    scores['recall'] = round(metrics.recall_score(y, y_pred), 4)
    probs = model.predict_proba(X).T[1]
    precisions, recalls, thresholds = metrics.precision_recall_curve(y, probs)
    scores['area under precision-recall curve'] = round(
        metrics.auc(recalls, precisions), 4)

    # Print scores
    for metric, score in scores.items():
        if test_train == 'test':
            print(f'Test {metric}: {score}')
        elif test_train == 'train':
            print(f'Train {metric}: {score}')
        else:
            raise TypeError("Must pass either 'test' or 'train'")

    # print(f"Model weights:\n{model.named_steps['est'].coef_}")

    return scores


def run_model(path, features, estimator, param_grid=None):
    """Run classifier model starting from raw csv"""
    # Preprocess
    df = preprocess(path)
    X, y = get_features_labels(df, features)

    # Test train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = fit_model(features, X_train, y_train, estimator, param_grid)

    # Evaluate
    # if not estimator.__class__.__name__ == 'SVC':
    #     model_evaluation(model, X_test, y_test, 'test')
    #     model_evaluation(model, X_train, y_train, 'train')
    # else:
    #     print(metrics.classification_report(y_test, model.predict(X_test)))

    print(metrics.classification_report(y_test, model.predict(X_test)))

    return model


if __name__ == "__main__":
    pass
