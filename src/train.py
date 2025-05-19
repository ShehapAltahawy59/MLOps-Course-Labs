"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import pickle
import joblib

### Import MLflow
import mlflow
import mlflow.sklearn
def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())
   

    # Log the transformer as an artifact
    with open("./transformer.pkl", "wb") as f:
        pickle.dump(col_transf, f)

    # Log it as an artifact to MLflow
    mlflow.log_artifact("./transformer.pkl")

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train,model,max_iter=1000):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    
    model.fit(X_train, y_train)
    
    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = mlflow.models.infer_signature(X_train, y_train)
    

    # Log model
    mlflow.sklearn.log_model( model,
        artifact_path="SVM",
        signature=signature,
        input_example=X_train)

    ### Log the data
    mlflow.log_artifact("dataset/Churn_Modelling.csv")

    return model


def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")

    ### Set the experiment name
    mlflow.set_experiment("bank_churn_prediction")
 
    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run(run_name="Logistic Regression"):

        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        ### Log the max_iter parameter
        mlflow.log_param("max_iter",1000)

        log_reg = LogisticRegression(max_iter=1000)

        model = train(X_train, y_train , log_reg)

        
        y_pred = model.predict(X_test)
        

        ### Log metrics after calculating them
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred)) 
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))


        ### Log tag
        mlflow.set_tag("model_type", "Logistic Regression")

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        
        # Log the image as an artifact in MLflow
        plt.savefig("confusion_matrix_LR.png")
        mlflow.log_artifact("confusion_matrix_LR.png")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Show the plot
        plt.show()
        
        joblib.dump(model, "./Models/LogisticRegression.pkl")

    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run(run_name="Support Vector Machine"):

        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        ### Log the max_iter parameter
        mlflow.log_param("max_iter",1000)

        svm_model = SVC(max_iter=1000)

        model = train(X_train, y_train , svm_model)

        
        y_pred = model.predict(X_test)

        ### Log metrics after calculating them
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred)) 
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))


        ### Log tag
        mlflow.set_tag("model_type", "Support Vector Machine")

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        
        # Log the image as an artifact in MLflow
        plt.savefig("confusion_matrix_SVC.png")
        mlflow.log_artifact("confusion_matrix_SVC.png")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Show the plot
        plt.show()
        joblib.dump(model, "./Models/svm.pkl")

    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run(run_name="Random Forest"):

        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        ### Log the max_iter parameter
        #mlflow.log_param("max_iter",1000)

        random_for = RandomForestClassifier()

        model = train(X_train, y_train , random_for)

        
        y_pred = model.predict(X_test)

        ### Log metrics after calculating them
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred)) 
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))


        ### Log tag
        mlflow.set_tag("model_type", "Random Forest")

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        
        # Log the image as an artifact in MLflow
        plt.savefig("confusion_matrix_RF.png")
        mlflow.log_artifact("confusion_matrix_RF.png")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Show the plot
        plt.show()
        joblib.dump(model, "./Models/RandomForest.pkl")

    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run(run_name="GBOOST"):

        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        ### Log the max_iter parameter
        #mlflow.log_param("max_iter",1000)

        gboost = GradientBoostingClassifier()

        model = train(X_train, y_train , gboost)

        
        y_pred = model.predict(X_test)

        ### Log metrics after calculating them
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred)) 
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))


        ### Log tag
        mlflow.set_tag("model_type", "GBOOST")

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        
        # Log the image as an artifact in MLflow
        plt.savefig("confusion_matrix_GBOOST.png")
        mlflow.log_artifact("confusion_matrix_GBOOST.png")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Show the plot
        plt.show()
        joblib.dump(model, "./Models/GBOOST.pkl")
        



if __name__ == "__main__":
    main()
