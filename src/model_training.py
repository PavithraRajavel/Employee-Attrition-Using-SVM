import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from src.logger import logger


def create_dummies(df, cat_cols):
    try:
        to_get_dummies_for = ['BusinessTravel', 'Department','Education', 'EducationField','EnvironmentSatisfaction', 
                              'Gender', 'JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus']
        df = pd.get_dummies(data=df, columns=to_get_dummies_for, drop_first=True)
        
        dict_OverTime = {'Yes': 1, 'No':0}
        dict_attrition = {'Yes': 1, 'No': 0}
        df['OverTime'] = df.OverTime.map(dict_OverTime)
        df['Attrition'] = df.Attrition.map(dict_attrition)
        
        logger.info("Dummy variables created successfully")
        return df
    except Exception as e:
        logger.error(f"Error creating dummy variables: {e}")
        raise


def scale_data(df):
    try:
        Y = df.Attrition
        X = df.drop(columns=['Attrition'])
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        logger.info("Data scaling completed successfully")
        return X_scaled, Y
    except Exception as e:
        logger.error(f"Error scaling data: {e}")
        raise


def split_data(X_scaled, Y):
    try:
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)
        logger.info("Data split into train and test sets successfully")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise


def metrics_score(actual, predicted, output_dir, model_name):
    try:
        print(classification_report(actual, predicted))
        cm = confusion_matrix(actual, predicted)
        plt.figure(figsize=(8,5))
        sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()
        logger.info(f"Confusion matrix for {model_name} saved successfully")
    except Exception as e:
        logger.error(f"Error saving confusion matrix for {model_name}: {e}")
        raise


def logistic_regression(x_train, y_train):
    try:
        lg = LogisticRegression()
        lg.fit(x_train, y_train)
        logger.info("Logistic regression model trained successfully")
        return lg
    except Exception as e:
        logger.error(f"Error training logistic regression model: {e}")
        raise


def svm_model(x_train, y_train, kernel='linear', degree=3):
    try:
        svm = SVC(kernel=kernel, degree=degree)
        model = svm.fit(X=x_train, y=y_train)
        logger.info(f"SVM model with {kernel} kernel trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training SVM model with {kernel} kernel: {e}")
        raise
