import os
from src.data_processing import read_data, preprocess_data
from src.visualization import plot_histograms, plot_categorical, plot_correlation
from src.model_training import create_dummies, scale_data, split_data, metrics_score, logistic_regression, svm_model
from src.logger import logger


def main():
    try:
        file_path = 'data/HR_Employee_Attrition.xlsx'
        output_dir = 'output_charts'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df = read_data(file_path)
        df, num_cols, cat_cols = preprocess_data(df)
        
        plot_histograms(df, num_cols, output_dir)
        plot_categorical(df, cat_cols, output_dir)
        plot_correlation(df, num_cols, output_dir)
        
        df = create_dummies(df, cat_cols)
        X_scaled, Y = scale_data(df)
        x_train, x_test, y_train, y_test = split_data(X_scaled, Y)
        
        # Logistic Regression
        lg = logistic_regression(x_train, y_train)
        y_pred_train = lg.predict(x_train)
        metrics_score(y_train, y_pred_train, output_dir, 'logistic_regression_train')
        y_pred_test = lg.predict(x_test)
        metrics_score(y_test, y_pred_test, output_dir, 'logistic_regression_test')
        
        # SVM Linear
        model_linear = svm_model(x_train, y_train, kernel='linear')
        y_pred_train_svm_linear = model_linear.predict(x_train)
        metrics_score(y_train, y_pred_train_svm_linear, output_dir, 'svm_linear_train')
        y_pred_test_svm_linear = model_linear.predict(x_test)
        metrics_score(y_test, y_pred_test_svm_linear, output_dir, 'svm_linear_test')
        
        # SVM RBF
        model_rbf = svm_model(x_train, y_train, kernel='rbf')
        y_pred_train_svm_rbf = model_rbf.predict(x_train)
        metrics_score(y_train, y_pred_train_svm_rbf, output_dir, 'svm_rbf_train')
        y_pred_test_svm_rbf = model_rbf.predict(x_test)
        metrics_score(y_test, y_pred_test_svm_rbf, output_dir, 'svm_rbf_test')
        
        # SVM Polynomial
        model_poly = svm_model(x_train, y_train, kernel='poly', degree=3)
        y_pred_train_svm_poly = model_poly.predict(x_train)
        metrics_score(y_train, y_pred_train_svm_poly, output_dir, 'svm_poly_train')
        y_pred_test_svm_poly = model_poly.predict(x_test)
        metrics_score(y_test, y_pred_test_svm_poly, output_dir, 'svm_poly_test')
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
