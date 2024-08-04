import pandas as pd
import os
from src.logger import logger


def read_data(file_path):
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Data read successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading data from {file_path}: {e}")
        raise


def preprocess_data(df):
    try:
        df = df.drop(['EmployeeNumber','Over18','StandardHours'], axis=1)
        num_cols = ['DailyRate','Age','DistanceFromHome','MonthlyIncome','MonthlyRate','PercentSalaryHike',
                    'TotalWorkingYears','YearsAtCompany','NumCompaniesWorked','HourlyRate',
                    'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager','TrainingTimesLastYear']
        
        cat_cols = ['Attrition','OverTime','BusinessTravel', 'Department','Education', 'EducationField','JobSatisfaction',
                    'EnvironmentSatisfaction','WorkLifeBalance','StockOptionLevel','Gender', 'PerformanceRating', 
                    'JobInvolvement','JobLevel', 'JobRole', 'MaritalStatus','RelationshipSatisfaction']
        
        logger.info("Data preprocessing completed successfully")
        return df, num_cols, cat_cols
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise
