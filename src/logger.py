import logging
import os


def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# Set up a specific logger with our desired output level
if not os.path.exists('logs'):
    os.makedirs('logs')

logger = setup_logger('HR_Attrition_Prediction', 'logs/HR_Attrition_Prediction.log')
