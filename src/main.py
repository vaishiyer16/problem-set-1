'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.etl import download_datasets
from src.preprocessing import preprocess_data
from src.logistic_regression import train_logistic_model
from src.decision_tree import train_decision_tree
from src.calibration_plot import evaluate_models



# Call functions / instanciate objects from the .py files
def main():
    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    pred_universe_raw, arrest_events_raw = download_datasets()
    print("ETL complete: CSVs saved to data/")

    # PART 2: Call functions/instanciate objects from preprocessing
    df_clean = preprocess_data(pred_universe_raw, arrest_events_raw)
    print("Preprocessing complete.")

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests_train, df_arrests_test, lr_model = train_logistic_model(df_clean)
    print("Logistic Regression complete.")

    # PART 4: Call functions/instanciate objects from decision_tree
    df_arrests_train, df_arrests_test, tree_model = train_decision_tree(df_arrests_train, df_arrests_test)
    print("Decision Tree complete.")

    # PART 5: Call functions/instanciate objects from calibration_plot
    evaluate_models(df_arrests_test)


if __name__ == "__main__":
    main()