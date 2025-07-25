'''
PART 3: Logistic Regression
- Read in `df_arrests`
- Use train_test_split to create two dataframes from `df_arrests`, the first is called `df_arrests_train` and the second is called `df_arrests_test`. Set test_size to 0.3, shuffle to be True. Stratify by the outcome  
- Create a list called `features` which contains our two feature names: num_fel_arrests_last_year, current_charge_felony
- Create a parameter grid called `param_grid` containing three values for the C hyperparameter. (Note C has to be greater than zero) 
- Initialize the Logistic Regression model with a variable called `lr_model` 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv` 
- Run the model 
- What was the optimal value for C? Did it have the most or least regularization? Or in the middle? Print these questions and your answers. 
- Now predict for the test set. Name this column `pred_lr`
- Return dataframe(s) for use in main.py for PART 4 and PART 5; if you can't figure this out, save as .csv('s) in `data/` and read into PART 4 and PART 5 in main.py
'''

# Import any further packages you may need for PART 3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

def train_logistic_model(df_arrests):
   
    features = ['num_fel_arrests_last_year', 'current_charge_felony']
    X = df_arrests[features]
    y = df_arrests['y']


    df_arrests_train, df_arrests_test = train_test_split(
        df_arrests, test_size=0.3, shuffle=True, stratify=y, random_state=42
    )

    X_train = df_arrests_train[features]
    y_train = df_arrests_train['y']

    X_test = df_arrests_test[features]
    y_test = df_arrests_test['y']

    param_grid = {'C': [0.01, 1, 100]}  # small = more regularization, large = less

  
    lr_model = LogisticRegression(max_iter=1000, solver='liblinear')


    gs_cv = GridSearchCV(
        lr_model,
        param_grid,
        cv=5,
        scoring='accuracy'
    )


    gs_cv.fit(X_train, y_train)

  
    best_C = gs_cv.best_params_['C']
    print("What was the optimal value for C?")
    print(best_C)

    if best_C == min(param_grid['C']):
        regularization = "most regularization"
    elif best_C == max(param_grid['C']):
        regularization = "least regularization"
    else:
        regularization = "in the middle"
    
    print(f"Did it have the most or least regularization? {regularization}")

    df_arrests_test = df_arrests_test.copy()
    df_arrests_test['pred_lr'] = gs_cv.predict(X_test)

    return df_arrests_train, df_arrests_test, gs_cv



