'''
PART 5: Calibration-light
Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
Which model is more calibrated? Print this question and your answer. 

Extra Credit
Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
Compute AUC for the logistic regression model
Compute AUC for the decision tree model
Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calibration_plot(y_true, y_prob, n_bins=10):
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

def evaluate_models(df_arrests_test):

    print("\nCalibration Plot: Logistic Regression")
    calibration_plot(df_arrests_test["y"], df_arrests_test["pred_lr"], n_bins=5)


    print("\nCalibration Plot: Decision Tree")
    calibration_plot(df_arrests_test["y"], df_arrests_test["pred_dt"], n_bins=5)


    print("Which model is more calibrated?")
    print("→ Check visually: the model whose curve is closer to the diagonal line is more calibrated.")

    # --- Extra Credit ---


    top50_lr = df_arrests_test.sort_values("pred_lr", ascending=False).head(50)
    top50_dt = df_arrests_test.sort_values("pred_dt", ascending=False).head(50)

    ppv_lr = top50_lr["y"].mean()
    ppv_dt = top50_dt["y"].mean()

    auc_lr = roc_auc_score(df_arrests_test["y"], df_arrests_test["pred_lr"])
    auc_dt = roc_auc_score(df_arrests_test["y"], df_arrests_test["pred_dt"])

    print(f"\nPPV for top 50 Logistic Regression: {ppv_lr:.3f}")
    print(f"PPV for top 50 Decision Tree: {ppv_dt:.3f}")
    print(f"AUC for Logistic Regression: {auc_lr:.3f}")
    print(f"AUC for Decision Tree: {auc_dt:.3f}")

    print("\nDo both metrics agree that one model is more accurate?")
    if auc_lr > auc_dt and ppv_lr > ppv_dt:
        print("Yes — Logistic Regression is more accurate on both metrics.")
    elif auc_dt > auc_lr and ppv_dt > ppv_lr:
        print("Yes — Decision Tree is more accurate on both metrics.")
    else:
        print("No — the models disagree depending on the metric.")
