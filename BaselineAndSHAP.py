import transtab
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import shap
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

def outputEvaluation(model, x_test, y_test):
    ypred = transtab.predict(model, x_test, y_test)
    y_pred_binary = (ypred > 0.33).astype(int)
    # calculate sensitivity, specificity, f1_score, and confusion matrix
    sensitivity = recall_score(y_test, y_pred_binary, pos_label=1)
    specificity = recall_score(y_test, y_pred_binary, pos_label=0)
    acc = accuracy_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    confusion = confusion_matrix(y_test, y_pred_binary)
    total = confusion.sum()
    percentage_confusion = confusion/total * 100
    # print the results
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", confusion)
    print(acc)
    print(percentage_confusion)



def shapEvaluation(model, x_test, startidx, endidx):
    background = x_test.head(100)
    def model_wrapper(x, batch_size=10):
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=background.columns)
        results = []
        for i in range(0, len(x), batch_size):
            batch_x = x.iloc[i:i + batch_size]
            model_output = model(batch_x)
            if isinstance(model_output, tuple):
                model_output = model_output[0]
            if isinstance(model_output, torch.Tensor):
                model_output = model_output.cpu().detach().numpy()
            results.append(model_output)
        return np.concatenate(results)
    
    explainer = shap.KernelExplainer(model_wrapper, background)
    instance_to_explain = x_test.iloc[startidx:endidx]
    shap_values = explainer.shap_values(instance_to_explain)
    explanation = shap.Explanation(values=shap_values.squeeze(),
                               base_values=explainer.expected_value,
                               data=instance_to_explain,
                               feature_names=background.columns.tolist())
    shap.summary_plot(shap_values.reshape((endidx-startidx,34)), instance_to_explain, plot_type="dot")