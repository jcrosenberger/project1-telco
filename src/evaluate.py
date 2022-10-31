# standard modules
import seaborn as sns
import pandas as pd
import numpy as np
import os
#import math

# Modules for Displaying Figures
import matplotlib.pyplot as plt
import scipy.stats as stats


# Data Science Modules 
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# My modules
import src.acquire as ac
import src.prepare as pp
import src.helper as helper
import src.evaluate as evaluate



def get_data(): 
    df = ac.get_telco_data()
    return df 

def print_classification_metrics(actuals, predictions):
    '''A method which allows me to say time and coding space by making the metrics for evaluating 
    a model simple and easy to access. Calculates Accuracy, Precision, Recall, Sensitivity and f1
    '''    
    
    TN, FP, FN, TP = confusion_matrix(actuals, predictions).ravel()
    ALL = TP + TN + FP + FN
    
    accuracy = (TP + TN)/ALL
    print(f"Accuracy: {round(accuracy*100,2)}")

    true_positive_rate = TP/(TP+FN)
    print(f"True Positive Rate: {round(true_positive_rate*100,2)}")

    false_positive_rate = FP/(FP+TN)
    print(f"False Positive Rate: {round(false_positive_rate*100,2)}")

    true_negative_rate = TN/(TN+FP)
    print(f"True Negative Rate: {round(true_negative_rate*100,2)}")

    false_negative_rate = FN/(FN+TP)
    print(f"False Negative Rate: {round(false_negative_rate*100,2)}")

    precision = TP/(TP+FP)
    print(f"Precision: {round(precision*100,2)}")

    recall = TP/(TP+FN)
    print(f"Recall: {round(recall*100,2)}")

    f1_score = 2*(precision*recall)/(precision+recall)
    print(f"F1 Score: {round(f1_score*100,2)}")

    support_pos = TP + FN
    print(f"Support (0): {round(support_pos*100,2)}")

    support_neg = FP + TN
    print(f"Support (1): {round(support_neg*100,)}")





def baseline(df,var):
    ''' Method calculates the minimum odds that a model will need to beat,
    based on the target variable and the dataframe passed in
    '''
    telco = ac.get_telco_data()
    if (df[var].value_counts(normalize=True))[1] > (df[var].value_counts(normalize=True))[0]:
        baseline = (df[var].value_counts(normalize=True))[1]
    else: 
        baseline = (df[var].value_counts(normalize=True))[0]
    return baseline

#example template for how to call function
#print(f"Baseline accuracy = {round(baseline*100,2)}%") 



def make_pie(df, var):
    '''Make Pie Chart demonstrating churn as a problem'''

    # set values and labels for chart
    #yes = len(telco.churn[telco.churn == 'Yes'])
    #no  = len(telco.churn[telco.churn == 'No'])
    #yes = len(df.var[df.var == 'Yes'])
    #no  = len(df.var[df.var == 'No'])
    #values = [yes, no]
    values = [len(df[var][df[var] == 'Yes']), len(df[var][df[var] == 'No'])] 
    labels = ['Did not Churn','Churned', ] 

    # generate and show chart
    plt.pie(values, labels=labels, autopct='%.0f%%', colors=['#ffc3a0', '#c0d6e4'])
    plt.title('Customers who churned.')
    plt.show()


def baseline_bar(var):
    "get graph of baseline rating"

    # assign values and labels
    labels = ['Churned', 'Did not Churn']
    values = [var*100, (1-var)*100]

    # generate and display graph
    plt.bar(labels, values, color=['#ffc3a0', '#c0d6e4'])
    plt.title('The baseline statistic our model must beat')
    plt.tight_layout()
    plt.show()


def get_t_score(df, var, target_var):
    "get t-test for mean rating difference in tenure "
    t, p = stats.ttest_ind(df[var][df[target_var] == 'Yes'],df[var][df[target_var] == 'No'])
#    t, p = stats.ttest_ind(train.rating_difference[(train.upset == True)],train.rating_difference[(train.upset == False)])

    return(t, p)
#    print(f't = {t:.4f}')
#    print(f'p = {p:.4f}')


def decision_tree_model(x_train, y_train, x_test, y_test):
    # Make the model
    tree = DecisionTreeClassifier(max_depth=5, random_state=7)

    # Fit the model (on train and only train)
    final_model = tree.fit(x_train, y_train)

    # Use the model
    # We'll evaluate the model's performance on train, first
    #in_sample_accuracy = tree.score(x_train, y_train)

    final_model_accuracy = tree.score(x_test, y_test)
    #y_predictions = final_model.predict(x_test)
    #report = classification_report(y_test, y_predictions, output_dict=True)

    return final_model_accuracy




def knn_model(x_train, y_train, x_test, y_test):
    test_predict = {
        'model': [],
        'accuracy': [],
        'true_positive_rate': [],
        'false_positive_rate': [],
        'true_negative_rate': [],
        'false_negative_rate': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'support_0': [],
        'support_1': []
    }
    n = 10

    knn = KNeighborsClassifier(n_neighbors=n)

    knn.fit(x_train, y_train)

    y_preds = knn.predict(x_test)

    TN, FP, FN, TP = confusion_matrix(y_test, y_preds).ravel()
    ALL = TP + TN + FP + FN

    accuracy = (TP + TN)/ALL
    true_positive_rate = TP/(TP+FN)
    false_positive_rate = FP/(FP+TN)
    true_negative_rate = TN/(TN+FP)
    false_negative_rate = FN/(FN+TP)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score = 2*(precision*recall)/(precision+recall)
    support_pos = TP + FN
    support_neg = FP + TN

    test_predict['model'].append(f'knn_n_{n}')
    test_predict['accuracy'].append(accuracy)
    test_predict['true_positive_rate'].append(true_positive_rate)
    test_predict['false_positive_rate'].append(false_positive_rate)
    test_predict['true_negative_rate'].append(true_negative_rate)
    test_predict['false_negative_rate'].append(false_negative_rate)
    test_predict['precision'].append(precision)
    test_predict['recall'].append(recall)
    test_predict['f1_score'].append(f1_score)
    test_predict['support_0'].append(support_pos)
    test_predict['support_1'].append(support_neg)

    test_predict = pd.DataFrame(test_predict).T

    return test_predict.T['accuracy'][0]