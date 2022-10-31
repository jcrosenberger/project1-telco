from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import src.acquire as ac




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