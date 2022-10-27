from sklearn.metrics import confusion_matrix

def print_classification_metrics(actuals, predictions):
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