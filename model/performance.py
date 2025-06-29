from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def measurePerformanceMetrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    #precision = precision_score(y_true, y_pred)
    #recall = recall_score(y_true, y_pred)
    #f1 = f1_score(y_true, y_pred)
    #roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy
