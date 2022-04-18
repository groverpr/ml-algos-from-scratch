from src.libraries import *


def roc_curve_scratch(y_labels, y_scores):
    thresholds = np.linspace(0, 1, 1001)
    fprs, tprs = [], []
    for thres in thresholds:
        tp, tn, fp, fn = confusion_matrix(y_labels, y_scores, thres)
        tprs.append(tpr(tp, tn, fp, fn))
        fprs.append(fpr(tp, tn, fp, fn))
        
    # force first and last fpr, tpr at 0 and 1 thresholds
    fprs[0] = 1
    fprs[-1] = 0
    tprs[0] = 1
    tprs[-1] = 0
    return fprs, tprs, thresholds
    
def confusion_matrix(y_labels, y_scores, thres):
    y_preds = (y_scores >= thres).astype(int)
    tp = (np.equal(y_labels, 1) & np.equal(y_preds, 1)).sum()
    tn = (np.equal(y_labels, 0) & np.equal(y_preds, 0)).sum()
    fp = (np.equal(y_labels, 0) & np.equal(y_preds, 1)).sum()
    fn = (np.equal(y_labels, 1) & np.equal(y_preds, 0)).sum()
    
    return tp, tn, fp, fn

def tpr(tp, tn, fp, fn):
    return tp/(tp+fn)

def fpr(tp, tn, fp, fn):
    return fp/(fp+tn)

def auc_scratch(fprs, tprs):
    """
    Cut in small rectangles and sum areas
    """
    total_auc = 0.
    for i in range(1000):  # divide curve in 1000 rectangles
        total_auc += (fprs[i] - fprs[i+1])*((tprs[i+1] + tprs[i])/2.)
    return total_auc