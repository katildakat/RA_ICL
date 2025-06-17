from sklearn import metrics
from scipy.stats import spearmanr
import numpy as np
import yaml
import re
import pandas as pd

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def clean_transcript(transcript):
    transcript = transcript.lower()
    
    # a fix for one transcript:
    transcript = transcript.replace("<name*", "<name>")
    
    clean_t = re.sub('<.*?>', '', transcript)  
    clean_t = "".join(c for c in clean_t if c.isalpha() or c.isspace())
    clean_t = " ".join(clean_t.split())
    return clean_t

def get_hist_bin(values, range_min=1, range_max=7):
    n_bins = range_max-range_min+1
    
    bin_labels=[]
    
    # get bin edges
    _, bin_edges = np.histogram([x for x in range(range_min,range_max+1)], bins=n_bins)
    for v in values:
        if v==9999:
            bin_labels.append(9999)
        else:
            i = 1
            while v > bin_edges[i]:
                i+=1
            b=i
            bin_labels.append(b)
    
    return bin_labels

def macro_mae(y_true, y_pred):
    """
    Calculate average MAE per class
    """
    levels = list(set(y_true)) # get all unique levels
    levels.sort()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = 0
    mae_per_bin = []
    for level in levels:
        level_mask = y_true == level
        level_y_true = y_true[level_mask]
        level_y_pred = y_pred[level_mask]
        level_mae = metrics.mean_absolute_error(level_y_true, level_y_pred)
        mae_per_bin.append(level_mae)
        mae += level_mae
    return mae/len(levels), mae_per_bin

def print_results(results, shot_types):
    y_true = results["y_true"]
    # for each shot type
    # print s win rate
    # accuracy, macro f1, macro mae, kappa, spearman correlation
    for shot_column in shot_types:
        y_pred = results[f"{shot_column}_prediction"]
        y_pred_hard_bin = get_hist_bin(y_pred)
        y_pred_soft_bin = results[f"{shot_column}_soft_prediction_bin"]
        y_pred_soft_bin = get_hist_bin(y_pred_soft_bin)
        
        f1_hard = metrics.f1_score(y_true, y_pred_hard_bin, average='macro')
        acc_hard = metrics.accuracy_score(y_true, y_pred_hard_bin)
        mae_hard, mae_per_bin_hard = macro_mae(y_true, y_pred_hard_bin)
        kappa_hard = metrics.cohen_kappa_score(y_true, y_pred_hard_bin, weights='quadratic')
        spearman_hard = spearmanr(y_true, y_pred_hard_bin)[0]
        
        # Calculate per-bin F1 scores for hard predictions
        f1_per_bin_hard = metrics.f1_score(y_true, y_pred_hard_bin, average=None)
        
        # print rounded values
        print(f"Shot type: {shot_column}")
        #print(f"Acc: {acc_hard:.2f}, F1: {f1_hard:.2f}, QWK: {kappa_hard:.2f}, MAE: {mae_hard:.2f}, Spearman: {spearman_hard:.2f}")
        #print("F1 per bin (hard):", ", ".join([f"{score:.2f}" for score in f1_per_bin_hard]))

        f1_soft = metrics.f1_score(y_true, y_pred_soft_bin, average='macro')
        acc_soft = metrics.accuracy_score(y_true, y_pred_soft_bin)
        mae_soft, mae_per_bin_soft = macro_mae(y_true, y_pred_soft_bin)
        kappa_soft = metrics.cohen_kappa_score(y_true, y_pred_soft_bin, weights='quadratic')
        spearman_soft = spearmanr(y_true, y_pred_soft_bin)[0]
        
        # Calculate per-bin F1 scores for soft predictions
        f1_per_bin_soft = metrics.f1_score(y_true, y_pred_soft_bin, average=None)
        
        # print rounded values
        print(f"Acc: {acc_soft:.2f}, F1: {f1_soft:.2f}, QWK: {kappa_soft:.2f}, MAE: {mae_soft:.2f}, Spearman: {spearman_soft:.2f}")
        print("F1 per bin (soft):", ", ".join([f"{score:.2f}" for score in f1_per_bin_soft]))
        print("MAE per bin (soft):", ", ".join([f"{score:.2f}" for score in mae_per_bin_soft]))
        print("\n")

def calculate_metrics(y_true, y_pred, regression=False):
    
    if regression:
        y_true_bin = get_hist_bin(y_true)
        y_pred_bin = get_hist_bin(y_pred)
        acc = metrics.accuracy_score(y_true_bin, y_pred_bin)
        f1 = metrics.f1_score(y_true_bin, y_pred_bin, average='macro')
        kappa = metrics.cohen_kappa_score(y_true_bin, y_pred_bin, weights='quadratic')
        mae, _ = macro_mae(y_true_bin, y_pred_bin)
        spearman = spearmanr(y_true, y_pred)[0]
    else:
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        kappa = 0
        mae = 0
        spearman = 0
        
    return {
        'ACC': acc,
        'F1': f1,
        'QWK': kappa,
        'MAE': mae,
        'Spearman': spearman}

def create_confusion_matrix(y_true, y_pred, labels=None): 
    
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    
    summed_values = cm.sum(axis=1) # sum rows
    summed_values = summed_values[:, np.newaxis]
    normalized_matrix = cm/summed_values

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    # Create a pandas DataFrame for the confusion matrix for more readable printing
    cm_df = pd.DataFrame(cm, index=[f'TRUE {label}' for label in labels], columns=[f'PRED {label}' for label in labels])
    normalized_df = pd.DataFrame(normalized_matrix, index=[f'TRUE {label}' for label in labels], columns=[f'PRED {label}' for label in labels])
    return cm_df, normalized_df