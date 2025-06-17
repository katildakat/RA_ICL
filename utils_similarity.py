import numpy as np
import itertools
from datasets import Dataset
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import metrics
import utils_general
from collections import Counter

def cefr_to_cosine_distance(cefr1, cefr2, score_interval=[1,7]):
    # calculate cosine similarity label based on cefr scores
    score_diff = abs(cefr1 - cefr2)
    max_diff = score_interval[1] - score_interval[0]
    cosine_distance = 1 - (score_diff / max_diff)
    return cosine_distance

def make_cefr_pairs(df):
    all_pairs = []

    print(f"---MAKING CEFR PAIRS FOR {len(df)} SAMPLES---")
    same_dataset_sample_pairs = [pair for pair in itertools.combinations(df['sample'].values, 2)]
    for pair in same_dataset_sample_pairs:
        cefr1 = df[df['sample'] == pair[0]]['cefr_mean'].values[0]
        cefr2 = df[df['sample'] == pair[1]]['cefr_mean'].values[0]
        label = cefr_to_cosine_distance(cefr1, cefr2)
        all_pairs.append([pair[0], pair[1], label])
    
    return all_pairs

def create_cefr_dataset(df, text_column):
        print("---CREATING CEFR DATASET---")
        cefr_pairs = make_cefr_pairs(df)
        
        dataset_dict = {'sentences_1' : [],
                        'sentences_2' : [],
                        'label' : [],
                    }
    
        for pair in cefr_pairs:
            dataset_dict['sentences_1'].append(df[df['sample'] == pair[0]][text_column].values[0])
            dataset_dict['sentences_2'].append(df[df['sample'] == pair[1]][text_column].values[0])
            dataset_dict['label'].append(pair[2])
        
        dataset = Dataset.from_dict(dataset_dict)
        # shuffle the dataset
        dataset = dataset.shuffle()
        return dataset

def create_knn_dataset(data_df, label_column, text_column):
    labels = data_df[label_column].tolist()
    texts = data_df[text_column].tolist()
    knn_dataset_dict = {'text' : texts,
                        'label' : labels}
    knn_dataset = Dataset.from_dict(knn_dataset_dict)
    return knn_dataset

def compute_knn_metrics(train_dataset, val_dataset, model, n_neighbors=1, regression=False, output_preds=False):

    print("Embedding validation and training data")
    val_embeds = model.encode(val_dataset['text'], convert_to_numpy=True)
    train_embeds = model.encode(train_dataset['text'], convert_to_numpy=True)
    val_labels = np.array(val_dataset['label'])
    train_labels = np.array(train_dataset['label'])

    if regression:
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric='cosine', weights='distance')
    else:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine', weights='distance')
    
    knn.fit(train_embeds, train_labels)
    val_preds = knn.predict(val_embeds)
    distances, _ = knn.kneighbors(val_embeds, n_neighbors=min(n_neighbors, len(train_embeds)), return_distance=True)
    
    val_preds_binned = val_preds
    val_labels_binned = val_labels
        
    # For regression: bin before evaluation
    if regression:
        val_preds_binned = utils_general.get_hist_bin(val_preds_binned)
        val_labels_binned = utils_general.get_hist_bin(val_labels_binned)

    # Compute and print metrics
    val_metrics = utils_general.calculate_metrics(val_labels_binned, val_preds_binned, regression=regression)
    print("Evaluation Metrics:")
    print(val_metrics)
    print(metrics.classification_report(val_labels_binned, val_preds_binned))
    print(Counter(val_labels_binned))

    print("Sample predictions vs ground truth:")
    for yp, yt, dist in zip(val_preds_binned[:10], val_labels_binned[:10], distances[:10]):
        if isinstance(dist, np.ndarray):
            dist_str = ", ".join([f"{d:.2f}" for d in dist.flatten() if d is not None])
        else:
            dist_str = ", ".join([f"{d:.2f}" for d in dist if d is not None])
        print(f"PRED: {yp}, TRUE: {yt}, DIST: [{dist_str}]")
    
    if output_preds:
        return val_preds, val_preds_binned


def get_cluster_score(df, label_column, embeddings):
    X = np.vstack(embeddings)
    s_score = metrics.silhouette_score(X, df[label_column].tolist(), metric='cosine')
    c_score = metrics.calinski_harabasz_score(X, df[label_column].tolist())
    return s_score, c_score