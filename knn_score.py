from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import utils_general
import utils_similarity
from collections import Counter
#---------------------------------------------------------
if __name__ == "__main__":
    
    # READ CONFIG
    config_path = os.getenv('CONFIG_PATH')
    config = utils_general.load_config(config_path)
    print(config)
    #---------------------------------------------------------
    # SET UP PARAMETERS
    df_path = config['df_path']
    model_name = config['model_name']
    transcript_column = config['transcript_column']
    #---------------------------------------------------------
    # DATA PREP
    df = pd.read_csv(df_path)
    if transcript_column == "transcript_clean":
        df["transcript_clean"] = df["transcript"].apply(lambda x: utils_general.clean_transcript(x))
    # Filter for task 2
    df_task_2 = df[df['task']=="2"].copy()
    df_task_2 = df_task_2.reset_index()  # saves original indices as a column
    original_indices = df_task_2['index']
    train = df[df['round']!="aalto"].copy()
    train = train[train['task_type']=="free"].copy()
    train = train[train['task']!="2"].copy()
    #---------------------------------------------------------
    # LOAD MODEL
    model = SentenceTransformer(model_name, device="cuda")

    # EMBED SENTENCES
    print("---EVALUATING WITH KNN---")
    train_knn_dataset = utils_similarity.create_knn_dataset(train, "cefr_mean", transcript_column)
    val_knn_dataset = utils_similarity.create_knn_dataset(df_task_2, "cefr_mean", transcript_column)
    scores_mean, scores_binned = utils_similarity.compute_knn_metrics(train_knn_dataset, 
                                                                      val_knn_dataset, 
                                                                      model, 
                                                                      regression=True, 
                                                                      output_preds=True)
    
    print(Counter(scores_binned))
    # save the scores to the dataframe
    df['knn_mean'] = 9999.0
    df['knn_bin'] = 9999
    
    # Map predictions back to task 2 entries using original indices
    df.loc[original_indices, 'knn_mean'] = scores_mean
    df.loc[original_indices, 'knn_bin'] = scores_binned
    
    # Save the updated dataframe
    df.to_csv(df_path, index=False)
    print(f"Saved predictions to {df_path}")