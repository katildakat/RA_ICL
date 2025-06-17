print("---LOADING PACKAGES")
import utils_general
import utils_similarity
import pandas as pd
import os
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
)   
from sentence_transformers.losses import CosineSimilarityLoss
#-----------------------------------------------------------
if __name__ == "__main__":
    # READ CONFIG
    config_path = os.getenv('CONFIG_PATH')
    config = utils_general.load_config(config_path)
    #-----------------------------------------------------------
    # SET UP PARAMETERS
    split = config['split']
    df_path = config['df_path']
    transcript_column = config['transcript_column']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    embedding_model_name = config['embedding_model_name']
    model_name_save = config['model_name_save']
    #-----------------------------------------------------------
    print("---LOADING DATA")
    df = pd.read_csv(df_path)
    # only freeform, only school, not task 2
    df = df[df['task_type']=='free'].copy()
    df = df[df['round']!='aalto'].copy()
    if transcript_column == "transcript_clean":
        df["transcript_clean"] = df["transcript"].apply(lambda x: utils_general.clean_transcript(x))
    if split == "all":
        train = df[df['task']!="2"].copy()
        val = df[df['task']=="2"].copy()
    else:
        non_2_df = df[df['task']!="2"].copy()
        train = non_2_df[non_2_df['split']!=split].copy()
        val = non_2_df[non_2_df['split']==split].copy()
    #-----------------------------------------------------------
    # LOAD MODEL
    print("---INITIALIZING THE MODEL---")
    model = SentenceTransformer(embedding_model_name, device="cuda")
    #-----------------------------------------------------------
    # CREATE CEFR SIMILARITY DATASETS
    print("---CREATING CEFR SIMILARITY DATASETS---")
    cefr_train_dataset = utils_similarity.create_cefr_dataset(train, transcript_column)
    cefr_val_dataset = utils_similarity.create_cefr_dataset(val, transcript_column)
    print(f"Number of samples in train dataset: {len(cefr_train_dataset)}")
    print(f"Number of samples in val dataset: {len(cefr_val_dataset)}")
    #-----------------------------------------------------------
    # SETTING UP TRAINING
    print("---SETTING UP TRAINING---")
    cefr_loss = CosineSimilarityLoss(model)
    # create an evaluator
    cefr_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=cefr_val_dataset["sentences_1"],
        sentences2=cefr_val_dataset["sentences_2"],
        scores=cefr_val_dataset["label"],
        main_similarity=SimilarityFunction.COSINE,
        name="cefr_similarity",
    )
    # set up training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"similarity_models/{model_name_save}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no"
    )

    # create a trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=cefr_train_dataset,
        eval_dataset=cefr_val_dataset,
        loss=cefr_loss,   
        evaluator=cefr_evaluator,
    )

    #-----------------------------------------------------------
    print("---EVALUATING BEFORE TRAINING---")
    results = cefr_evaluator(model)
    print(cefr_evaluator.primary_metric)
    print(results[cefr_evaluator.primary_metric])

    # evaluate with knn
    print("---EVALUATING WITH KNN---")
    train_knn_dataset = utils_similarity.create_knn_dataset(train, "cefr_mean", transcript_column)
    val_knn_dataset = utils_similarity.create_knn_dataset(val, "cefr_mean", transcript_column)
    utils_similarity.compute_knn_metrics(train_knn_dataset, val_knn_dataset, model, regression=True)

    # evaluate with cluster score
    print("---EVALUATING WITH CEFR BIN CLUSTER SCORE---")
    train_embeddings = model.encode(train[transcript_column].tolist())
    val_embeddings = model.encode(val[transcript_column].tolist())
    train_cluster_score, train_calinski_harabasz_score = utils_similarity.get_cluster_score(train, "cefr_bin", train_embeddings)
    val_cluster_score, val_calinski_harabasz_score = utils_similarity.get_cluster_score(val, "cefr_bin", val_embeddings)
    print(f"Train cluster score: {train_cluster_score}")
    print(f"Train calinski harabasz score: {train_calinski_harabasz_score}")
    print(f"Val cluster score: {val_cluster_score}")
    print(f"Val calinski harabasz score: {val_calinski_harabasz_score}")
    #-----------------------------------------------------------
    # TRAINING
    print("---TRAINING THE MODEL---")
    trainer.train()

    if model_name_save:
        print("---SAVING THE MODEL---")
        model.save_pretrained(f"similarity_models/{model_name_save}")
    #-----------------------------------------------------------
    print("---EVALUATING AFTER TRAINING---")
    results = cefr_evaluator(model)
    print(cefr_evaluator.primary_metric)
    print(results[cefr_evaluator.primary_metric])

    # evaluate with knn
    print("---EVALUATING WITH KNN---")
    train_knn_dataset = utils_similarity.create_knn_dataset(train, "cefr_mean", transcript_column)
    val_knn_dataset = utils_similarity.create_knn_dataset(val, "cefr_mean", transcript_column)
    utils_similarity.compute_knn_metrics(train_knn_dataset, val_knn_dataset, model, regression=True)

    # evaluate with cluster score
    print("---EVALUATING WITH CEFR BIN CLUSTER SCORE---")
    train_embeddings = model.encode(train[transcript_column].tolist())
    val_embeddings = model.encode(val[transcript_column].tolist())
    train_cluster_score, train_calinski_harabasz_score = utils_similarity.get_cluster_score(train, "cefr_bin", train_embeddings)
    val_cluster_score, val_calinski_harabasz_score = utils_similarity.get_cluster_score(val, "cefr_bin", val_embeddings)
    print(f"Train cluster score: {train_cluster_score}")
    print(f"Train calinski harabasz score: {train_calinski_harabasz_score}")
    print(f"Val cluster score: {val_cluster_score}")
    print(f"Val calinski harabasz score: {val_calinski_harabasz_score}")
