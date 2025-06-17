from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os
import utils_general
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
    config_name = config_path.split("_")[-1].split(".")[0]
    #---------------------------------------------------------
    # DATA PREP
    df = pd.read_csv(df_path)

    df_task_2 = df[df['task']=="2"].copy()

    if transcript_column == "transcript_clean":
        df_task_2["transcript_clean"] = df_task_2["transcript"].apply(lambda x: utils_general.clean_transcript(x))
    #---------------------------------------------------------
    # LOAD MODEL
    model = SentenceTransformer(model_name, device="cuda")

    # EMBED SENTENCES
    sentences = df_task_2[transcript_column].values

    print(f"Embedding {len(sentences)} sentences")
    embeddings = model.encode(sentences, convert_to_numpy=True)
    print(embeddings.shape)

    # make a dictionary of the embeddings
    sample_to_embedding = {sample: embedding for sample, embedding in zip(df_task_2['sample'], embeddings)}

    # save the embeddings
    np.save(f"embeddings/sample_to_embedding_{config_name}.npy", sample_to_embedding)

