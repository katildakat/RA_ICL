# IMPORTING LIBRARIES
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoTokenizer 
import json
import pandas as pd
import numpy as np
import utils_prompts, utils_general
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
    embedding_path = config['embedding_path']
    transcript_column = config['transcript_column']
    label_column = config['label_column']
    random_seed = config['random_seed']
    non_similar = config['non_similar']
    similar_type = config['similar_type']
    #--------------------------------------------------------- 
    # DATA PREP
    df = pd.read_csv(df_path)
    df_task_2 = df[df['task']=="2"].copy()
    if transcript_column == "transcript_clean":
        df_task_2["transcript_clean"] = df_task_2["transcript"].apply(lambda x: utils_general.clean_transcript(x))

    # load the embeddings
    if embedding_path:
        sample_to_embedding = np.load(embedding_path, allow_pickle=True).item()
    else:
        sample_to_embedding = {}
    #--------------------------------------------------------- 
    # LOAD THE MATERIALS FOR THE PROMPTS
    # load templates and scales
    prompt_templates = utils_prompts.load_prompt_templates("data/prompt_templates.yaml")
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #---------------------------------------------------------
    def generate_basic_prompts(df, transcript_column, label_column, prompt_templates, tokenizer):
        print("Starting generate_basic_prompts")
        # basic zero, one a few-shot prompts

        prompts_zero = utils_prompts.make_zero_shot_prompt(df,
                                                           transcript_column,
                                                           prompt_templates, 
                                                           tokenizer)
        prompts_one, sample_to_one_shot_demonstration = utils_prompts.make_one_shot_prompt(df,
                                                                                           transcript_column,
                                                                                           label_column,
                                                                                           prompt_templates, 
                                                                                           tokenizer)
        prompts_one_same_bin, sample_to_one_shot_demonstration_same_bin = utils_prompts.make_one_shot_prompt(df,
                                                                                            transcript_column,
                                                                                           label_column,
                                                                                           prompt_templates, 
                                                                                           tokenizer,
                                                                                           same_bin=True)
        
        prompts_few, sample_to_few_shot_demonstration = utils_prompts.make_few_shot_prompt(df,
                                                                                           transcript_column,
                                                                                           label_column,
                                                                                           prompt_templates, 
                                                                                           tokenizer)
        print("Finished generate_basic_prompts")
        # save the prompts
        with open(f"prompts/sample_to_zero_shot_prompt.json", 'w') as f:
            json.dump(prompts_zero, f)
        with open(f"prompts/sample_to_one_shot_prompt_{label_column}.json", 'w') as f:
            json.dump(prompts_one, f)
        with open(f"prompts/sample_to_one_shot_prompt_same_bin_{label_column}.json", 'w') as f:
            json.dump(prompts_one_same_bin, f)
        with open(f"prompts/sample_to_one_shot_demonstration_same_bin_{label_column}.json", 'w') as f:
            json.dump(sample_to_one_shot_demonstration_same_bin, f)
        with open(f"prompts/sample_to_few_shot_prompt_{label_column}.json", 'w') as f:
            json.dump(prompts_few, f)
        with open(f"prompts/sample_to_one_shot_demonstration_{label_column}.json", 'w') as f:
            json.dump(sample_to_one_shot_demonstration, f)
        with open(f"prompts/sample_to_few_shot_demonstration_{label_column}.json", 'w') as f:
            json.dump(sample_to_few_shot_demonstration, f)
        print("Finished saving prompts")
    
    def generate_similar_prompts(df, sample_to_embedding, transcript_column, score_column, prompt_templates, tokenizer):
        print("Starting generate_similar_prompts")
        one_shot_prompts, sample_to_one_shot_demonstration = utils_prompts.make_n_shot_prompt_similarity(df,
                                                                                                         sample_to_embedding,
                                                                                                         transcript_column,
                                                                                                         score_column,
                                                                                                         prompt_templates,
                                                                                                         tokenizer,
                                                                                                         n_shots=1)
        
        three_shot_prompts, sample_to_three_shot_demonstration = utils_prompts.make_n_shot_prompt_similarity(df,
                                                                                                         sample_to_embedding,
                                                                                                         transcript_column,
                                                                                                         score_column,
                                                                                                         prompt_templates,
                                                                                                         tokenizer,
                                                                                                         n_shots=3)
        
        few_shot_prompts, sample_to_few_shot_demonstration = utils_prompts.make_few_shot_prompt_similarity(df,
                                                                                                         sample_to_embedding,
                                                                                                         transcript_column,
                                                                                                         score_column,
                                                                                                         prompt_templates,
                                                                                                         tokenizer)
        print("Finished generate_similar_prompts")
        # save the prompts
        with open(f"prompts/sample_to_one_shot_prompt_{similar_type}_{score_column}.json", 'w') as f:
            json.dump(one_shot_prompts, f)
        with open(f"prompts/sample_to_three_shot_prompt_{similar_type}_{score_column}.json", 'w') as f:
            json.dump(three_shot_prompts, f)
        with open(f"prompts/sample_to_one_shot_demonstration_{similar_type}_{score_column}.json", 'w') as f:
            json.dump(sample_to_one_shot_demonstration, f)
        with open(f"prompts/sample_to_three_shot_demonstration_{similar_type}_{score_column}.json", 'w') as f:
            json.dump(sample_to_three_shot_demonstration, f)
        with open(f"prompts/sample_to_few_shot_prompt_{similar_type}_{score_column}.json", 'w') as f:
            json.dump(few_shot_prompts, f)
        with open(f"prompts/sample_to_few_shot_demonstration_{similar_type}_{score_column}.json", 'w') as f:
            json.dump(sample_to_few_shot_demonstration, f)
        print("Finished saving prompts")
    #---------------------------------------------------------
    # GENERATE PROMPTS
    print("---------GENERATING PROMPTS---------")
    if non_similar:
        generate_basic_prompts(df_task_2, 
                               transcript_column, 
                               label_column, 
                               prompt_templates, 
                               tokenizer)

    else:
        generate_similar_prompts(df_task_2, 
                                 sample_to_embedding, 
                                 transcript_column, 
                                 label_column, 
                                 prompt_templates, 
                                 tokenizer)

