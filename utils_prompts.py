import yaml
import json
import random
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

def load_prompt_templates(prompts_path: str):
    with open(prompts_path, 'r') as file:
        return yaml.safe_load(file)['prompts']
    
def tokenize_chat_into_string(tokenizer, chat_dict, print_chat=False):
    eos_token = tokenizer.eos_token
    len_eos_token = len(eos_token)
    chat_string = tokenizer.apply_chat_template(chat_dict, tokenize=False)
    if print_chat:
        print(chat_string)
    # remove the end of string token
    chat_string = chat_string[:-len_eos_token]
    # check that the last token is a whitespace
    if chat_string[-1] != " ":
        chat_string = chat_string + " "
    return chat_string
    
def make_zero_shot_prompt(task_df, transcript_column, prompt_templates, tokenizer):
    task_zero_shot_prompts = {}

    # fill in the system message template
    system_message = prompt_templates['system_message'].format(proficiency_scale=prompt_templates['proficiency_scale'],
                                                               task_description=prompt_templates['task_description'])
    zero_shot_chat_general = [
        {"role": "system", "content": system_message} 
    ]

    # finish the prompt for each sample
    for i, sample in enumerate(task_df['sample']):
        sample_zero_shot_chat = zero_shot_chat_general.copy()
        
        transcript = task_df[task_df['sample']==sample][transcript_column].values[0]

        # add a transcript that needs grading to the chat
        sample_zero_shot_chat.append({"role": "user", "content": transcript})
        
        # add the start of the assistant message
        sample_zero_shot_chat.append({"role": "assistant", "content": prompt_templates['assistant_message']})
        
        # turn chat template into a string
        sample_zero_shot_string = tokenize_chat_into_string(tokenizer, sample_zero_shot_chat)

        if i == 0:
            print(sample_zero_shot_chat)
            tokenize_chat_into_string(tokenizer, sample_zero_shot_chat, print_chat=True)
            print(sample_zero_shot_string)
        task_zero_shot_prompts[sample] = sample_zero_shot_string

    return task_zero_shot_prompts

def make_one_shot_prompt(task_df, transcript_column, score_column, prompt_templates, tokenizer, random_seed=42, same_bin=False):
    """
    Select a random example from the same task and create a one-shot prompt for each sample.
    If random_label is True, the label for the demonstration is selected randomly.
    """
    sample_to_demonstrations = {}
    task_one_shot_prompts = {}

    # Fill in system message template
    system_message = prompt_templates['system_message'].format(
        proficiency_scale=prompt_templates['proficiency_scale'],
        task_description=prompt_templates['task_description']
    )

    one_shot_chat_general = [
        {"role": "system", "content": system_message}
    ]

    # Finish the prompt for each sample
    for i, sample in enumerate(task_df['sample'].tolist()):
        non_sample_df = task_df[task_df['sample'] != sample]
        if same_bin:
            sample_bin = task_df[task_df['sample'] == sample][score_column].values[0]
            non_sample_df = non_sample_df[non_sample_df[score_column] == sample_bin]
            # if empty, sample from the closest bins
            if non_sample_df.empty:
                closest_bins = [sample_bin - 1, sample_bin + 1]
                closest_bins = [bin for bin in closest_bins if bin >= 1 and bin <= 7]
                print("no same bin for sample", sample)
                print("closest bins", closest_bins)
                non_sample_df = non_sample_df[non_sample_df[score_column].isin(closest_bins)]
        
        sample_one_shot_chat = one_shot_chat_general.copy()
        transcript = task_df[task_df['sample'] == sample][transcript_column].values[0]

        # Select a random example
        reproducible_seed = random_seed + i*10
        example_row = non_sample_df.sample(1, random_state=reproducible_seed)

        example_id = example_row['sample'].values[0]
        example_transcript = example_row['transcript_clean'].values[0] # human transcript without metadata
        example_bin = example_row[score_column].values[0]

        # Add example to the template
        sample_one_shot_chat.append({"role": "user", "content": example_transcript})
        sample_one_shot_chat.append({"role": "assistant", "content": prompt_templates['assistant_message'] + " " + str(example_bin)})

        # Add transcript to grade
        sample_one_shot_chat.append({"role": "user", "content": transcript})
        # Add the start of the assistant message
        sample_one_shot_chat.append({"role": "assistant", "content": prompt_templates['assistant_message']})

        # Turn chat template into a string
        sample_one_shot_string = tokenize_chat_into_string(tokenizer, sample_one_shot_chat)

        task_one_shot_prompts[sample] = sample_one_shot_string
        sample_to_demonstrations[sample] = example_id

    return task_one_shot_prompts, sample_to_demonstrations

def make_few_shot_prompt(task_df, transcript_column, score_column, prompt_templates, tokenizer, random_seed=42):
    """
    Create few-shot prompts for each sample by selecting examples and building a prompt string.
    """

    sample_to_demonstrations = {}
    task_few_shot_prompts = {}

    # Fill in system message template
    system_message = prompt_templates['system_message'].format(
        proficiency_scale=prompt_templates['proficiency_scale'],
        task_description=prompt_templates['task_description']
    )

    few_shot_chat_general = [
        {"role": "system", "content": system_message}
    ]   

    # Generate prompts for each sample
    for i, sample in enumerate(task_df['sample']):
        reproducible_seed = random_seed + i*10
        non_sample_df = task_df[task_df['sample'] != sample]
        sample_few_shot_chat = few_shot_chat_general.copy()

        # sample 1 example from each bin
        grouped_bins = non_sample_df.groupby([score_column])
        examples_list = []
        for _, group in grouped_bins:
            example = group.sample(1, random_state=reproducible_seed)
            examples_list.append(example)
        examples_df = pd.concat(examples_list)
        # shuffle the examples
        examples_df = examples_df.sample(frac=1, random_state=reproducible_seed)

        # Track examples used for the current sample
        demonstration_ids = []
        for _, example_row in examples_df.iterrows():
            example_id = example_row['sample']
            example_transcript = example_row['transcript_clean']
            example_bin = example_row[score_column]
            demonstration_ids.append(example_id)
            sample_few_shot_chat.append({"role": "user", "content": example_transcript})
            sample_few_shot_chat.append({"role": "assistant", "content": prompt_templates['assistant_message'] + " " + str(example_bin)})

        # Save demonstrations for the sample
        sample_to_demonstrations[sample] = demonstration_ids

        # Get the transcript for the current sample
        transcript = task_df[task_df['sample'] == sample][transcript_column].values[0]

        # Add the sample transcript to grade
        sample_few_shot_chat.append({"role": "user", "content": transcript})
        # Add the start of the assistant message
        sample_few_shot_chat.append({"role": "assistant", "content": prompt_templates['assistant_message']})

        # Turn chat template into a string
        sample_few_shot_string = tokenize_chat_into_string(tokenizer, sample_few_shot_chat)

        # Save the prompt
        task_few_shot_prompts[sample] = sample_few_shot_string

    return task_few_shot_prompts, sample_to_demonstrations


def make_few_shot_prompt_similarity(task_df, task_vsm, transcript_column, label_column, prompt_templates, tokenizer, random_seed=42):

    sample_to_demonstrations = {}
    task_few_shot_prompts = {}

    # Prepare system message
    system_message = prompt_templates['system_message'].format(
        proficiency_scale=prompt_templates['proficiency_scale'],
        task_description=prompt_templates['task_description']
    )
    few_shot_chat_general = [
        {"role": "system", "content": system_message}
    ]

    label_bins = task_df[label_column].unique()

    for i, sample in enumerate(task_df['sample']):
        sample_vector = task_vsm[sample].reshape(1, -1)
        non_sample_df = task_df[task_df['sample'] != sample]
        sample_few_shot_chat = few_shot_chat_general.copy()

        demonstration_ids = []

        # For each bin/label, find the closest sample
        for bin_value in label_bins:
            group = non_sample_df[non_sample_df[label_column] == bin_value]
            if group.empty:
                continue  # no examples for this bin

            # Get sample ids and vectors for this bin
            group_ids = group['sample'].tolist()
            group_vectors = np.stack([task_vsm[ex_id] for ex_id in group_ids])
            # Compute cosine distances
            nn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
            nn_model.fit(group_vectors)
            # Get the closest example
            _, index = nn_model.kneighbors(sample_vector)
            # the closest example
            closest_id = group_ids[index[0][0]]
            demonstration_ids.append(closest_id)

        # shuffle the demonstration ids with the same seed
        reproducible_seed = random_seed + i*10
        random.seed(reproducible_seed)
        random.shuffle(demonstration_ids)
        
        for closest_id in demonstration_ids:
            # get the row by closest sample id
            example_row = task_df.loc[task_df['sample'] == closest_id].iloc[0]
            example_transcript = example_row[transcript_column]
            example_label = example_row[label_column]

            sample_few_shot_chat.append({"role": "user", "content": example_transcript})
            sample_few_shot_chat.append(
                {"role": "assistant", "content": f"{prompt_templates['assistant_message']} {example_label}"}
            )

        sample_to_demonstrations[sample] = demonstration_ids

        # Add the current sample transcript
        transcript = task_df[task_df['sample'] == sample][transcript_column].values[0]
        sample_few_shot_chat.append({"role": "user", "content": transcript})
        sample_few_shot_chat.append({"role": "assistant", "content": prompt_templates['assistant_message']})

        # Stringify prompt
        sample_few_shot_string = tokenize_chat_into_string(tokenizer, sample_few_shot_chat)
        task_few_shot_prompts[sample] = sample_few_shot_string

    return task_few_shot_prompts, sample_to_demonstrations

def make_n_shot_prompt_similarity(
    task_df, task_vsm, transcript_column, label_column, prompt_templates, tokenizer, n_shots=3
):

    sample_to_demonstrations = {}
    task_n_shot_prompts = {}

    # Prepare system message
    system_message = prompt_templates['system_message'].format(
        proficiency_scale=prompt_templates['proficiency_scale'],
        task_description=prompt_templates['task_description']
    )
    base_chat = [{"role": "system", "content": system_message}]

    all_sample_ids = task_df['sample'].tolist()
    all_vectors = np.stack([task_vsm[s] for s in all_sample_ids])
    # Fit on all, for efficiency
    nn_model = NearestNeighbors(n_neighbors=n_shots + 1, metric='cosine')
    nn_model.fit(all_vectors)

    for i, sample in enumerate(all_sample_ids):
        sample_vector = task_vsm[sample].reshape(1, -1)
        # Find n_shots + 1 nearest (includes itself at dist 0)
        dists, indices = nn_model.kneighbors(sample_vector, n_neighbors=n_shots + 1)
        # Remove itself from results
        indices = indices[0]
        demo_indices = [idx for idx in indices if all_sample_ids[idx] != sample][:n_shots]
        demo_ids = [all_sample_ids[idx] for idx in demo_indices]

        chat = base_chat.copy()
        for demo_id in demo_ids:
            row = task_df.loc[task_df['sample'] == demo_id].iloc[0]
            transcript = row['transcript_clean']
            label = row[label_column]
            chat.append({"role": "user", "content": transcript})
            chat.append({"role": "assistant", "content": f"{prompt_templates['assistant_message']} {label}"})

        sample_transcript = task_df.loc[task_df['sample'] == sample, transcript_column].values[0]
        chat.append({"role": "user", "content": sample_transcript})
        chat.append({"role": "assistant", "content": prompt_templates['assistant_message']})

        prompt_string = tokenize_chat_into_string(tokenizer, chat)

        task_n_shot_prompts[sample] = prompt_string
        sample_to_demonstrations[sample] = demo_ids

    return task_n_shot_prompts, sample_to_demonstrations

def load_prompts(prompts_path):
    with open(prompts_path) as f:
        prompts = json.load(f)
    return prompts

def make_llama_prompt_dataloader(sample_to_prompt, batch_size, df):

    samples = list(sample_to_prompt.keys())
    # make sure that the samples are in the dataframe[]
    sample_df = df[df['sample'].isin(samples)]
    # make sure that the order as in samples
    sample_df = sample_df.set_index('sample').loc[samples].reset_index()

    y_true = sample_df['cefr_bin'].tolist()
    tasks = sample_df['task'].tolist()
    y_true_means = sample_df['cefr_mean'].tolist()
    y_true_all_scores = sample_df['cefr_all_scores'].tolist()
    prompts = [sample_to_prompt[sample] for sample in samples]
    dataset = Dataset.from_dict({'sample':samples,
                                 'y_true':y_true,
                                 'task':tasks,
                                 'y_true_mean':y_true_means,
                                 'y_true_all_scores':y_true_all_scores,                
                                 'prompt':prompts})
    
    dataloader = DataLoader(dataset, batch_size=batch_size)

    return dataloader