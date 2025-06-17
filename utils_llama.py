import torch
from utils_general import get_hist_bin

def process_batch(batch, model, tokenizer, level_to_token_id):
    """
    Process a batch of prompts and return the predicted levels and the probabilities of each level.
    """
    levels = list(level_to_token_id.keys())
    levels.sort()

    inputs = tokenizer(batch['prompt'], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits[:, -1, :] # batch_size x vocab_size (last token logits)
    last_logits_levels = logits[:, [level_to_token_id[level] for level in levels]] # batch_size x n_levels
    last_logits_probs = torch.nn.functional.softmax(last_logits_levels, dim=-1).cpu()
    y_pred = torch.argmax(last_logits_probs, dim=-1) 
    y_pred_ids = [level_to_token_id[levels[i]] for i in y_pred] 

    return y_pred_ids, last_logits_probs.numpy()

def grade_shots(loader, shot_type, model, tokenizer, level_to_token_id, data_records, split=None):
    print(f"GRADING {shot_type.upper()}-SHOT")
    
    # create a lookup dictionary if data_records is not empty
    record_lookup = {(r['sample'], r['split']): r for r in data_records} if data_records else {}

    for i, batch in enumerate(loader):
        print(f"Batch {i+1}/{len(loader)}")
        y_pred, y_probs = process_batch(batch, model, tokenizer, level_to_token_id)
        
        for i, answer_token_id in enumerate(y_pred):
            key = (batch['sample'][i], split)

            # Check for record using composite key
            record = record_lookup.get(key)
            
            if record is None:
                record = {
                    'sample': batch['sample'][i],
                    'split': split,
                    'y_true': int(batch['y_true'][i]), 
                    'task': batch['task'][i],
                    'y_true_mean': float(batch['y_true_mean'][i]),
                    'y_true_all_scores': batch['y_true_all_scores'][i]
                }
                data_records.append(record)
                record_lookup[key] = record  # update lookup for future use

            # Record found or created: add predictions/probabilities
            answer_token = tokenizer.decode(answer_token_id)
            record[f'{shot_type}_prediction'] = int(answer_token)
            
            levels = sorted(level_to_token_id.keys())
            soft_prediction = sum(y_probs[i, j] * int(level) for j, level in enumerate(levels))
            
            record[f'{shot_type}_soft_prediction'] = soft_prediction
            record[f'{shot_type}_soft_prediction_bin'] = get_hist_bin([soft_prediction])[0]

            for j, level in enumerate(levels):
                record[f'{shot_type}_prob_{level}'] = y_probs[i, j]

    return data_records