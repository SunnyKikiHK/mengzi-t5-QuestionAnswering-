import json
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Text template 
PROMPT_Q_TEMPLATE = "问题:" 
PROMPT_C_TEMPALTE = "上下文:"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

def read_json(input_file: str) -> list:
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading JSON file"):
            data.append(json.loads(line))
        return data 

def get_data_stats(data: list, tokenizer: T5Tokenizer) -> dict:
    question_lengths, context_lengths, answer_lengths = [], [], []
    for sample in data:
        question_lengths.append(len(tokenizer.tokenize(sample["question"])))
        context_lengths.append(len(tokenizer.tokenize(sample["context"])))
        answer_lengths.append(len(tokenizer.tokenize(sample["answer"])))
    return {
        "question_num": len(question_lengths),
        "context_num": len(context_lengths),
        "answer_num": len(answer_lengths),
        "question_mean_length": sum(question_lengths) / len(question_lengths),
        "context_mean_length": sum(context_lengths) / len(context_lengths),
        "answer_mean_length": sum(answer_lengths) / len(answer_lengths),
        "question_max_length": max(question_lengths),
        "context_max_length": max(context_lengths),
        "answer_max_length": max(answer_lengths)
        }
        

def exceed_length_threshold(obj):
    if isinstance(obj['answer'], list):
        for answer in obj['answer']:
            prompt_input = f"{PROMPT_Q_TEMPLATE}{obj['question']} {PROMPT_C_TEMPALTE}{obj['context']}"
            if len(prompt_input) > MAX_INPUT_LENGTH or len(answer) > MAX_TARGET_LENGTH:
                return True
    else:
        prompt_input = f"{PROMPT_Q_TEMPLATE}{obj['question']} {PROMPT_C_TEMPALTE}{obj['context']}"
        if len(prompt_input) > MAX_INPUT_LENGTH or len(obj['answer']) > MAX_TARGET_LENGTH:
            return True
    return False 

def collote_train_fn(batch_samples: list, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer) -> tuple:
    """
    Question and context need to be combined as the input of encoder.
    """
    batch_inputs , batch_answers, batch_ids = [], [], []
    for sample in batch_samples:
        # input format: "问题:the_question 上下文:the_context"
        prompt_input = f"{PROMPT_Q_TEMPLATE}{sample['question']} {PROMPT_C_TEMPALTE}{sample['context']}"

        batch_inputs.append(prompt_input)
        batch_answers.append(sample["answer"])
        batch_ids.append(sample["id"])
    
    batch_data = tokenizer(
        batch_inputs,
        padding="max_length",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt"
        )
    # We don't need to shift right for answers as the T5 from transformers will do that for us.
    # We don't need the attention mask as the decoder will make it for us.
    # decoder_input_ids is not necessary as model will also handle that despite it is written here.
    labels = tokenizer(
        batch_answers,
        padding="max_length",
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        return_tensors="pt"
    )["input_ids"] # (bsz, seq_len)
    batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
    end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
    for i, end_i in enumerate(end_token_index):
        labels[i][end_i+1:] = -100 #Ignored value for cross entropy
    batch_data['labels'] = labels
    return batch_data

def collote_valid_fn(batch_samples: list, model, tokenizer) -> dict:
    """
    Collate function for validation.
    1. Prepares Tensors for the model (using only the 1st answer).
    2. Passes raw answer lists for BLEU metrics.
    """
    batch_inputs = []
    batch_primary_answers = [] # For Tensor/Loss (only the first answer)
    batch_all_answers_raw = [] # For BLEU (list of lists)
    
    for sample in batch_samples:
        # input format: "问题:the_question 上下文:the_context"
        prompt_input = f"{PROMPT_Q_TEMPLATE}{sample['question']} {PROMPT_C_TEMPALTE}{sample['context']}"
        
        batch_inputs.append(prompt_input)
        
        # sample["answer"] is now a list: ["Ans1", "Ans2"]
        answers_list = sample["answer"]
        
        # Store the full list for BLEU later
        batch_all_answers_raw.append(answers_list)
        
        # We assume the first answer is the "main" one for calculating perplexity/loss
        batch_primary_answers.append(answers_list[0])

    # Tokenize Inputs 
    batch_data = tokenizer(
        batch_inputs,
        padding="max_length",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt"
    )

    # Tokenize Labels (Using only primary answers) 
    # This creates a tensor of shape (batch_size, seq_len)
    labels_encoding = tokenizer(
        batch_primary_answers,
        padding="max_length",
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        return_tensors="pt"
    )
    
    labels = labels_encoding.input_ids

    # Replace padding token id with -100 so loss is ignored there
    labels[labels == tokenizer.pad_token_id] = -100
    
    # Add to batch
    batch_data['labels'] = labels
    
    batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)

    # Attach Raw Answers for valid_loop 
    # This is the key key that valid_loop will .pop()
    batch_data['answer'] = batch_all_answers_raw

    return batch_data

def save_checkpoint(model, epoch, output_dir, recent_checkpoints, max_keep=3):
    ckpt_path = output_dir / f"ckpt-epoch{epoch}.pt"

    print(f"Saving checkpoint to {ckpt_path}")
    recent_checkpoints.append(ckpt_path)
    torch.save(model.state_dict(), ckpt_path)

    if len(recent_checkpoints) > max_keep:
        oldest = recent_checkpoints.pop(0)

        oldest.unlink(missing_ok=True)