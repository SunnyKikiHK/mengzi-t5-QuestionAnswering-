import json
from tqdm import tqdm
from transformers import T5Tokenizer

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
        "question_mean_length": sum(question_lengths) / len(question_lengths),
        "context_mean_length": sum(context_lengths) / len(context_lengths),
        "answer_mean_length": sum(answer_lengths) / len(answer_lengths),
        "question_max_length": max(question_lengths),
        "context_max_length": max(context_lengths),
        "answer_max_length": max(answer_lengths)
        }

def train_collote_fn(batch_samples: list, tokenizer: T5Tokenizer) -> tuple:
    batch_questions, batch_contexts, batch_answers, batch_ids = [], [], [], []
    for sample in batch_samples:
        batch_questions.append(sample["question"])
        batch_contexts.append(sample["context"])
        batch_answers.append(sample["answer"])
        batch_ids.append(sample["id"])
    batch_questions = tokenizer(
        batch_questions,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
        )
    batch_contexts = tokenizer(
        batch_contexts,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
        )
    batch_answers = tokenizer(
        batch_answers,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )