#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:22:59 2022

@author: heavens
"""
from lib2to3.pgen2 import token
import os
import sys
import torch
import datasets
import numpy as np
from pynvml import *
import argparse
from transformers import pipeline
from transformers import AutoTokenizer, BertForTokenClassification,DataCollatorForTokenClassification,Trainer,TrainingArguments,AutoConfig
# from huggingface_hub import notebook_login
# notebook_login()

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    
def tokenize_function(sample):
    sample['sentence'] = [' '.join(t) for t in sample['tokens']]
    return tokenizer(sample["sentence"], truncation=True)

def compute_metrics(eval_preds):
    metric = datasets.load_metric("seqeval")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    result =  {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
    print(result)
    return result

def torch_call(self, features):
    import torch
    label_name = "label" if "label" in features[0].keys() else "labels"
    labels = [x.pop('labels') for x in features] if label_name in features[0].keys() else None
    # labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
    batch = self.tokenizer.pad(
        features,
        padding=self.padding,
        max_length=self.max_length,
        pad_to_multiple_of=self.pad_to_multiple_of,
        # Conversion to tensors will fail if we have labels as they are not of the same length yet.
        return_tensors="pt" if labels is None else None,
    )

    if labels is None:
        return batch

    sequence_length = torch.tensor(batch["input_ids"]).shape[1]
    padding_side = self.tokenizer.padding_side
    if padding_side == "right":
        batch[label_name] = [
            list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
        ]
    else:
        batch[label_name] = [
            [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
        ]

    batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
    return batch

def id_mapping(label_names):
    id2label = {i:label for i,label in enumerate(label_names)}
    label2id = {v:k for k,v in id2label.items()}
    return id2label,label2id

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='allenai/scibert_scivocab_uncased')
    parser.add_argument('--model_out', type=str, default='scBERT_SER')
    parser.add_argument('--dataset', type=str, default='conllpp',help="A dataset repository in huggingface, can be conllapp or havens2/naacl2022")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=5e-5)

    args = parser.parse_args(sys.argv[1:]) 
    LR = args.learning_rate
    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_length
    EPOCHS = args.epochs
    MODEL = args.model # 'allenai/scibert_scivocab_uncased'
    MODEL_OUT = args.model_out
    DATASET = args.dataset # conllapp or havens2/naacl2022

    #Load dataset
    print("Loading dataset")
    raw_datasets = datasets.load_dataset(DATASET)
    raw_datasets #Show the structure of raw datasets
    raw_datasets['train'].features
    label_names = raw_datasets['train'].features['ner_tags'].feature.names
    id2label,label2id = id_mapping(label_names)

    #Load pretrained model
    N_LABEL = len(label_names)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = BertForTokenClassification.from_pretrained(MODEL,id2label = id2label,label2id = label2id,ignore_mismatched_sizes=True)
    print_gpu_utilization()
    classifier = pipeline('ner', model=MODEL, tokenizer=tokenizer)

    #Tokenized the datasets
    tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, 
                                          batched=True,
                                          remove_columns=raw_datasets["train"].column_names,)
    # tokenized_datasets = tokenized_datasets.rename_column("ner_tags", "labels")
    tokenized_datasets.set_format("torch")
    tokenized_datasets['train']
    DataCollatorForTokenClassification.torch_call = torch_call
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Customized Training pipeline
    ## Create DataLoader
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], collate_fn=data_collator, batch_size=BATCH_SIZE
    )
    
    # model_ckpt =  #Load from local directory.
    
    ## Create training parameters
    from torch.optim import AdamW
    from accelerate import Accelerator
    optimizer = AdamW(model.parameters(), lr=LR)

    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    from transformers import get_scheduler
    num_train_epochs = EPOCHS
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    ##Set the repository if the model is going to be uploaded
    from huggingface_hub import Repository, get_full_repo_name

    model_name = MODEL_OUT
    repo_name = get_full_repo_name(model_name)
    print("Model repository: ",repo_name)
    output_dir = MODEL_OUT
    repo = Repository(output_dir, clone_from=repo_name)
    
    from tqdm.auto import tqdm
    import torch
    
    progress_bar = tqdm(range(num_training_steps))
    metric = datasets.load_metric("seqeval")
    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
    
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
    
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
    
            # Necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
    
            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
    
            true_predictions, true_labels = postprocess(predictions_gathered, labels_gathered)
            metric.add_batch(predictions=true_predictions, references=true_labels)
    
        results = metric.compute()
        print(
            f"epoch {epoch}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )
    
        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            repo.push_to_hub(
                commit_message=f"Training in progress epoch {epoch}", blocking=False
            )
    
    # #Training pipeline using Huggingface Trainer
    per_device_train_batch_size = 64
    per_device_eval_batch_size = 64
    wd = 0.01 #weight decay
    # training_args = TrainingArguments(output_dir = "scBERT_finetune", 
    #                                   evaluation_strategy = "steps", 
    #                                   save_strategy="epoch",
    #                                   learning_rate=lr, 
    #                                   num_train_epochs=n_epoches, 
    #                                   per_device_train_batch_size=per_device_train_batch_size, 
    #                                   per_device_eval_batch_size=per_device_eval_batch_size, 
    #                                   weight_decay=wd, 
    #                                   push_to_hub = False)
    # trainer = Trainer(model,
    #                   training_args,
    #                   train_dataset=tokenized_datasets['train'],
    #                   eval_dataset=tokenized_datasets['validation'],
    #                   data_collator=data_collator,
    #                   tokenizer=tokenizer,
    #                   compute_metrics = compute_metrics)
    # trainer.train()

    # #Evaluation
    # predictions = trainer.predict(tokenized_datasets['test'])
    # print(predictions.metrics)
