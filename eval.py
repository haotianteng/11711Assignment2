import datasets
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer, BertForTokenClassification,DataCollatorForTokenClassification,Trainer,TrainingArguments,AutoConfig

def id_mapping(label_names):
    id2label = {i:label for i,label in enumerate(label_names)}
    label2id = {v:k for k,v in id2label.items()}
    return id2label,label2id

def conll_str(sentence, predictions):
    content = []
    curr = 0
    contens = []
    word_start,word_end = [],[]
    for word in sentence.split(' '):
        content.append(word)
        word_start.append(curr)
        word_end.append(curr+len(word))
        curr += len(word)+1
    word_start = np.asarray(word_start)
    word_end = np.asarray(word_end)
    labels = ['O']*len(content)
    for pred in predictions:
        idxs = np.where(np.logical_and(word_start<pred['end'], word_end > pred['start']))[0]
        for i in idxs:
            if labels[i] == 'O':
                labels[i] = pred['entity']
    out = ""
    for i in range(len(content)):
        out += content[i] + ' ' + labels[i] + '\n'
    return out

if __name__ == "__main__":
    #Load dataset for format
    DATASET='havens2/naacl2022'
    print("Loading dataset")
    raw_datasets = datasets.load_dataset(DATASET)
    raw_datasets #Show the structure of raw datasets
    raw_datasets['train'].features
    label_names = raw_datasets['train'].features['ner_tags'].feature.names
    id2label,label2id = id_mapping(label_names)

    #Reading test data
    INPUT_FILE = 'test_data/anlp-sciner-test.txt'
    OUT_FILE = 'test_data/anlp_haotiant_scBertSER.conll'
    out_content = ""
    with open(INPUT_FILE,'r') as f:
        input_content = f.read().rstrip().split('\n')
    max_len = np.max([len(x) for x in input_content])

    MODEL="havens2/scBERT_SER2"
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased',model_max_length=512)
    model = BertForTokenClassification.from_pretrained(MODEL,id2label = id2label,label2id = label2id,ignore_mismatched_sizes=True)
    token_classifier = pipeline("token-classification", model=MODEL, tokenizer = tokenizer, aggregation_strategy="none")


    for line in tqdm(input_content,desc = "Inference:"):
        out = conll_str(line.strip('\n'),token_classifier(line.strip('\n')))
        out_content += out + '\n'
    with open(OUT_FILE,'w+') as f:
        f.write(out_content)

    #print(token_classifier("MedNLI BERT Is Not Immune : Natural Language Inference Artifacts in the Clinical Domain"))
    #print(token_classifier("We used subsets of two datasets : ( 1 ) Yelp -predicting sentiments of restaurant reviews ( positive or negative ) and ( 2 ) Amazon Products -classifying product reviews into one of four categories ( Clothing Shoes and Jewelry , Digital Music , Office Products , or Toys and Games ) ( He and McAuley , 2016 ) . We sampled 500 and 100 examples to be the training data for Yelp and Amazon Products , respectively ."))