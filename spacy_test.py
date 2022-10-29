#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 09:56:55 2022

@author: heavens
"""

# import spacy
# # Load English tokenizer, tagger, parser and NER
# nlp = spacy.load("en_core_web_sm")
# # Process whole documents
# text = ("When Sebastian Thrun started working on self-driving cars at "
#         "Google in 2007, few people outside of the company took him "
#         "seriously. “I can tell you very senior CEOs of major American "
#         "car companies would shake my hand and turn away because I wasn’t "
#         "worth talking to,” said Thrun, in an interview with Recode earlier "
#         "this week.")
# doc = nlp(text)
# # Analyze syntax
# print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
# print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
# # Find named entities, phrases and concepts
# for entity in doc.ents:
#     print(entity.text, entity.label_)
    

from PyPDF2 import PdfReader
f = "/home/heavens/twilight/CMU/Semester7/11711AdvNLP/assignments/assignment2/tacl-2022/2022.tacl-1.10.pdf"
pdf = PdfReader(f)
for page in pdf.pages:
    break
    for line in page.extract_text().split('\n'):
        break
    
from pdfminer.high_level import extract_text
text = extract_text(f)