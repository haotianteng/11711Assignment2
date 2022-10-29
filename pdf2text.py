#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 08:30:10 2022

@author: heavens
"""
import io
import argparse
import os
import re
from tqdm import tqdm
import pdfminer
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
import spacy

# Perform layout analysis for all text
laparams = pdfminer.layout.LAParams()
setattr(laparams, 'all_texts', True)

def pdf2text(pdf_path, text_path,nlp):
    """
    Convert pdf to text
    """
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=laparams)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)

        content = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    content = content.replace('\xad','')
    content = content.replace('\u200b\x07','')
    text = ''
    for line in content.split('\n'):
        line = line.strip()
        if len(line) == 0:
            text += '\n'
            continue
        if line[-1] == '-':
            line = line[:-1]
        else:
            line += ' '
        text += line
    MAX_LEN = 3000000
    nlp.max_length = MAX_LEN
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN]
    doc = nlp(text)
#    print(doc)
    with open(text_path, 'w+') as out:
        for sent in doc.sents:
            s_text = sent.text.strip()
            if len(s_text) > 2:
                out.write(s_text+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required = True, help='Input folder.')
    parser.add_argument('-o', '--output', required = True, help='Output folder.')
    args = parser.parse_args()
    nlp = spacy.load("en_core_web_sm")
    if not os.path.exists(args.input):
        print('Input file does not exist')
        exit()
    os.makedirs(args.output,exist_ok = True)
    for file in tqdm(os.listdir(args.input)):
        if file.endswith('.pdf'):
            filename = os.path.splitext(file)[0]
            pdf2text(os.path.join(args.input,file), os.path.join(args.output,filename+'.txt'),nlp)
