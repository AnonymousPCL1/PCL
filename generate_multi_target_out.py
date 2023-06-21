import os
import json
from transformers import BartForConditionalGeneration, BartTokenizer, PegasusTokenizer, PegasusForConditionalGeneration
import torch
import sys
import argparse
from typing import List
import numpy as np
from compare_mt.rouge.rouge_scorer import RougeScorer
from nltk import sent_tokenize
import tracemalloc
import random

device = 'cuda:0'
cmd1 = 'cat paraphrase.out | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > paraphrase.out.tokenized'
all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

mname_paraphrase = './bart-paraphrase'

model_paraphrase = BartForConditionalGeneration.from_pretrained(mname_paraphrase).to(device)
#model_paraphrase = PegasusForConditionalGeneration.from_pretrained(mname_paraphrase)
model_paraphrase.eval()
tokenizer_paraphrase = BartTokenizer.from_pretrained(mname_paraphrase)

required_num = 6

beam_size = 16

early_break = True

generate_target_dir = "./cnndm_p" + str(required_num) + "_lc" + "/diverse/"




tgt_dir = './paraphrase_PLUS1.out'


tgt_dir1 = './paraphrase_PLUS1.source'


final_tgt_dir = './paraphrase_PLUS.out'

final_tgt_dir1 = './paraphrase_PLUS.source'

def sampling_paraphrase(list1):
  
  big_set = set()
  old_length = 0
  
  ranging = range(beam_size)
  
  output_list = []
  
  for i in range(required_num):
    finding = True
    
    while(finding):
      
      selected_ouput = ''
      
      for j in range(len(list1)): 
        selected_ouput = selected_ouput + ' ' + list1[j][np.random.choice(ranging)]
      selected_ouput = selected_ouput[1:] 
      big_set = big_set.union( {selected_ouput} )
      if(old_length != len( big_set ) ):
        old_length += 1
        finding = False
        output_list.append(selected_ouput) 
      '''
      print(big_set)  
      '''
  return output_list
    
  


 
def generate_paraphrase_dataset(split):
  
  
  source_path = './brio/BRIO/cnndm/diverse/' + split
  
  files = os.listdir(source_path)
  with open(tgt_dir, 'w') as fout:
    with open(tgt_dir1, 'w') as fout1: 
      for i in range(80000):
      
        f = open(source_path + "/" + str(i) +  '.json')
        
        source = json.load(f)
        f.close()
    
        abstract = source['abstract']
        
        abstract_ = " ".join(abstract)
        
        fout.write(abstract_ + '\n')
        
        fout.flush()
        
        print(" ".join(source['abstract']) )
        
        big_list = []
        
        
        
        source_text = " ".join(source['article'])
        
        
        fout1.write(source_text + '\n')
        
        fout1.flush()
        
        
        for sentence in abstract:
          
          with torch.no_grad():
            
            
            dct_paraphrase = tokenizer_paraphrase.batch_encode_plus([sentence], max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
            
            
            try:             
              summaries_paraphrase = model_paraphrase.generate(
                                  input_ids=dct_paraphrase["input_ids"].to(device),
                                  attention_mask=dct_paraphrase["attention_mask"].to(device),
                                  num_return_sequences=beam_size, num_beam_groups=beam_size, diversity_penalty=1.0, num_beams=beam_size,
                                  max_length=140,  # +2 from original because we start at step=1 and stop before max_length
                                  min_length=min(len(sentence.split()), 1),  # +1 from original because we start at step=1
                                  no_repeat_ngram_size=2,
                                  length_penalty=2.0,
                                  early_stopping=True,
                              )
                              
            except:
              print(sentence)
              print(abstract)
              summaries_paraphrase = model_paraphrase.generate(
                                  input_ids=dct_paraphrase["input_ids"].to(device),
                                  attention_mask=dct_paraphrase["attention_mask"].to(device),
                                  num_return_sequences=beam_size, num_beam_groups=beam_size, diversity_penalty=1.0, num_beams=beam_size,
                                  max_length=120,  # +2 from original because we start at step=1 and stop before max_length
                                  min_length=1,  # +1 from original because we start at step=1
                                  no_repeat_ngram_size=3,
                                  length_penalty=2.0,
                                  early_stopping=True,
                              )
                            
            
            dec_paraphrase = [tokenizer_paraphrase.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries_paraphrase]
            
            
            
            big_list.append(dec_paraphrase)
      
        selected = sampling_paraphrase(big_list)
        
        
    
              
        
        for hypothesis in selected:
          hypothesis = hypothesis.replace("\n", " ")
          fout.write(hypothesis + '\n')
          fout.flush()
          fout1.write(source_text + '\n')
          fout1.flush()
  
  fout1.close()
  fout.close()
  
'''  
  random.seed(8)
  
  with open(tgt_dir,'r') as source:
    data = [line for line in source]
  random.shuffle(data)
  with open(final_tgt_dir,'w') as target:
      for line in data:
          target.write(line)
          
  random.seed(8)
  with open(tgt_dir1,'r') as source:
    data = [line for line in source]
  random.shuffle(data)
  with open(final_tgt_dir1,'w') as target:
      for line in data:
          target.write(line)
'''
    

  
  
if __name__ ==  "__main__":
    
    print("nice")
    generate_paraphrase_dataset('train')


    
    

          
        
    
  
  
  
  
    