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


device = 'cuda:0'
cmd1 = 'cat paraphrase.out | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > paraphrase.out.tokenized'
all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

mname_paraphrase = './bart-paraphrase'

#model_paraphrase = BartForConditionalGeneration.from_pretrained(mname_paraphrase).to(device)
model_paraphrase = BartForConditionalGeneration.from_pretrained(mname_paraphrase)
model_paraphrase.eval()
tokenizer_paraphrase = BartTokenizer.from_pretrained(mname_paraphrase)

required_num = 2

beam_size = 16


generate_target_dir = "./cnndm_p" + str(required_num) + "_lc" + "/diverse/"




tgt_dir = './paraphrase.out'
tgt_dir_tokenized = './paraphrase.out.tokenized'



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

  return output_list
    
  


 
def generate_paraphrase_dataset(split):
  
  
  source_path = './brio/BRIO/cnndm/diverse/' + split
  
  files = os.listdir(source_path)
  with open(tgt_dir_tokenized, 'r') as f1:
    with open(tgt_dir, 'r') as f2:
      for i in range(len(files)):
      
        f = open(source_path + "/" + str(i) +  '.json')
      
        source = json.load(f)
        
        f.close()
        
        now_paraphrasing_tokenized = []
        now_paraphrasing = []
        for j in range(required_num):
          l1 = f1.readline()
          l2 = f2.readline()
          
          now_paraphrasing_tokenized.append(l1)
          now_paraphrasing.append(l2)
        
        
        
        
        cands = [sent_tokenize(x) for x in now_paraphrasing_tokenized]
        cands_untok = [sent_tokenize(x) for x in now_paraphrasing]   
        
        _abstract = "\n".join(source["abstract"])
        
        def compute_rouge(hyp):
          score = all_scorer.score(_abstract, "\n".join(hyp))
          return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
        
        candidates = [[x, compute_rouge(x)] for x in cands]
        
        
        candidates_untok = [[cands_untok[i], candidates[i][1]] for i in range(len(candidates))]
        
        select_index = np.random.choice(range(len(source['candidates'])), size=required_num, replace=False)
        for j in range(required_num):
          source['candidates'][select_index[j]] = candidates[j]
          source['candidates_untok'][select_index[j]] = candidates_untok[j]
          
        with open(generate_target_dir + split + "/" + str(i) +  '.json', "w") as outfile:
          json.dump(source, outfile)
          outfile.close()

  
    

  os.system(cmd1)

  with open(tgt_dir_tokenized, 'r') as f1:
    with open(tgt_dir, 'r') as f2:
      for i in range(len(files)):
      
        f = open(source_path + "/" + str(i) +  '.json')
      
        source = json.load(f)
        
        f.close()
        
        now_paraphrasing_tokenized = []
        now_paraphrasing = []
        for j in range(required_num):
          l1 = f1.readline()
          l2 = f2.readline()
          
          now_paraphrasing_tokenized.append(l1)
          now_paraphrasing.append(l2)
        
        
        
        
        cands = [sent_tokenize(x) for x in now_paraphrasing_tokenized]
        cands_untok = [sent_tokenize(x) for x in now_paraphrasing]   
        
        _abstract = "\n".join(source["abstract"])
        
        def compute_rouge(hyp):
          score = all_scorer.score(_abstract, "\n".join(hyp))
          return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3
        
        candidates = [[x, compute_rouge(x)] for x in cands]
        
        
        candidates_untok = [[cands_untok[i], candidates[i][1]] for i in range(len(candidates))]
        
        select_index = np.random.choice(range(len(source['candidates'])), size=required_num, replace=False)
        for j in range(required_num):
          source['candidates'][select_index[j]] = candidates[j]
          source['candidates_untok'][select_index[j]] = candidates_untok[j]
          
        with open(generate_target_dir + split + "/" + str(i) +  '.json', "w") as outfile:
          json.dump(source, outfile)
          outfile.close()

 
        
 
          

        
        
  
  
if __name__ ==  "__main__":
    
    print("nice")
    generate_paraphrase_dataset('train')


    
    

          
        
    
  
  
  
  
    