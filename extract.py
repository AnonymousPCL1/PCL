import os
import sys
import argparse
from typing import List
import numpy as np
import tracemalloc
import random

extract_num = 3

paraphrase_only = False

source_tgt1 = "paraphrase_PLUS1.out"

tgt1 = "muli4.out" 

source_tgt2 = "paraphrase_PLUS1.source"

tgt2 = "muli4.source" 

count = 0

para_num = 6

for line in open(source_tgt1, "r"):
  count += 1

print(count)

with open(source_tgt1, "r") as source1:
  with open(tgt1, "w") as fout1:
    for i in range(int(count/(para_num+1))):
      for j in range(para_num + 1):
        line = source1.readline()
        
        if paraphrase_only:
          if j > 0 and j < (extract_num + 1):
            fout1.write(line)
        else:
          if j >= 0 and j < (extract_num):
            fout1.write(line)
  source1.close()
  fout1.close()
  
  
with open(source_tgt2, "r") as source2:
  with open(tgt2, "w") as fout2:
    for i in range(int(count/(para_num+1))):
      for j in range(para_num + 1):
        line = source2.readline()
        
        if paraphrase_only:
          if j > 0 and j < (extract_num + 1):
            fout2.write(line)
        else:
          if j >= 0 and j < (extract_num):
            fout2.write(line)
  source2.close()
  fout2.close()


tgt_dir = './muli4.out'

tgt_dir1 = './muli4.source'

final_tgt_dir = './train.target'

final_tgt_dir1 = './train.source'


random.seed(8)


with open(tgt_dir,'r') as source1:
  with open(tgt_dir1,'r') as source2:
    data1 = [line for line in source1]
    data2 = [line for line in source2]
    data = list(zip(data1, data2))
random.shuffle(data)
data1, data2 = zip(*data)
   
   
with open(final_tgt_dir,'w') as target:
    for line in data1:
        target.write(line)
        


  

with open(final_tgt_dir1,'w') as target:
    for line in data2:
        target.write(line)

print('finish')
        
    