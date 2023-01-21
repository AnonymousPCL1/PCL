import nltk
from nltk.util import ngrams
#ref is input documents here, out is inference summary
def compare(trigrams1, trigrams2):
    common=[]
    for grams1 in trigrams1:
        if grams1 in trigrams2:
            common.append(grams1)
    return common
f_ref = open("../../candidates_nyt/test.source.tokenized")
#f_ref = open("./cnndm/test.source.tokenized")
#f_out = open("ref.token")
#f_out = open("./nyt.out.tokenized")
f_out = open("./result/23-01-09-37/test.out.tokenized")
#f_out = open("./novel.out")
ref = f_ref.readlines()
out = f_out.readlines()
n = 3
#ref = ref[:len(out)]
statistics = []
for each_ref, each_out in zip(ref, out):
    grams_ref = list(ngrams(each_ref.split(), n) )
    grams_out = list(ngrams(each_out.split(), n) )
    #print(grams_ref)
    common = compare(grams_out, grams_ref)
    statistics.append( 1- (len(set(common)) / len(set(grams_out))) )

print(sum(statistics)/len(statistics))
n = 2
statistics = []
for each_ref, each_out in zip(ref, out):
    grams_ref = list(ngrams(each_ref.split(), n) )
    grams_out = list(ngrams(each_out.split(), n) )
    #print(grams_ref)
    common = compare(grams_out, grams_ref)
    statistics.append( 1- (len(set(common)) / len(set(grams_out))) )

print(sum(statistics)/len(statistics))

n = 1
statistics = []
for each_ref, each_out in zip(ref, out):
    grams_ref = list(ngrams(each_ref.split(), n) )
    grams_out = list(ngrams(each_out.split(), n) )
    #print(grams_ref)
    common = compare(grams_out, grams_ref)
    statistics.append( 1- (len(set(common)) / len(set(grams_out))) )

print(sum(statistics)/len(statistics))



