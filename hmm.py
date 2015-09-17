import re
import random
import math
import os
import time
from collections import Counter

BROWN_CORPUS_DIR = 'brown'
REUTER_CORPUS_DIR_TEST = 'test_tok'
REUTER_CORPUS_DIR_TRAIN = 'traning_tok'
TOKEN_DIR = 'tokenizer'
TEST_CORPUS_DIR = 'test_copus'
TEST_REUTER_DIR = 'test_reuter'
TEST_DIR = 'test'
TRAIN_CORPUS_DIR = 'train_copus'
TRAIN_REUTER_DIR = 'train_reuter'
NUM_OF_TRAIN_FILES = 490
DICTIONARY_DIR = 'dictionary'



WORD = 'word'
WORD_TAG = 'word_tag'
UNIGRAM = 'unigram'
BIGRAM = 'bigram'
TRIGRAM = 'trigram'
UNIGRAM_PROB = 'unigram_prob'
BIGRAM_PROB = 'bigram_prob'
TRIGRAM_PROB = 'trigram_prob'
POSSIBLE_TAGS = 'possible_tags'
FILE_TEST_TAG_ORIGIN = 'test_tag_origin'
FILE_TEST = 'test'

def save_trained_data(data, filename, brown_corpus):
    if brown_corpus:
        dir_path = DICTIONARY_DIR + "/" + TRAIN_CORPUS_DIR
    else:
        dir_path = DICTIONARY_DIR + "/" + TRAIN_REUTER_DIR

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + '/' + filename
    file = open(file_path, 'w')
    file.write(str(data))
    file.close()

def get_trained_data(filename, brown_corpus):
    if brown_corpus:
        dir_path = DICTIONARY_DIR + "/" + TRAIN_CORPUS_DIR
    else:
        dir_path = DICTIONARY_DIR + "/" + TRAIN_REUTER_DIR
    
    file_path = dir_path + '/' + filename
    file = open(file_path, 'r')
    file_content = file.read()
    file.close()
    return eval(file_content)

#For printing various dictionaries
def print_hash(hash):
    for item in hash:
        print(item,hash[item])

#Unigram generation
# Input:preprocessed file name
# Output: unigram probability, unigram frequency dictionary
def unigram(filename, isBrown):
    
    if isBrown:
        dir_path = DICTIONARY_DIR + "/" + TRAIN_CORPUS_DIR
    else:
        dir_path = DICTIONARY_DIR + "/" + TRAIN_REUTER_DIR
    
    if os.path.exists(dir_path + "/" + UNIGRAM):
        unigram_prob = get_trained_data(UNIGRAM_PROB, isBrown)
        unigram_hash = get_trained_data(UNIGRAM, isBrown)
    else:
        fproc = open(filename, 'r').read()
        token = fproc.split()
        unigram_hash = dict(Counter(token).items())
        unigram_prob= dict(unigram_hash)
        unigram_count = sum(unigram_hash.values())
        for value in unigram_hash.keys():
            temp = float(unigram_hash.get(value))/float(unigram_count)
            unigram_prob[value] = temp
        save_trained_data(unigram_prob, UNIGRAM_PROB, isBrown)
        save_trained_data(unigram_hash, UNIGRAM, isBrown)
    
    return(unigram_prob,unigram_hash)

#Bigram generation
# Input: Preprocessed file name, unigram frequency
# Output: bigram probability, bigram frequency dictionary
def bigram(filename,unigram_hash, isBrown):
    
    if isBrown:
        dir_path = DICTIONARY_DIR + "/" + TRAIN_CORPUS_DIR
    else:
        dir_path = DICTIONARY_DIR + "/" + TRAIN_REUTER_DIR
    
    if os.path.exists(dir_path + "/" + BIGRAM):
        bigram_prob = get_trained_data(BIGRAM_PROB, isBrown)
        bigram_hash = get_trained_data(BIGRAM, isBrown)
    else:
        newlist = [] #to append the generated bigrams
        fproc = open(filename, 'r').read()
        token = fproc.split()
        i=0
        bigramlist = token
            
        while i<len(bigramlist):
            if i+1<(len(bigramlist) -1):
                newlist.append(bigramlist[i]+" " + bigramlist[i+1])
            i += 1
        
        bigram_hash = dict(Counter(newlist).items())
        bigram_prob = dict(bigram_hash)
            
        for w in newlist[:]:
            first=w.split(" ")
            bigramfrequency= bigram_hash.get(w)
            unifrequency=unigram_hash.get(first[0])
            temp= float(bigramfrequency)/float(unifrequency)
            bigram_prob[w]= temp

        save_trained_data(bigram_prob, BIGRAM_PROB, isBrown)
        save_trained_data(bigram_hash, BIGRAM, isBrown)

    return(bigram_prob,bigram_hash)

# Trigram generation
# Input: Preprocessed file name, bigram frequency
# Output: trigram probability, trigram frequency dictionary
def trigram(filename, bigram_hash, isBrown):
    if isBrown:
        dir_path = DICTIONARY_DIR + "/" + TRAIN_CORPUS_DIR
    else:
        dir_path = DICTIONARY_DIR + "/" + TRAIN_REUTER_DIR
    
    if os.path.exists(dir_path + "/" + TRIGRAM):
        trigram_prob = get_trained_data(TRIGRAM_PROB, isBrown)
        trigram_hash = get_trained_data(TRIGRAM, isBrown)
    else:
        newlist1 = [] #to append the generated trigrams
        fproc = open(filename, 'r').read()
        token = fproc.split()
        j=0
        trigramlist = token
            
        while j<len(trigramlist):
            if (j+2)<(len(trigramlist) -1):
                newlist1.append(trigramlist[j]+" " + trigramlist[j+1]+" "+trigramlist[j+2])
            j += 1
        
        trigram_hash = dict(Counter(newlist1).items())
        trigram_prob = dict(trigram_hash)
            
        for w in newlist1[:]:
            first1=w.split(" ")
            trigramfrequency1= trigram_hash.get(w)
            seq=[first1[0], first1[1]]
            bigram_split=" ".join(seq)
            bigramfrequency1=bigram_hash.get(bigram_split)
            temp= float(trigramfrequency1)/float(bigramfrequency1)
            trigram_prob[w]= temp
        save_trained_data(trigram_prob, TRIGRAM_PROB, isBrown)
        save_trained_data(trigram_hash, TRIGRAM, isBrown)
    
    return(trigram_prob,trigram_hash)

#Check if the count is present in the hash_count dictionary(which has
# been precalculated to hold the Nc values.
def countN(hash,count):
    if count in hash:
        return hash[count]
    else:
        return 0

#Implement the good turning smoothing.
#Input: The unigram/bigram/trigram dictionary
#Output: The smoothed probabilities
def good_turing_smoothing(fd):
    gt_temp = dict()
    count_hash = dict()
    fd['<UNK>'] = 0 #Include an entry in the hash to handle unknowns
    
    count_hash = dict(Counter(fd.values()).most_common())
    N = sum(fd.values()) + 1
    
    for sample in fd:
        count = fd[sample]
        if count < 1:
            gt_temp[sample] = float (countN(count_hash,1)) / float(N)
                
        if count> 1 and count < 5:
            nc = countN(count_hash,count)
            ncn = countN(count_hash,count + 1)
                            
            if nc ==0 or ncn == 0:
                gt_temp[sample] = float(fd[sample])/float(N)
            else:
                gt_temp[sample] = float(count + 1) * float(float(ncn) / float(nc * N))

    print_hash(gt_temp)
                                    
    return gt_temp

#Calculate the perplexity
#Input: Unigram/bigram/trigram frequency table, ngram -1(unigram) 2(bigram) 3(trigram)
# and the preprocessed file name
#Output: Perplexity value

def get_p_sentence(sentence):
    sentence = sentence.split()
    n = len(sentence)
    p_sentence = 1.0
    for k in range(1, n + 1):
        word = get_word(sentence, k - 1)
        last_word = get_word(sentence, k - 2)
        penult_word = get_word(sentence, k - 3)
        
        p_sentence *= get_q_trigram(penult_word, last_word, word)
    
    return (p_sentence, n)

def perplexity(prob_hash, n_gram,filename):
    
    fproc = open(filename, 'r').read()
    token = fproc.split()
    M = len(token)
                
    newlist_per = [] #for bi/trigram model
                    
    p=0
    i=0
    j=0
                                
    if n_gram == 1:
        for word in token:
            if word in prob_hash:
                p+= math.log(prob_hash[word],2)
            else:
                p+= math.log(prob_hash["<UNK>"],2)
                                                        
    elif n_gram == 2:
        while i<len(token):
            if i+1<(len(token) -1):
                newlist_per.append(token[i]+" " + token[i+1])
            i += 1
                                                                            
            for word in newlist_per:
                if word in prob_hash:
                    p+= math.log(prob_hash[word],2)
                else:
                    p+= math.log(prob_hash["<UNK>"],2)
                                                                                                
    elif n_gram == 3:
        while j<M:
            if (j+2)<(len(token) -1):
                print(">>" + token[j]+" " + token[j+1]+" "+token[j+2])
                newlist_per.append(token[j]+" " + token[j+1]+" "+token[j+2])
            j+= 1
                                                                                                                    
            for word in newlist_per:
                if word in prob_hash:
                    p+= math.log(prob_hash[word],2)
#                else:
#                    p+= math.log(prob_hash["<UNK>"],2)

    l= float(p)/float(M)
    perplexity = 2 ** (-1 *l)
    return (perplexity)

# Random Sentence
# Input: Bigram/Trigram frequency dictionary, ngram - 2 for bigram and 3 for trigram
# Output: Prints the random sentences
def random_sentence(prob_hash, ngram):
    
    sentence_len = 0
    random_sentence = ''
    maxlen = 30
            
    if ngram==2:
        while (sentence_len < maxlen):
            random_p = random.uniform(0,1)
                        
            for key,value in prob_hash.items():
                if sentence_len==0:
                    if((value == random_p and key.split()[0] == '<s>') or (value < random_p + 0.1 and value > random_p - 0.1 and key.split()[0] == '<s>')):
                            random_sentence += key.split()[1] + ' '
                            prev = key.split()[1]
                            sentence_len += 1
                elif sentence_len == maxlen -1:
                    if((value == random_p and key.split()[1] == '</s>') or (value < random_p + 0.1 and value > random_p - 0.1 and key.split()[1] == '</s>')):
                        random_sentence += key.split()[0]
                        sentence_len += 1
                else:
                    if((value == random_p and key.split()[0] == prev) or (value < random_p + 0.1 and value > random_p - 0.1 and key.split()[0] == prev)):
                        random_sentence +=  key.split()[1] + ' '
                        prev = key.split()[1]
                        sentence_len += 1
    elif ngram==3:
        while (sentence_len < maxlen):
            random_p = random.uniform(0,1)
                        
            for key,value in prob_hash.items():
                if sentence_len==0:
                    if((value == random_p and key.split()[0] == '<s>') or (value < random_p + 0.1 and value > random_p - 0.1 and key.split()[0] == '<s>')):
                        random_sentence += key.split()[1]+' '+key.split()[2] + ' '
                        prev1= key.split()[1]
                        prev2 = key.split()[2]
                        sentence_len += 2
                        break
                elif sentence_len >= maxlen -2:
                    if((value == random_p and key.split()[2] == '</s>') or (value < random_p + 0.1 and value > random_p - 0.1 and key.split()[2] == '</s>')):
                        random_sentence += key.split()[0]+' '+key.split()[1]
                        sentence_len += 2
                        break
                else:
                    if((value == random_p and key.split()[0] == prev1 and key.split()[1]==prev2) or (value < random_p + 0.1 and value > random_p - 0.1 and key.split()[0] == prev1 and key.split()[1]==prev2)):
                        random_sentence +=  key.split()[2] + ' '
                        prev1 = key.split()[1]
                        prev2= key.split()[2]
                        sentence_len += 1
                        break
            
    random_sentence = re.sub(' </s> <s>', "", random_sentence)
    random_sentence = re.sub('</s>', "", random_sentence)
    random_sentence = re.sub('<s>', "", random_sentence)
    print(random_sentence)

def main():
    prepareData()
    oper = -1
    trainFile = TOKEN_DIR + "/" + TRAIN_REUTER_DIR
    while int(oper) != 0:
        print('')
        print('Choose one of the following: ')
        print('1 - Unigram Model')
        print('2 - Bigram Model')
        print('3 - Trigram Model')
        print('4 - Random Sentence Generation')
        print('5 - Perplexity')
        print('0 - Exit')
        print('')
        oper = int(input("Enter your options: "))
        
        if oper > 0:
            if oper == 1: #Unigram Model
                unigram_prob,unigram_hash = unigram(trainFile, False)
                print("Unigram probability"+"\n")
                print_hash(unigram_prob)
            elif oper == 2: #Bigram Model
                unigram_prob,unigram_hash = unigram(trainFile, False)
                bigram_prob,bigram_hash = bigram(trainFile,unigram_hash, False)
                print("Bigram probability"+"\n")
                print_hash(bigram_hash)
            elif oper == 3: #Trigram Model
                unigram_prob,unigram_hash = unigram(trainFile, False)
                bigram_prob,bigram_hash = bigram(trainFile,unigram_hash, False)
                trigram_prob,trigram_hash = trigram(trainFile,bigram_hash, False)
                print("Trigram probability"+"\n")
                print_hash(trigram_prob)
            elif oper == 4: #Random Sentence Generation
                unigram_prob,unigram_hash = unigram(trainFile, False)
                bigram_prob,bigram_hash = bigram(trainFile,unigram_hash, False)
                trigram_prob,trigram_hash = trigram(trainFile,bigram_hash, False)
                print("Random sentences using bigram model:"+"\n")
                for i in range(5):
                    random_sentence(bigram_prob,2)
                print("Random sentences using trigram model:"+"\n")
                for i in range(5):
                    random_sentence(trigram_prob,3)
            elif oper == 5: #Perplexity
                test_file = TOKEN_DIR + "/" + TEST_REUTER_DIR
                unigram_prob,unigram_hash = unigram(trainFile, False)
                bigram_prob,bigram_hash = bigram(trainFile,unigram_hash, False)
                trigram_prob,trigram_hash = trigram(trainFile,bigram_hash, False)

#                gt_prob = dict(unigram_prob)
#                gt_bi_prob = dict(bigram_prob)
#                                
#                gt_prob = good_turing_smoothing(unigram_hash)
#                print("Good Turing smoothing - unigram")
##                print_hash(gt_prob)
#                gt_tri_prob = good_turing_smoothing(trigram_hash)
#                for key in gt_bi_prob.keys():
#                    bigram_prob[key] = gt_bi_prob[key]
#                print("Good Turing smoothing - bigram"+"\n")
#                print_hash(gt_bi_prob)

                unigram_perplexity = perplexity(trigram_prob,3,test_file)
#                bigram_perplexity = perplexity(gt_bi_prob,2,test_file)
#                trigram_perplexity = perplexity(trigram_prob,3,test_file)
                print("Unigram perplexity is "+ str(unigram_perplexity))
#                print("Bigram perplexity is "+ str(bigram_perplexity))
#                print("Trigram perplexity is "+ str(trigram_perplexity))
            else:
                exit()

def prepareDataBrown():
    print("Prepare Brown:"+"\n")
    
    list_of_filename = os.listdir(BROWN_CORPUS_DIR)
    
    content = ''
    for file in list_of_filename[:NUM_OF_TRAIN_FILES]:
        if len(file) == 4:
            with open(BROWN_CORPUS_DIR + '/' + file, 'r') as file_content:
                lines = file_content.readlines()
                for line in lines:
                    if line and line.strip():
                        for word_tag in line.split():
                            current_word = word_tag.split('/', 1)[0]
                            content += current_word + ' '
                        content += '\n'
            file_content.close()

    file_path = TOKEN_DIR + "/" + TRAIN_CORPUS_DIR
    with open(file_path, "w+") as file:
        file.write(str(content))
    file.close()
    
    content = ''
    for file in list_of_filename[NUM_OF_TRAIN_FILES:]:
        if len(file) == 4:
            with open(BROWN_CORPUS_DIR + '/' + file, 'r') as file_content:
                lines = file_content.readlines()
                for line in lines:
                    if line and line.strip():
                        for word_tag in line.split():
                            current_word = word_tag.split('/', 1)[0]
                            content += current_word + ' '
                        content += '\n'
            file_content.close()

    file_path = TOKEN_DIR + '/' + TEST_CORPUS_DIR
    with open(file_path, "w+") as file:
        file.write(str(content))
    file.close()

def prepareDataReuter():
    list_of_reuter_test = os.listdir(REUTER_CORPUS_DIR_TEST)
    rTest = open(TOKEN_DIR + "/" + TEST_REUTER_DIR,"w+")
    for filename in list_of_reuter_test:
        if re.match('[0-9]', filename) is not None:
            with open(REUTER_CORPUS_DIR_TEST + "/" + filename, encoding = "ISO-8859-1") as reuter_file_test:
                lines = reuter_file_test.readlines()
                for line in lines:
                    rTest.write(line)
    rTest.close()

    list_of_reuter_train = os.listdir(REUTER_CORPUS_DIR_TRAIN)
    rTrain = open(TOKEN_DIR + "/" + TRAIN_REUTER_DIR,"w+")
    for filename in list_of_reuter_train:
        if re.match('[0-9]', filename) is not None:
            with open(REUTER_CORPUS_DIR_TRAIN + "/" + filename, encoding = "ISO-8859-1") as reuter_file_train:
                lines = reuter_file_train.readlines()
                for line in lines:
                    rTrain.write(line)
    rTrain.close()

def prepareData():
    print("Prepare data:"+"\n")
    start_time = time.time()
    if os.path.isdir(TOKEN_DIR):
        #Read data
        content = ''
    else:
        os.makedirs(TOKEN_DIR)
        #Brown Corpus
        prepareDataBrown()
        
        #Reuter Corpus
        prepareDataReuter()
        
    end_time = time.time()
    print('(Time to initialize data: %s)' % (end_time - start_time))

if __name__ == "__main__":
    main()

