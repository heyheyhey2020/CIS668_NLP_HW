#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk . corpus import stopwords
from nltk import FreqDist
from nltk.collocations import *
from nltk . stem import WordNetLemmatizer
import xlwt


# In[2]:


#open file_1
corpus_root = '/Users/lc/Desktop/CIS 668/HW/hw1'
wordlists = PlaintextCorpusReader(corpus_root,'.*')


# In[3]:


#tokenizing
wordlist1 = wordlists.words(fileids = 'state_union_part1.txt')


# In[4]:


#remove len(words) < 2
wordlist1 = [ w for w in wordlist1 if len(w) > 2]


# In[5]:


#translate into low case format
wordlist1 = [ w.lower() for w in wordlist1 if w.isalpha()]


# In[6]:


#removing meaningless stop words
stopwordSet = set(stopwords.words('english'))
wordlist1 = [ w for w in wordlist1 if w not in stopwordSet]


# In[7]:


#lemmatizatio
wordlist1 = [ WordNetLemmatizer ().lemmatize(w) for w in wordlist1]


# In[8]:


freWords1 = FreqDist(wordlist1)
wordlist1_freq = freWords1.most_common(50)
# wordlist1_freq


# In[9]:


bigram_measures = nltk.collocations.BigramAssocMeasures()
finder1 = BigramCollocationFinder.from_words(wordlist1)
bigram1_freq = finder1.score_ngrams(bigram_measures.raw_freq)
bigram1_freq = bigram1_freq[:50]


# In[10]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


wordlist2 = wordlists.words(fileids = 'state_union_part2.txt')
wordlist2 = [ w.lower() for w in wordlist2 if w.isalpha()]
wordlist2 = [ w for w in wordlist2 if len(w) > 2]
wordlist2 = [ w for w in wordlist2 if w not in stopwordSet]
wordlist2 = [ WordNetLemmatizer ().lemmatize(w) for w in wordlist2]


# In[12]:


freWords = FreqDist(wordlist2)
wordlist2_freq = freWords.most_common(50)
# wordlist2_freq


# In[42]:


finder2 = BigramCollocationFinder.from_words(wordlist2)
bigram2_freq = finder2.score_ngrams(bigram_measures.raw_freq)
bigram2_freq = bigram2_freq[:50]
bigram2_freq


# In[14]:


finder2. apply_freq_filter (5)
bigram2_pmi = finder2.score_ngrams(bigram_measures.pmi)
bigram2_pmi = bigram2_pmi[:50]


# In[16]:


workbook1 = xlwt.Workbook(encoding='utf-8') 
sheet1 = workbook1.add_sheet("wordlist1_freq",cell_overwrite_ok=True) 
for i in range(len(wordlist1_freq)):
    sheet1.write(i,0,wordlist1_freq[i][0])
    sheet1.write(i,1,wordlist1_freq[i][1])
    i += 1
workbook1.save('/Users/lc/Desktop/CIS 668/HW/hw1/wordlist1_freq.xls') 


# In[17]:


workbook2 = xlwt.Workbook(encoding='utf-8') 
sheet2 = workbook2.add_sheet("wordlist2_freq",cell_overwrite_ok=True) 
for i in range(len(wordlist2_freq)):
    sheet2.write(i,0,wordlist2_freq[i][0])
    sheet2.write(i,1,wordlist2_freq[i][1])
    i += 1
workbook2.save('/Users/lc/Desktop/CIS 668/HW/hw1/wordlist2_freq.xls') 


# In[18]:


workbook3 = xlwt.Workbook(encoding='utf-8') 
sheet3 = workbook3.add_sheet("bigram1_freq",cell_overwrite_ok=True) 
for i in range(len(bigram1_freq)):
    sheet3.write(i,0,bigram1_freq[i][0][0])
    sheet3.write(i,1,bigram1_freq[i][0][1])
    i += 1
workbook3.save('/Users/lc/Desktop/CIS 668/HW/hw1/bigram1_freq.xls') 


# In[19]:


workbook4 = xlwt.Workbook(encoding='utf-8') 
sheet4 = workbook4.add_sheet("bigram2_freq",cell_overwrite_ok=True) 
for i in range(len(bigram2_freq)):
    sheet4.write(i,0,bigram2_freq[i][0][0])
    sheet4.write(i,1,bigram2_freq[i][0][1])
    i += 1
workbook4.save('/Users/lc/Desktop/CIS 668/HW/hw1/bigram2_freq.xls') 


# In[20]:


workbook5 = xlwt.Workbook(encoding='utf-8') 
sheet5 = workbook5.add_sheet("bigram1_pmi",cell_overwrite_ok=True) 
for i in range(len(bigram1_pmi)):
    sheet5.write(i,0,bigram1_pmi[i][0][0])
    sheet5.write(i,1,bigram1_pmi[i][0][1])
    i += 1
workbook5.save('/Users/lc/Desktop/CIS 668/HW/hw1/bigram1_pmi.xls') 


# In[21]:


workbook6 = xlwt.Workbook(encoding='utf-8') 
sheet6 = workbook6.add_sheet("bigram2_pmi",cell_overwrite_ok=True) 
for i in range(len(bigram2_freq)):
    sheet6.write(i,0,bigram2_pmi[i][0][0])
    sheet6.write(i,1,bigram2_pmi[i][0][1])
    i += 1
workbook6.save('/Users/lc/Desktop/CIS 668/HW/hw1/bigram2_pmi.xls') 


# In[28]:


list1 = [ pair[0] for pair in wordlist1_freq]
list2 = [ pair[0] for pair in wordlist2_freq]
freq_same = list(set(list1)&set(list2))
freq_same


# In[29]:


list1_diff = list(set(list1)-set(freq_same))
list1_diff


# In[31]:


list2_diff = list(set(list2)-set(freq_same))
list2_diff


# In[32]:


print(len(freq_same),len(list1_diff),len(list2_diff))


# In[53]:


list1 = [ pair[0] for pair in bigram1_freq]
list2 = [ pair[0] for pair in bigram2_freq]
bgrFreq_same = list(set(list1)&set(list2)) 
print(bgrFreq_same)
print(len(bgrFreq_same))


# In[56]:


# list1_diff = list(set(list1)-set(bgrFreq_same))
# list2_diff = list(set(list2)-set(bgrFreq_same))
# print(list1_diff)
# print(list2_diff)


# In[57]:


list1 = [ pair[0] for pair in bigram1_pmi]
list2 = [ pair[0] for pair in bigram2_pmi]
bgrPmi_same = list(set(list1)&set(list2)) 
print(bgrPmi_same)
print(len(bgrPmi_same))


# In[ ]:




