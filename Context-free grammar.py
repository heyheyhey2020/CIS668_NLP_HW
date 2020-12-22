#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


HW3_grammar = nltk.PCFG.fromstring ("""
    S -> NP VP NP [0.25]| NP VP PP [0.5]| NP VP [0.25]
    VP -> V NP [0.25]| V P VP [0.25]| Aux V [0.25]| V Adv Adv Adj [0.25] 
    NP -> N [0.143]| Det Adj N [0.143]| Prop [0.571]| Prop N [0.143]
    PP -> Adj N Adv [0.666]| Adv [0.333]
    V -> "had" [0.2]| "came" [0.2]| "go" [0.2]| "visit" [0.2]| "are" [0.2] 
    Prop -> "We" [0.2]| "She" [0.2]| "You" [0.2]| "Their" [0.2]| "me" [0.2] 
    Det -> "a" [1]
    N -> "party" [0.25]| "kids" [0.25]| "yesterday" [0.25]| "days" [0.25]
    P -> "to" [1]
    Adj -> "nice" [0.333]| "naive" [0.333]| "two" [0.333]
    Adv -> "always" [0.25]| "ago" [0.25]| "now" [0.25]| "not" [0.25]
    Aux -> "may" [1]
""")


# In[3]:


HW3_parser = nltk.ViterbiParser(HW3_grammar)


# In[4]:


# HW3_parser = nltk.RecursiveDescentParser(HW3_grammar)


# In[5]:


sen1 = "We had a nice party yesterday"
tree1 = HW3_parser.parse(sen1.split())
for tree in list(tree1):
	print (tree)


# In[6]:


sen2 = "She came to visit me two days ago"
tree2 = HW3_parser.parse(sen2.split())
for tree in list(tree2):
	print (tree)


# In[7]:


sen3 = "You may go now"
tree3 = HW3_parser.parse(sen3.split())
for tree in list(tree3):
	print (tree)


# In[8]:


sen4 = "Their kids are not always naive"
tree4 = HW3_parser.parse(sen4.split())
for tree in list(tree4):
	print (tree)


# In[9]:


# sen5 = "I walk home today"
# tree5 = HW3_parser.parse(sen5.split())
# for tree in list(tree5):
# 	print (tree)


# In[10]:


# sen6 = "Tom will arrive several days later"
# tree6 = HW3_parser.parse(sen6.split())
# for tree in list(tree6):
# 	print (tree)


# In[11]:


# sen7 = "A red paper walks home today"
# tree7 = HW3_parser.parse(sen7.split())
# for tree in list(tree7):
# 	print (tree)

