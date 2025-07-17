#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Tong Minh Hieu Le
# #### Student ID: 4098368
# 
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# * RegexpTokenizer 
# * chain
# * ...
# 
# ## Introduction
# 
# In this file, we perform basic text pre-processing on the given dataset, including, but not limited to tokenization, removing most/least frequent words and stop words. In this task, we focus on pre-processing the “Review Text” only.
# 1. Extract information about the review. Perform the pre-processing steps mentioned below to the extracted reviews
# 2. Tokenize each clothing review. The word tokenization must use the following regular expression, r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?";
# 3. All the words must be converted into the lower case;
# 4. Remove words with a length less than 2.
# 5. Remove stopwords using the provided stop words list (i.e., stopwords_en.txt). It is located
# inside the same downloaded folder.
# 6. Remove the word that appears only once in the document collection, based on term frequency.
# 7. Remove the top 20 most frequent words based on document frequency.
# 8. Save the processed data as processed.csv file.
# 9. Build a vocabulary of the cleaned/processed reviews, and save it in a txt file (please refer to the
# Required Output section);

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
import pandas as pd
import numpy as np
from nltk import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from itertools import chain
from nltk.probability import *


# ### 1.1 Examining and loading data
# - Examine the data and explain your findings
# - Load the data into proper data structures and get it ready for processing.

# In[2]:


# Code to inspect the provided data file...
# Loading the data
df = pd.read_csv('assignment3.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


# Code to check for missing values 
df.isnull().any().sum()


# In[6]:


# Check data types
df.dtypes


# In[7]:


df.describe()


# ### 1.2 Pre-processing data
# Perform the required text pre-processing steps.
# 
# 1. **Extract review information**: Extract the text from the "Review Text" column for processing.
# 
# 2. **Tokenize reviews**: Use the regular expression `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"` to tokenize each clothing review into individual words.
# 
# 3. **Convert to lowercase**: Transform all words to lowercase to ensure consistency.
# 
# 4. **Remove short words**: Remove words with length less than 2 characters.
# 
# 5. **Remove stopwords**: Filter out common stopwords using the provided stopwords_en.txt file.
# 
# 6. **Remove infrequent words**: Eliminate words that appear only once in the entire collection.
# 
# 7. **Remove most frequent words**: Remove the top 20 most frequent words based on document frequency.
# 
# 8. **Save processed data**: Store the processed reviews in a CSV file named "processed.csv".

# #### 1.2.1. Extract review information**: Extract the text from the "Review Text" column for processing.

# In[8]:


reviews = df['Review Text']


# In[9]:


len(reviews)


# #### 1.2.2 + 1.2.3: Tokenize reviews and convert to lowercase

# In[10]:


def tokenizeReview(raw_review):
    """
    This function converts all words to lowercase,
    segments the raw review into sentences, tokenizes each sentence
    and returns the review as a list of tokens.
    """
    # Handle NaN or non-string values
    if not isinstance(raw_review, str):
        return []
        
    nl_review = raw_review.lower()  # convert all words to lowercase
    
    # segment into sentences
    sentences = sent_tokenize(nl_review)
    
    # tokenize each sentence
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern)
    token_lists = [tokenizer.tokenize(sen) for sen in sentences]
    
    # merge them into a list of tokens
    tokenised_review = list(chain.from_iterable(token_lists))
    return tokenised_review


# In[11]:


tk_reviews = [tokenizeReview(r) for r in reviews]  # list comprehension, generate a list of tokenized articles


# In[12]:


def stats_print(tk_reviews):
    words = list(chain.from_iterable(tk_reviews)) # we put all the tokens in the corpus in a single list
    vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
    lexical_diversity = len(vocab)/len(words)
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of reviews:", len(tk_reviews))
    lens = [len(article) for article in tk_reviews]
    print("Average review length:", np.mean(lens))
    print("Maximun review length:", np.max(lens))
    print("Minimun review length:", np.min(lens))
    print("Standard deviation of review length:", np.std(lens))


# In[13]:


# index to test element in tk reviews
test_index = 1


# In[14]:


print("Raw review:\n",reviews[test_index],'\n')
print("Tokenized review:\n",tk_reviews[test_index])


# In[15]:


stats_print(tk_reviews)


# #### 1.2.4. Remove short words**: Remove words with length less than 2 characters.

# In[16]:


# filter out single character tokens
tk_reviews = [[w for w in review if len(w) >=2]                       for review in tk_reviews]


# In[17]:


print("Tokenized review:\n",tk_reviews[test_index])


# In[18]:


stats_print(tk_reviews)


# #### 1.2.5. Remove stopwords**: Filter out common stopwords using the provided stopwords_en.txt file.

# In[19]:


# Loading the stop words
with open('stopwords_en.txt', 'r') as f:
    stopwords = f.read().splitlines()


# In[20]:


len(stopwords)


# In[21]:


# Filter out stopwords from tokenized reviews
tk_reviews = [[w for w in review if w not in stopwords] 
              for review in tk_reviews]

# Check the result on the sample review
print("Tokenized review after removing stopwords:\n", tk_reviews[test_index])


# In[22]:


stats_print(tk_reviews)


# #### 1.2.6. Remove infrequent words**: Eliminate words that appear only once in the document collection, based on term frequency

# In[23]:


words = list(chain.from_iterable(tk_reviews)) # we put all the tokens in the corpus in a single list
term_freq = FreqDist(words) # compute the term frequency distribution 


# In[24]:


# Find the less frequent words
lessFreqWords = set(term_freq.hapaxes())
lessFreqWords 


# In[25]:


len(lessFreqWords)


# In[26]:


def removeLessFreqWords(review):
    return [w for w in review if w not in lessFreqWords]

tk_reviews = [removeLessFreqWords(review) for review in tk_reviews]


# In[27]:


stats_print(tk_reviews)


# #### 1.2.7. Remove most frequent words**: Remove the top 20 most frequent words based on document frequency.

# In[28]:


words_2 = list(chain.from_iterable([set(review) for review in tk_reviews]))


# In[29]:


doc_fd = FreqDist(words_2)  # compute document frequency for each unique word/type
mostCommonFredWords = doc_fd.most_common(20) # top 20 most frequent words
mostCommonFredWords


# In[30]:


# Extract the words from the most common words
most_common_words = [word for word, freq in mostCommonFredWords]
most_common_words


# In[31]:


def removeMostFreqWords(review):
    return [w for w in review if w not in most_common_words]

tk_reviews = [removeMostFreqWords(review) for review in tk_reviews]


# In[32]:


stats_print(tk_reviews)


# #### 1.2.8. Save processed data**: Store the processed reviews in a CSV file named "processed.csv".

# In[33]:


print("Raw review:\n",reviews[test_index],'\n')
print("Tokenized review:\n",tk_reviews[test_index])


# In[34]:


joined_reviews = [' '.join(review) for review in tk_reviews]
joined_reviews[test_index]


# In[35]:


# We save processed reviews to a file with the new column name
df['Processed Review Text'] = joined_reviews
df.head()


# In[36]:


# Check the count of null values in both DataFrames
print(f"Null values in process reviews: {df['Processed Review Text'].isnull().sum()}")


# In[37]:


# Check for empty strings too
print(f"Empty strings in process reviews: {(df['Processed Review Text'] == '').sum()}")
display(df.shape)


# In[38]:


# Remove rows with null or empty strings in the 'Processed Review Text' column because they are not useful for our analysis
df = df[(df['Processed Review Text'].notna()) & (df['Processed Review Text'] != '')]


# In[39]:


# Check it again
print(f"Empty strings in process reviews: {(df['Processed Review Text'] == '').sum()}")


# In[40]:


df.shape


# In[41]:


# Save the processed DataFrame to a new CSV file
df.to_csv('processed.csv', index=False)


# ## Saving required outputs
# Save the requested information as per specification.
# - vocab.txt

# In[42]:


stats_print(tk_reviews)


# In[43]:


# generating the vocabulary
words_3 = list(chain.from_iterable(tk_reviews)) # we put all the tokens in the corpus in a single list
vocab = sorted(list(set(words_3))) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
vocab[:10] # print the first 10 words in the vocabulary
len(vocab) 


# In[44]:


# Save the vocabulary to vocab.txt
with open('vocab.txt', 'w') as f:
    for i, word in enumerate(vocab):
        f.write(f"{word}:{i}\n") 


# ## Summary
# Give a short summary and anything you would like to talk about the assessment task here.

# ## Reference
# - Activities and labs files for this course. 
# - Github Copilot
