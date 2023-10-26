#!/usr/bin/python3
import re
import nltk
import contractions

nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

# 1) Load data from a CSV file into a Pandas dataframe
df = pd.read_csv('./data/wikihowALL.csv', delimiter=',',engine='python', error_bad_lines=False, nrows=3000)
df.info()

pd.set_option('display.max_colwidth', -1)  
df.head(1)

# Dropping rows with empty fields 
# df[df.isnull().any(axis=1)]
df = df.dropna()


# 2) Preprocessing Start
 
# Applying Contractions 
# I'll -> I will; I'd -> I would 
df['text'] = df['text'].apply(lambda x: contractions.fix(x))
df['headline'] = df['headline'].apply(lambda x: contractions.fix(x))
df['title'] = df['title'].apply(lambda x: contractions.fix(x))
# df['overview'] = df['overview'].apply(lambda x: contractions.fix(x))

# Removing stop words
STOP_WORDS = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in STOP_WORDS]))
df['headline'] = df['headline'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in STOP_WORDS]))
# df['overview'] = df['overview'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in STOP_WORDS]))

print(df.head(1)['text'])

# Removing any special charecters
def clean_text(text):
    REGEX_SPECIAL_CHARS = '[^A-Za-z0-9]+'
    text = re.sub(REGEX_SPECIAL_CHARS, ' ', text).strip()
    return text
df['text'] = df['text'].apply(clean_text)
# df['overview'] = df['overview'].apply(clean_text)
df['headline'] = df['headline'].apply(clean_text)

print(df.head(1)['text'])

# 3 Saving the preprocessed text
df.to_csv('./data/prepocessedALL.csv', index=False)