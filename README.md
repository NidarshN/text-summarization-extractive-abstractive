# Text Summarization on BBC News Corpus using Extractive and Abstractive based approaches

## Introduction

The project is aimed to perform text summarization on BBC News Corpus using Extractive and Abstractive based Approaches.


# Dataset
<a href="https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive">BBC News Corpus</a> consists a total of 2225 documents from the BBC News website corresponding to stories in five tropical areas from 2004 - 2005.

It is categorized into 5 labels:
- sport
- business
- politics
- tech
- entertainment

# Abstraction Summarization

We have used T5ForConditionalGeneration pretrained model, which we then retrained for our BBC News Corpus for Text Abstraction.

# Extractive Summarization

We have used Text Rank algorithm to perform Extractive Summarization using the nltk framework.

# Evaluation

For evaluation of both models, we utilised rouge scores by averaging the following metrics rouge-1, rouge-2, rouge-3, rouge-n, rouge-l and rouge-w for each test data and further averaging across the whole dataset in accordance with the F1-Score, Precision and Recall rate.

## Installation

1. Create Conda Environment using Environment yml file inside the project structure</li>
```
cd <THIS_PROJECT_DIRECTORY>
conda env create --name base -f environment.yml
```
2. Activate the Conda Environment
```
conda activate ./base
```
3. Run the text_summarize.ipynb jupyter notebook file

``` 
jupyter notebook <THIS_PROJECT_DIRECTORY>/Notebooks/text_summarize.ipynb
```






