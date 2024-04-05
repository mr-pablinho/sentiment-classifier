# Sentiment Analysis

## Dataset

The dataset used in this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

Please download the dataset manually and place it in the appropriate directory before running the code.

## NLTK Data
To get NLTK working, you need to download the necessary datasets. You may need to run the following command in your terminal to overcome SSL certificate issues.

```python
import nltk
import sl

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('all')
```
