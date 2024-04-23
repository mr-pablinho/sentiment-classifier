# Sentiment Analysis with VADER and RoBERTa

This project uses sentiment analysis to understand emotions behind the reviews posted on Amazon. We employ two different methods: VADER (Valence Aware Dictionary and sEntiment Reasoner), a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media, and RoBERTa (a robustly optimized BERT pretraining approach), a more advanced model that uses machine learning to interpret the context of a word within a sentence.

## Dataset

The dataset used in this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).

Please download the dataset manually and place it in the appropriate directory before running the code.

## Dependencies

Ensure you have Python installed and proceed to install the following packages:

```bash
pip install pandas numpy matplotlib seaborn nltk transformers tqdm
```

## NLTK Data
To get NLTK working, you need to download the necessary datasets. You may need to run the following command in your terminal to overcome SSL certificate issues:

```python
import nltk
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('all')
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.