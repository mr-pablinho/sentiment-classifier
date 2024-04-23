from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize models and tokenizer
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
sia = SentimentIntensityAnalyzer()

def run_sentiment_analysis(df):
    """
    This function takes a DataFrame with a 'Text' column and performs sentiment analysis using both
    VADER and RoBERTa. It adds the results as new columns to the DataFrame.

    :param df: DataFrame containing the 'Text' column with review texts.
    :return: DataFrame with original data and new columns for sentiment scores.
    """
    # Initialize result dictionary
    sentiment_analysis = {}

    # Process each review in the DataFrame
    for index, row in df.iterrows():
        text = row['Text']
        sentiment_scores = {}

        try:
            # Run VADER analysis
            vader_scores = sia.polarity_scores(text)
            sentiment_scores.update({
                'vader_neg': vader_scores['neg'],
                'vader_neu': vader_scores['neu'],
                'vader_pos': vader_scores['pos'],
                'vader_compound': vader_scores['compound']
            })

            # Run RoBERTa analysis
            encoded_text = tokenizer(text, return_tensors='pt')
            output = model(**encoded_text)
            scores = softmax(output.logits[0].detach().numpy())
            sentiment_scores.update({
                'roberta_neg': scores[0],
                'roberta_neu': scores[1],
                'roberta_pos': scores[2]
            })

        except RuntimeError as e:
            print(f"Error processing index {index}: {e}")
            # You might want to continue or handle the specific error
            continue

        # Append scores to the dictionary
        sentiment_analysis[index] = sentiment_scores

    # Convert dictionary to DataFrame
    sentiment_df = pd.DataFrame.from_dict(sentiment_analysis, orient='index')
    return df.join(sentiment_df)


def plot_sentiment_comparison(df):
    """
    Plots a pairplot to compare the sentiment scores from VADER and RoBERTa.

    :param df: DataFrame containing the sentiment scores from both VADER and RoBERTa.
    """
    # Create the pairplot
    pair_plot = sns.pairplot(
        data=df,
        markers='o',
        vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos'],
        hue='Score',
        palette=['red', 'orange', 'goldenrod', 'green', 'blue'],
        plot_kws={'alpha': 0.5},
        height=1.5,  # Height of each subplot
        aspect=1     # Aspect ratio of each subplot (width/height)
    )

    # Set legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Display the plot
    plt.show()


if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('data/amazon_reviews.csv')

    # Reduce the dataset size for testing. take the first 1000 rows
    df = df.head(100)

    # Perform sentiment analysis
    df = run_sentiment_analysis(df)
    plot_sentiment_comparison(df)

    # Save the results
    df.to_csv('data/sentiment_analysis.csv', index=False)
