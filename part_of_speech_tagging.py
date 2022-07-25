import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class POSTagging:
    """Part of Speech Tagging on text data"""

    def __init__(self):
       pass

    def classify(self, text):
        """
        Generate Part of Speech tags.

        Parameters:
            text (str): The user input string to generate tags for

        Returns:
            predictions (list): list of tuples containing words and their respective tags
        """

        text = word_tokenize(text)
        predictions = nltk.pos_tag(text)
        return predictions