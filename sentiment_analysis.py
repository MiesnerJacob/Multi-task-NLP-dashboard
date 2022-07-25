from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
import torch
import pandas as pd


class SentimentAnalysis:
    """
    Sentiment on text data.

    Attributes:
        tokenizer: An instance of Hugging Face Tokenizer
        model: An instance of Hugging Face Model
        explainer: An instance of SequenceClassificationExplainer from Transformers interpret
    """

    def __init__(self):
        # Load Tokenizer & Model
        hub_location = 'cardiffnlp/twitter-roberta-base-sentiment'
        self.tokenizer = AutoTokenizer.from_pretrained(hub_location)
        self.model = AutoModelForSequenceClassification.from_pretrained(hub_location)

        # Change model labels in config
        self.model.config.id2label[0] = "Negative"
        self.model.config.id2label[1] = "Neutral"
        self.model.config.id2label[2] = "Positive"
        self.model.config.label2id["Negative"] = self.model.config.label2id.pop("LABEL_0")
        self.model.config.label2id["Neutral"] = self.model.config.label2id.pop("LABEL_1")
        self.model.config.label2id["Positive"] = self.model.config.label2id.pop("LABEL_2")

        # Instantiate explainer
        self.explainer = SequenceClassificationExplainer(self.model, self.tokenizer)

    def justify(self, text):
        """
        Get html annotation for displaying sentiment justification over text.

        Parameters:
            text (str): The user input string to sentiment justification

        Returns:
            html (hmtl): html object for plotting sentiment prediction justification
        """

        word_attributions = self.explainer(text)
        html = self.explainer.visualize("example.html")

        return html

    def classify(self, text):
        """
        Recognize Sentiment in text.

        Parameters:
            text (str): The user input string to perform sentiment classification on

        Returns:
            predictions (str): The predicted probabilities for sentiment classes
        """

        tokens = self.tokenizer.encode_plus(text, add_special_tokens=False, return_tensors='pt')
        outputs = self.model(**tokens)
        probs = torch.nn.functional.softmax(outputs[0], dim=-1)
        probs = probs.mean(dim=0).detach().numpy()
        predictions = pd.Series(probs, index=["Negative", "Neutral", "Positive"], name='Predicted Probability')

        return predictions

    def run(self, text):
        """
        Classify and Justify Sentiment in text.

        Parameters:
            text (str): The user input string to perform sentiment classification on

        Returns:
            predictions (str): The predicted probabilities for sentiment classes
            html (hmtl): html object for plotting sentiment prediction justification
        """

        predictions = self.classify(text)
        html = self.justify(text)

        return predictions, html