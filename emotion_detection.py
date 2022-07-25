from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
import torch
import pandas as pd


class EmotionDetection:
    """
    Emotion Detection on text data.

    Attributes:
        tokenizer: An instance of Hugging Face Tokenizer
        model: An instance of Hugging Face Model
        explainer: An instance of SequenceClassificationExplainer from Transformers interpret
    """

    def __init__(self):
        hub_location = 'cardiffnlp/twitter-roberta-base-emotion'
        self.tokenizer = AutoTokenizer.from_pretrained(hub_location)
        self.model = AutoModelForSequenceClassification.from_pretrained(hub_location)
        self.explainer = SequenceClassificationExplainer(self.model, self.tokenizer)

    def justify(self, text):
        """
        Get html annotation for displaying emotion justification over text.

        Parameters:
            text (str): The user input string to emotion justification

        Returns:
            html (hmtl): html object for plotting emotion prediction justification
        """

        word_attributions = self.explainer(text)
        html = self.explainer.visualize("example.html")

        return html

    def classify(self, text):
        """
        Recognize Emotion in text.

        Parameters:
            text (str): The user input string to perform emotion classification on

        Returns:
            predictions (str): The predicted probabilities for emotion classes
        """

        tokens = self.tokenizer.encode_plus(text, add_special_tokens=False, return_tensors='pt')
        outputs = self.model(**tokens)
        probs = torch.nn.functional.softmax(outputs[0], dim=-1)
        probs = probs.mean(dim=0).detach().numpy()
        labels = list(self.model.config.id2label.values())
        preds = pd.Series(probs, index=labels, name='Predicted Probability')

        return preds

    def run(self, text):
        """
        Classify and Justify Emotion in text.

        Parameters:
            text (str): The user input string to perform emotion classification on

        Returns:
            predictions (str): The predicted probabilities for emotion classes
            html (hmtl): html object for plotting emotion prediction justification
        """

        preds = self.classify(text)
        html = self.justify(text)

        return preds, html