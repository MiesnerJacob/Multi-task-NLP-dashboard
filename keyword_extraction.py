import nltk
import pytextrank
import re
from operator import itemgetter
import en_core_web_sm


class KeywordExtractor:
    """
    Keyword Extraction on text data

    Attributes:
        nlp: An instance English pipeline optimized for CPU for spacy
    """

    def __init__(self):
        self.nlp = en_core_web_sm.load()
        self.nlp.add_pipe("textrank")

    def get_keywords(self, text, max_keywords):
        """
        Extract keywords from text.

        Parameters:
            text (str): The user input string to extract keywords from

        Returns:
            kws (list): list of extracted keywords
        """

        doc = self.nlp(text)

        kws = [i.text for i in doc._.phrases[:max_keywords]]

        return kws

    def get_keyword_indices(self, kws, text):
        """
        Extract keywords from text.

        Parameters:
            kws (list): list of extracted keywords
            text (str): The user input string to extract keywords from

        Returns:
            keyword_indices (list): list of indices for keyword boundaries in text
        """

        keyword_indices = []
        for s in kws:
            indices = [[m.start(), m.end()] for m in re.finditer(re.escape(s), text)]
            keyword_indices.extend(indices)

        return keyword_indices

    def merge_overlapping_indices(self, keyword_indices):
        """
        Merge overlapping keyword indices.

        Parameters:
            keyword_indices (list): list of indices for keyword boundaries in text

        Returns:
            keyword_indices (list): list of indices for keyword boundaries in with overlapping combined
        """

        # Sort the array on the basis of start values of intervals.
        keyword_indices.sort()

        stack = []
        # insert first interval into stack
        stack.append(keyword_indices[0])
        for i in keyword_indices[1:]:
            # Check for overlapping interval,
            # if interval overlap
            if (stack[-1][0] <= i[0] <= stack[-1][-1]) or (stack[-1][-1] == i[0]-1):
                stack[-1][-1] = max(stack[-1][-1], i[-1])
            else:
                stack.append(i)
        return stack

    def merge_until_finished(self, keyword_indices):
        """
        Loop until no overlapping keyword indices left.

        Parameters:
            keyword_indices (list): list of indices for keyword boundaries in text

        Returns:
            keyword_indices (list): list of indices for keyword boundaries in with overlapping combined
        """

        len_indices = 0
        while True:
            # Merge overlapping indices
            merged = self.merge_overlapping_indices(keyword_indices)
            # Check to see if merging reduced number of annotation indices
            # If merging did not reduce list return final indicies
            if len_indices == len(merged):
                out_indices = sorted(merged, key=itemgetter(0))
                return out_indices
            else:
                len_indices = len(merged)

    def get_annotation(self, text, keyword_indices):
        """
        Create text annotation for extracted keywords.

        Parameters:
            keyword_indices (list): list of indices for keyword boundaries in text

        Returns:
            annotation (list): list of tuples for generating html
        """

        # Turn list to numpy array
        arr = list(text)

        # Loop through indices in list and insert delimeters
        for idx in sorted(keyword_indices, reverse=True):
            arr.insert(idx[0], "<kw>")
            arr.insert(idx[1]+1, "<!kw> <kw>")

        # join array
        joined_annotation = ''.join(arr)

        # split array on delimeter
        split = joined_annotation.split('<kw>')

        # Create annotation for keywords in text
        annotation = [(x.replace('<!kw> ', ''), "KEY", "#26aaef") if "<!kw>" in x else x for x in split]

        return annotation

    def generate(self, text, max_keywords):
        """
        Create text annotation for extracted keywords.

        Parameters:
            text (str): The user input string to extract keywords from
            max_keywords (int): Limit on number of keywords to generate

        Returns:
            annotation (list): list of tuples for generating html
            kws (list): list of extracted keywords
        """

        kws = self.get_keywords(text, max_keywords)

        indices = list(self.get_keyword_indices(kws, text))
        if indices:
            indices_merged = self.merge_until_finished(indices)
            annotation = self.get_annotation(text, indices_merged)
        else:
            annotation = None

        return annotation, kws

