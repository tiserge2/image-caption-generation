import torch


class AverageMeter(object):
    """Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(preds, targets, k):
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def calculate_caption_lengths(word_dict, captions):
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['start'], word_dict['eos']):
                continue
            else:
                lengths += 1
    return lengths


from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from tabulate import tabulate

def show_metrics(metrics):
    """
    Presents the provided metrics in a visually appealing tabular format.

    Args:
        metrics (dict): A dictionary containing the metrics to be displayed.
    """

    table_data = []
    # Create table data with header row and metric entries
    for metric, value in metrics.items():
        table_data.append([metric, value])

    # Additional headers for table columns (metric names)
    headers = ['Metric', 'Value'] + list(range(1, len(table_data) + 1))

    # Print the table using tabulate library with desired format
    print(tabulate(table_data, headers=headers, tablefmt='fancy_grid'))


class MetricsCalculator:
    """
    This class calculates various evaluation metrics for text generation tasks using pycocoevalcap.
    Please before using this class do: pip install pycocoevalcap
    Also you need to have java 1.8.0 installed

    Attributes:
        references (dict): A dictionary where keys are arbitrary identifiers (e.g., "ref") and values are lists of reference captions.
        hypotheses (dict): A dictionary where keys are the same identifiers as in 'references' and values are the predicted captions.
        n_gram (int): The maximum n-gram size for BLEU score calculation (default: 4).
        bleu (Bleu): An object from the `pycocoevalcap.bleu` module for BLEU score computation.
        rouge (Rouge): An object from the `pycocoevalcap.rouge` module for ROUGE score computation.
        cider (Cider): An object from the `pycocoevalcap.cider` module for CIDEr score computation.
        meteor (Meteor): An object from the `pycocoevalcap.meteor` module for METEOR score computation.
        scores (dict): A dictionary to store the computed scores for each metric.
    """

    def __init__(self, references, hypotheses, n_gram=4):
        """
        Initializes the class with references, hypotheses, and the n-gram size for BLEU.

        Args:
            references (dict): A dictionary containing reference captions. (see Attributes)
            hypotheses (dict): A dictionary containing predicted captions. (see Attributes)
            n_gram (int, optional): The maximum n-gram size for BLEU score calculation. Defaults to 4.
        """

        self.references = references
        self.hypotheses = hypotheses
        self.n_gram = n_gram
        self.bleu = Bleu(self.n_gram)
        self.rouge = Rouge()
        self.cider = Cider()
        self.meteor = Meteor()
        self.scores = {}

    def compute_bleu(self):
        """
        Calculates BLEU scores (up to n-gram size specified in 'n_gram') and stores them in the 'scores' dictionary.

        Converts the scores to percentages before storing.
        """

        bleu_scores = self.bleu.compute_score(self.references, self.hypotheses)
        for i in range(self.n_gram):
            self.scores[f"BLEU-{i+1}"] = f"{bleu_scores[0][i] * 100:.2f}"  # Convert to percentage

    def compute_rouge_l(self):
        """
        Calculates ROUGE-L score and stores it in the 'scores' dictionary.

        Converts the score to a percentage before storing.
        """

        rouge_scores, _ = self.rouge.compute_score(self.references, self.hypotheses)
        self.scores["ROUGE-L"] = f"{rouge_scores * 100:.2f}"

    def compute_cider(self):
        """
        Calculates CIDEr score and stores it in the 'scores' dictionary.

        Note: The second output from 'compute_score' is discarded here (not used).
        """

        scores, _ = self.cider.compute_score(self.references, self.hypotheses)
        self.scores["CIDEr"] = scores

    def compute_meteor(self):
        """
        Calculates METEOR score and stores it in the 'scores' dictionary.

        Converts the score to a percentage before storing.
        """

        scores, _ = self.meteor.compute_score(self.references, self.hypotheses)
        self.scores["METEOR"] = f"{scores * 100:.2f}"



if __name__ == "__main__":
    # Example usage
    hypothesis = {"ref_1": ["Transformers Transformers are fast plus efficient"]}
    hypothesis["ref_2"] = ["This is a generated sentence."]

    references = {"ref_1": [
        "HuggingFace Transformers are quick, efficient and awesome",
        "Transformers are awesome because they are fast to execute",
        "Good Morning Transformers",
        "People are eagerly waiting for new Transformer models",
        "People are very excited about new Transformers"
    ]}

    references["ref_2"] = ["This is the reference sentence.", "Another possible reference."]


    # Create an instance of MetricsCalculator
    metrics_calculator = MetricsCalculator(references, hypothesis)

    # Compute BLEU scores for n=1, 2, 3 and 4
    metrics_calculator.compute_bleu()

    # Compute ROUGE-N score for n=1
    metrics_calculator.compute_rouge_l()

    # Compute ROUGE-L score
    metrics_calculator.compute_rouge_l()

    # Compute CIDEr score
    metrics_calculator.compute_cider()

    # Compute Meteor score
    metrics_calculator.compute_meteor()

    scores = metrics_calculator.scores

    show_metrics(scores)
