import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.data.encode_sequence import decode_sequence
from uuid import uuid4
import os
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from tabulate import tabulate

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
    """
    Calculate the top-k accuracy of predictions given the targets.

    Args:
        preds (torch.Tensor): Predictions from the model of shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels of shape (batch_size).
        k (int): Top-k value for accuracy calculation.

    Returns:
        float: Top-k accuracy in percentage.
    """
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def calculate_caption_lengths(word_dict, captions):
    """
    Calculate the total number of non-special tokens in captions.

    Args:
        word_dict (dict): Dictionary containing special tokens like '<start>', '<eos>', '<pad>'.
        captions (list of lists): List of tokenized captions where each inner list represents a caption.

    Returns:
        int: Total number of non-special tokens in all captions combined.
    """
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['<start>'], word_dict['<eos>'], word_dict['<pad>']):
                continue
            else:
                lengths += 1
    return lengths


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
    metrics = tabulate(table_data, headers=headers, tablefmt='fancy_grid')
    print(metrics)
    return metrics


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


def get_attention_heatmap(image, context, word_index, tokenizer=None):
    """
    Visualizes the attention map overlaid on the original image.

    Parameters:
    - image (numpy.ndarray): The original image as a NumPy array.
    - context (torch.Tensor): The attention map context.
    - word (str): The actual word corresponding to the attention map.

    """
    # load tokenizer 
    tokens_path = r"/home/sagemaker-user/rscid/data/processed/tokenizer.p"
    
    word = decode_sequence([word_index], tokenizer)
    
    # Ensure the attention map is 2D
    attention_map = context.squeeze().detach().numpy()
    attention_map = np.uint8(255 * attention_map).reshape((16, 32))
    
     # Normalize the attention map
    attention_map = cv2.normalize(attention_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Resize the attention map to match the image size using OpenCV
    attention_map_resized = cv2.resize(attention_map, (image.shape[1], image.shape[0]))

    # Apply a colormap to the attention map
    heatmap = cv2.applyColorMap(np.uint8(255 * attention_map_resized), cv2.COLORMAP_JET)

    # Overlay the heatmap on the image
    overlayed_image = cv2.addWeighted(image, 0.9, heatmap, 0.3, 2)

    return {"attention_map_resized": attention_map_resized, "overlayed_image": overlayed_image, "word": word}

def save_visualize_attention(data, save_att_vis_path):
    """
    Save visualizations of attention for each word in a given image.

    Args:
        data (dict): A dictionary containing 'image' and 'word_attentions'.
                     'image' should be the original image tensor.
                     'word_attentions' should be a list of dictionaries, each containing:
                         - 'word': The word corresponding to the attention visualization.
                         - 'attention_map_resized': Resized attention map tensor.
                         - 'overlayed_image': Image with overlaid attention visualization.
        save_att_vis_path (str): Path to the directory where visualizations will be saved.
    """

    image = data['image']
    word_attentions = data['word_attentions']
    fig, axes = plt.subplots(len(word_attentions), 3, figsize=(26, 26))
    save_path = os.path.join(save_att_vis_path, f"{str(uuid4())}.png")

    for index, word_attention in enumerate(word_attentions):
        # Plot the original image, attention map, and the overlayed image    
        axes[index, 0].set_title(f"Original Image with word: {word_attention['word']}")
        axes[index, 0].imshow(image, cmap='viridis')
        axes[index, 0].axis('off')
    
        axes[index, 1].set_title("Attention Map")
        axes[index, 1].imshow(word_attention['attention_map_resized'], cmap='jet')
        axes[index, 1].axis('off')
    
        axes[index, 2].set_title("Overlayed Image")
        axes[index, 2].imshow(word_attention['overlayed_image'])
        axes[index, 2].axis('off')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to fit description
    plt.savefig(save_path)
    plt.close()


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