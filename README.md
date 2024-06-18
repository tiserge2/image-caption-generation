# Remote Sensing Caption Generation with Encoder-Decoder Architecture
![linux-tested][linux-tested] ![mac-tested][mac-tested] 

[***Introduction***](https://github.com/tiserge2/image-caption-generation#introduction)

[**Background and Theory**](https://github.com/tiserge2/image-caption-generation#background-and-theory)

[***Repository Description***](https://github.com/tiserge2/image-caption-generation#repository-description)

[***Implementation Details***](https://github.com/tiserge2/image-caption-generation#implementation-details)

[***Results***](https://github.com/tiserge2/image-caption-generation#results)

[***Usage***](https://github.com/tiserge2/image-caption-generation#usage)

[***Challenges and Future Work***](https://github.com/tiserge2/image-caption-generation#challenges-and-future-work)

[***How to Contribute***](https://github.com/tiserge2/image-caption-generation#how-to-contribute)

[***References***](https://github.com/tiserge2/image-caption-generation#references)


## 1. Introduction
### Overview of the Project
<!-- Briefly describe what remote sensing caption generation is.  
Introduce the GitHub repository and its primary objective. -->

Remote sensing caption generation is the process of creating descriptive text for remote sensing images. 
This GitHub repository contains an implementation of the paper [Exploring Models and Data for Remote Sensing Image Caption Generation](https://arxiv.org/pdf/1712.07835). 
This work was completed as part of my Master 2 class, "Modelisation of Vision System," in the track "Vision and Intelligent Machines."

### Importance and Applications
<!-- Explain why this project is significant.  
Highlight potential applications in real-world scenarios (e.g., environmental monitoring, disaster management). -->

The project I implemented is significant in the field of remote sensing images due to the inherent complexity and vast detail contained within such images. 
Remote sensing images often cover large geographic areas and include numerous intricate features, 
making it challenging for humans to accurately and efficiently describe their content. 
Automated caption generation addresses this challenge by providing a systematic way to generate meaningful descriptions, 
which can significantly enhance the accessibility and usability of remote sensing data. 
This capability is crucial for various applications, such as agricultural monitoring, disaster management and urban planning. 
By automating the description process, this project facilitates quicker and more accurate interpretation of remote sensing images, 
ultimately supporting better decision-making and resource management in these critical areas.


## 2. Background and Theory
### Remote Sensing
<!-- Define remote sensing and its importance.  
Types of remote sensing data (e.g., satellite imagery, aerial photography). -->
With significant advancements in the field of imaging and photography, we are currently witnessing an explosion of high-quality image acquisition methods. 
One of these methods is remote sensing, a technique for acquiring images remotely using aerial devices such as drones, airplanes, and satellites. 
Remote sensing images are particularly rich in information, as they are captured at high resolution.

<div style="text-align:center;">
    <figure>
    <img src="https://github.com/tiserge2/image-caption-generation/blob/main/data/external/industrial_331.jpg" 
        width="200"
        alt="Remote Sensing image of a residential area" />
    <figcaption style="font-size: 8px;">Remote Sensing image of an industrial area (example image)</figcaption>
    </figure>
</div>


### Caption Generation
Image caption generation is the process of recognizing the context of an image and annotating it with relevant captions using deep learning and computer vision. 
It includes labeling an image with keywords using datasets provided during model training.

### Encoder-Decoder Architecture
<!-- Explain the concepts of encoder and decoder.  
Discuss how they work together for caption generation. -->

Encoder-decoder are method usded to accomplish the task of caption generation in a single pipeline. 

<div style="text-align:center;">
    <figure>
    <img src="https://github.com/tiserge2/image-caption-generation/blob/main/data/external/encoder-decoder-arch.png" 
        width="500"
        alt="Remote Sensing image of a residential area" />
    <figcaption style="font-size: 8px;">Encoder-Decoder architecture for caption generation task</figcaption>
    </figure>
</div>

The encoder part consist of deep learning CNN which aims at extracting features from the input images. 
The Encoder encodes the input image with 3 color channels into a smaller image with "learned" channels.
This smaller encoded image is a summary representation of all that's useful in the original image.

<div style="text-align:center;">
    <figure>
    <img src="https://github.com/tiserge2/image-caption-generation/blob/main/data/external/encoder.png" 
        width="500"
        alt="Remote Sensing image of a residential area" />
    <figcaption style="font-size: 8px;">Encoder features extraction</figcaption>
    </figure>
</div>

The decoder part is reponsible to look at the encoded image and generate caption word by word while focusing on 
important part with attention mechanism. Since we are generating sequence, we need to use Recurrent Neural Network (RNN). I have used an LSTM there.

<div style="text-align:center;">
    <figure>
    <img src="https://github.com/tiserge2/image-caption-generation/blob/main/data/external/decoder.png" 
        width="600"
        alt="Remote Sensing image of a residential area" />
    <figcaption style="font-size: 8px;">Decoder used to generate the sequence</figcaption>
    </figure>
</div>

## 3. Repository Description
<!-- ### Repository Information
Provide the name of the repository, author(s), and the date it was created.  
Link to the repository. -->

### Structure of the Repository
<!-- List and describe the main directories and files (e.g., `src`, `data`, `models`, `notebooks`, `README.md`).  
Explain the purpose of key files and folders. -->

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Stored images for github readme description.
    │   ├── interim        <- Images and caption which is been used for training.
    │   ├── processed      <- Saved tokens folder.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── experiences        <- Default folder to hold the experimentations with saved best models
    │
    │
    ├── notebooks          <- Jupyter notebooks used for miscellanous tasks. 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    └── src                <- Source code for use in this project.
        │
        ├── data           <- Scripts to create tokens and usable data for training and evalutation.
        |   |
        |   └── data_loader.py
        |   └── encode_sequence.py
        |   └── generate_token.py
        |   └── make_dataset.py
        │
        ├── configuration  <- Folder to store configuration for experimentations.
        |   |
        |   └── config.json
        │
        ├── evaluation     <- Scripts to evaluate the models.
        |   |
        |   └── evaluate.py
        │                    
        ├── training       <- Scripts to train the models.
        |   |
        |   └── train.py
        │                    
        ├── models_        <- Scripts to create the architecture of the models.
        |   |
        |   └── attention.py
        |   └── cnn.py
        |   └── lstm.py
        │   │                 
        └── utilities      <- Scripts to visualize and compute necessary stuff.
            |
            └── plotting.py
            └── utils.py



--------



## 4. Implementation Details
### Data Preparation
<!-- Describe the dataset(s) used.  
Explain data preprocessing steps (e.g., image augmentation, normalization). -->
For this project we have used the RSCID dataset which can be found and downloaded here: [LINK](https://www.kaggle.com/datasets/thedevastator/rsicd-image-caption-dataset?resource=d).
This dataset consist of 10921 images, each of the images is accompanied by 5 descriptions created by human annotators. Here is the distribution of the different object categories within the dataset.

<div style="text-align:center;">
    <figure>
    <img src="https://github.com/tiserge2/image-caption-generation/blob/main/data/external/data_distribution.png" 
        width="700"
        alt="Remote Sensing image of a residential area" />
    <!-- <figcaption style="font-size: 8px;">Encoder-Decoder architecture for caption generation task</figcaption> -->
    </figure>
</div>

Another analysis consisted in checking the word usage in the descriptions created by the annotator, in order to understand to probable prediction of the models. By checking this stats the top 5 words used are **a**, **are**, **green**, **many**, **trees**. The word **a** is remove for consistency of the model, because it will create some imbalance, as it has been used a lot, and not having it beeing predicted might not be that harmful to our model. We can see also that some very descriptive words aren't use very much, so we can yet suppose that they won't be predicted that much by the model.

<div style="text-align:center;">
    <figure>
    <img src="https://github.com/tiserge2/image-caption-generation/blob/main/data/external/top_word_usage.png" 
        width="1200"
        alt="Remote Sensing image of a residential area" />
    <!-- <figcaption style="font-size: 8px;">Encoder-Decoder architecture for caption generation task</figcaption> -->
    </figure>
</div>

For this project I have divided the dataset in two part. The first one is the test set which is made of 10% of the whole dataset. And the other part
is the train/validation set which would be divided in different ratio during the training experimentations phase.

Once the dataset has been downloaded it can be added under the path data/raw for the pre-processing stage, consisting of removing trailing space and punctuation, 
adding special token words like <start>, <pad> and <eos>. In order to create the usable dataset for training and test, we can the following steps:

1- Clean all the captions contained in the dataset. This will create a file called captions_all.json which will be later used to create the tokens file. 

```sh
$ cd src/data
$ python make_dataset.py build_test=False build_all=False
``` 

2- Create train and test dataset. This will create training dataset file called captions_train.json and testing dataset called captions_test.json.

```sh
$ cd src/data
$ python make_dataset.py build_test=True
$ python make_dataset.py build_test=False shortage=False
``` 

If you want to create a small dataset to train on, you can run this command, by specifying the number of examples you want in the set.

```sh
$ python make_dataset.py build_test=False shortage=True examples=1000
``` 

3 - The final step is to create the tokens file which will contain the 1-of-K encoding for the vocabulary which will be used for the embedding phase.

```sh
$ cd src/data
$ python generate_token.py
``` 

### Model Architecture
In this project, the implemented method is designed to utilize four different CNNs for feature extraction from remote sensing images. 
The CNN architectures used are AlexNet, ResNet18, ResNet101, VGG19, and GoogLeNet. 

For the Recurrent Neural Network (RNN) component, an LSTM (Long Short-Term Memory) network is used. 
Specifically, LSTM cells are chosen for their ability to handle long-range dependencies, 
which is crucial for generating coherent and contextually accurate captions. 
The LSTM cells facilitate the integration of an attention mechanism, 
which enhances the model's ability to focus on specific parts of the image while generating each word in the caption.


### Training Process

To train the model, first of all we need to specify the required parameters in the form of a json file in this path src/configuration.config.json. 
From there you can specify the optimizer to be used like SGD, AdamW etc. The pretrained model to extract fetures can be specified alongside with the 
learning rate and epochs for the experimentations.Here is the expected structure of that json file:

```js
{
    "architecture": "resnet18",
    "freeze_weights": false,
    "pretrained": true,
    "epochs": 20,
    "lr": 1e-05,
    "train_split": 0.8,
    "show_loss_plot": false,
    "show_example": true,
    "optimizer": "SGD"
    "patience": 10
}
``` 

To launch the training, this command can be used:

```sh
$ cd src/training
$ python train.py 
``` 

### Evaluation Metrics
<!-- Describe the metrics used to evaluate the model's performance (e.g., BLEU score, ROUGE score). -->
In order to evaluate the performance of the model I have used different metrics which are more specific to text generation, 
and different to the classical ones we often use. 

**BLEU-n**: Compare sequences of words called n-grams (groups of n words). A high BLEU score indicates that the generated caption contains word sequences similar to those of the reference captions, thus reflecting good text generation quality.

**ROUGE**: It compares the longest common subsequences between the generated caption and the reference captions.

**CIDEr**: This metric uses the TF-IDF (Term Frequency-Inverse Document Frequency) frequency to weight words and compare the generated captions to the reference captions.

**METEOR**: Takes into account flexible matches between words and synonyms, as well as precision and recall.

To launch the evaluation of a trained models, run the following command:

```sh
$ cd src/evaluation
$ python evaluate.py 
``` 

 ## 5. Results
### Model Performance
<!-- Present the results of the training and evaluation.  
Include visual examples of generated captions. -->
To train the model I have used the 90% of the dataset, which is divided into 80% for train and 20% for validation. Here is the train/validation loss curve of the best model
from the experimentations conducted so far, with an early stopping with patience of 10 and learning rate 2e-05:  

<div style="text-align:center;">
    <figure>
    <img src="https://github.com/tiserge2/image-caption-generation/blob/main/data/external/loss_train_val.png" 
        width="700"
        alt="Remote Sensing image of a residential area" />
    <figcaption style="font-size: 8px;">Train/Validation loss curve for 50 epochs</figcaption>
    </figure>
</div>

The model was able obtain a BLEU-1 score of 44.4% on the test set. Suggesting is was able to find 44% of the sequence with 1 length. 

<div style="text-align:center;">
    <figure>
    <img src="https://github.com/tiserge2/image-caption-generation/blob/main/data/external/result.png" 
        width="170"
        alt="Remote Sensing image of a residential area" />
    <figcaption style="font-size: 8px;">Performance of the model on the test dataset </figcaption>
    </figure>
</div>

Here is a some examples of the captions generated by the models. Textes with light green background refer to the ground truth descriptions. 
Blue background is caption describing perfectly the image, whereas red background are not well descripted at all.

<div style="text-align:center;">
    <figure>
    <img src="https://github.com/tiserge2/image-caption-generation/blob/main/data/external/some_result.png" 
        width="1200"
        alt="Remote Sensing image of a residential area" />
    <figcaption style="font-size: 8px;">Some captions generated by the model</figcaption>
    </figure>
</div>

### Analysis of Results
<!-- Interpret the results and discuss their significance.  
Compare with baseline models or other similar works if available. -->

The results obtained are far from the actual results of the paper, where they were able to reach
60% for BLEU-1 score for the lowest result in allt their experimentations. In my case I reached 44%
of BLEU-1, suggesting there are lot of thing which needs to be improved. One of this thing is pretty 
sure the attention computation which is very crucial. While analysis the computed attention maps, I realized
they almost looks alike accross different images and even accross predicted words. Meaning there might be a problem in the computation or 
it's not well trained or initialized. 

<div style="text-align:center;">
    <figure>
    <img src="https://github.com/tiserge2/image-caption-generation/blob/main/data/external/attention_vis.png" 
        width="1200"
        alt="Remote Sensing image of a residential area" />
    <figcaption style="font-size: 8px;">Some captions generated by the model</figcaption>
    </figure>
</div>

The issue of this result doesn't only come from the fact the attention cannot focus on the relevant part of the image. When we check on the top predicted words by the model it reflect perfectly the top word used in the descriptions of the dataset. So in some how the model has seen those words a lot, so it got too good at predicting them, hence almost predicting them all the time. 

<div style="text-align:center;">
    <figure>
    <img src="https://github.com/tiserge2/image-caption-generation/blob/main/data/external/top_word_predicted.png" 
        width="1200"
        alt="Remote Sensing image of a residential area" />
    <figcaption style="font-size: 8px;">Top words predicted by the model</figcaption>
    </figure>
</div>

## 6. Usage
### How to Use the Repository
To use this repository, you need to install the required dependencies which can be found in the requirements.txt file of the project. Once install you can refer to 
the section x for dataset creation and section xx for training and evaluation. 

```sh
$ conda create -n rscid python=3.10.13
$ conda activate rscid
$ pip install -r requirements.txt
``` 
### System Requirements

The project was tested on two different environemnet. The first is AWS Sagemaker on an instance of T4 1 GPUs of 16 GB. The second environement is 
Macbook Pro M1 with 8 GB of memory. The training process was 4 time slower on the Macbook Pro compared to the Sagemaker instance.

## 7. Challenges and Future Work
### Challenges Faced
The biggest challenge faced in the project was the understanding of the paper. There aren't much explanation of the architecture used in the paper. Well this is because I
think initially they aren't the one implementing the method, they are just reimplementing a natural image captioning methology for remote sensing images. But by looking in 
some other paper I had a clear idea of the methology and they even give more insight about the formulae there. 

### Future Improvements
This section outlines potential areas of improvement for our model.

- #### Hard Attention Mechanism

Implementing a hard attention mechanism could help the model focus more accurately on relevant parts of the input. 

- #### L2 Regularization

Adding L2 regularization to our model could help prevent overfitting. 

- #### Online Data Augmentation

Incorporating online data augmentation could help our model generalize better. 
This involves creating new training examples on the fly by applying transformations to the existing data, 
such as rotation, scaling, or flipping.

- #### Beam Search for Sentence Construction

Implementing beam search for sentence construction could improve the quality of the generated sentences. 
Instead of choosing the most likely next word at each step, beam search considers the most likely sequences of words, 
which can lead to more coherent and grammatically correct sentences.


### Final Thoughts
<!-- Offer final reflections on the project and its impact. -->
This project is very important because of it's application in different domain which can be useful. Based on the preliminary 
result I have got, and future improvement which can be done, I am pretty sure, it can achieve some outstanding audience and be
very useful.

## 8. How to contribute

1. Fork the project 🍴
2. Make a pull request 🛬
3. Wait for it to get reviewed and merged ✔️
4. And be happy to contribute there 🤗
5. Don't forget to 🌟 the project


*Feel free to open Issues whenever possible. This is the very basic version, and we are looking forward to make it run as smooth as possible on every platform.*

## 8. References
<!-- ### Citations -->
<!-- List all academic papers, articles, and other sources referenced in the report. -->

[1] Lu, Xiaoqiang, et al. "Exploring models and data for remote sensing image caption
generation." IEEE Transactions on Geoscience and Remote Sensing 56.4 (2017): 2183-2195.

[2] Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating
image descriptions. IEEE Trans. Pattern Anal. Mach. Intell., 39(4):664–676, 2017.

[3] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification
with deep convolutional neural networks. Commun. ACM, 60(6):84–90, 2017

[4] Girish Kulkarni, Visruth Premraj, Vicente Ordonez, Sagnik Dhar, Siming Li,
Yejin Choi, Alexander C Berg, and Tamara L Berg. Babytalk: Understanding and
generating simple image descriptions. IEEE transactions on pattern analysis and
machine intelligence, 35(12):2891–2903, 2013

[5] Bo Qu, Xuelong Li, Dacheng Tao, and Xiaoqiang Lu. Deep semantic under-
standing of high resolution remote sensing image. In International Conference on
Computer, Information and Telecommunication Systems, CITS 2016, Kunming,
China, July 6-8, 2016, pages 1–5. IEEE, 2016

[6] Binqiang Wang, Xiaoqiang Lu, Xiangtao Zheng, and Xuelong Li. Semantic
descriptions of high-resolution remote sensing images. IEEE Geosci. Remote.
Sens. Lett., 16(8):1274–1278, 2019

[7] Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron C. Courville, Ruslan
Salakhutdinov, Richard S. Zemel, and Yoshua Bengio. Show, attend and tell:
Neural image caption generation with visual attention. In Francis R. Bach and
David M. Blei, editors, Proceedings of the 32nd International Conference on
Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, volume 37 of JMLR
Workshop and Conference Proceedings, pages 2048–2057. JMLR.org, 2015

[linux-tested]: https://img.shields.io/badge/Linux-Tested%20on%20Linux-brightgreen?raw=true
[windows-tested]: https://img.shields.io/badge/Windows-Tested%20on%20Windows-yellowgreen?raw=true
[mac-tested]: https://img.shields.io/badge/Mac-Tested%20on%Mac-brightgreen?raw=true
[mac-inp]: https://img.shields.io/badge/Mac-Mac%20Fix%20In%20Progress-lightgrey=true