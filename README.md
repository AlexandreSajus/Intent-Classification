# Intent Classification

A benchmark of different approaches on the task of Intent Classification on the Hugging Face Silicone Maptask dataset:

| **Approach**          | **Model**        | **Performance (Test Accuracy)** |
| --------------------- | ---------------- | ------------------------------- |
| Human                 | Manual Labelling | 54.1%                           |
| Training from Scratch | LSTM             | 61.0%                           |
| Fine-Tuning           | BERT             | *Not Yet Implemented*           |
| Prompting             | GPT-4            | *Not Yet Implemented*           |

## Dataset

The dataset has 12 labels:

| **Label**   | **Example**                                            |
| ----------- | ------------------------------------------------------ |
| acknowledge | uh-huh                                                 |
| align       | okay                                                   |
| check       | on the right-hand side roughly just                    |
| clarify     | right beside it                                        |
| explain     | i've got a gallows to the left like d-- below the left |
| instruct    | okay the start part is at the top left-hand corner     |
| query_w     | how far underneath the diamond mine                    |
| query_yn    | do you have a diamond mine there                       |
| ready       | well                                                   |
| reply_n     | no                                                     |
| reply_w     | no i haven't got that                                  |
| reply_y     | uh-huh                                                 |

The label distribution is not too imbalanced which means that accuracy is still a useful metric to use. We will also use confusion matrixes to make sure that some labels are not forgotten by our models.

<p align="center">
  <img src="media/label_distrib.png" alt="Label Distribution" width="50%"/>
</p>

## Human Level Performance

**The human accuracy is at 54.1% (13/24 examples correct)**

Measuring this metric was done by guessing 24 labels for 24 utterances by only using for reference 3 labelled examples per label.

The logs of the expermiment are available at `human_level_perf.txt`

This sets a baseline performance and highlights challenges when predicting these labels: some labels are very similar
- "right" can be ready but can also be acknowledge or reply_y which highlights intersection between labels
- clarify ("right down there") and check ("right beside it") can be confused without punctuation

## LSTM

In this part we train a simple LSTM model on the task:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 128)         3906816   
_________________________________________________________________
bidirectional (Bidirectional (None, None, 512)         788480    
_________________________________________________________________
dropout (Dropout)            (None, None, 512)         0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 512)               1574912   
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               131328    
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 12)                3084      
=================================================================
Total params: 6,404,620
Trainable params: 6,404,620
Non-trainable params: 0
_________________________________________________________________
```

Training for 20 epochs results in the following accuracy plots:

<p align="center">
  <img src="media/LSTM_train.png" alt="LSTM Train" width="50%"/>
</p>

We see that even with Dropout, the model suffers from overfitting.

**The best model results in a 61.0% test accuracy**

Which beats the human baseline and has the following confusion matrix:

<p align="center">
  <img src="media/LSTM_confusion.png" alt="LSTM Confusion" width="50%"/>
</p>

