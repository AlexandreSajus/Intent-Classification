# Intent Classification
A benchmark of different approaches on the task of Intent Classification on the Hugging Face Silicone Maptask dataset.

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

## Human Level Performance

**The human accuracy is at 54.1% (13/24 examples correct)**

Measuring this metric was done by guessing 24 labels for 24 utterances by only using for reference 3 labelled examples per label.

The logs of the expermiment are available at `human_level_perf.txt`

This sets a baseline performance and highlights challenges when predicting these labels: some labels are very similar
- "right" can be ready but can also be acknowledge or reply_y which highlights intersection between labels
- clarify ("right down there") and check ("right beside it") can be confused without punctuation
