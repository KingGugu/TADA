## TADA

Official source code for WWW 2026 paper: [Tail-Aware Data Augmentation for Long-Tail Sequential Recommendation]()


## Directory Structure
TADA/
|--- data/ # preprocessed dataset files
|--- src/
     |--- output/ 
     |--- weight/ # weights of the pre-trained backbone model
     |--- datasets.py # tail-aware operators augmentation
     |--- ht_process.py # head/tail partition and co-occurrence set construction
     |--- LIS.py # item-item relevance set construction
     |--- main.py
     |--- models.py
     |--- modules.py
     |--- trainers.py
     |--- utils.py

## Run the Code

To save time, we provide the pretrained weights for the original models in `src/weight`, which is the first stage in the paper. 

The pretrained weights can be loaded by running the commands and moving to the second stage to further improve the model performance using our method.

```
python main.py --aug=1 --gpu_id=0 --model_idx=5 --data_name=Toys_and_Games
python main.py --aug=1 --gpu_id=0 --model_idx=5 --data_name=Beauty
python main.py --aug=1 --gpu_id=0 --model_idx=5 --data_name=Sports_and_Outdoors --rate_a=0.61 --rate_b=0.81 --th=0.5
```

You can also use the following instructions to train original SASRec (without data augmentation)

```
python main.py --aug=0 --gpu_id=0 --model_idx=10 --data_name=[DATA NAME] --attention_probs_dropout_prob=0.5 --hidden_dropout_prob=0.5 --star_test=200
```

You can run other backbones using the following commands:
```
python main.py --aug=0 --gpu_id=0 --model_idx=10 --data_name=[DATA NAME] --model_name=[BACKBONE NAME]
```
For detailed hyperparameter settings, please refer to the log.

## Log Files

We also provide log files and trained weights on these three datasets in the `src/output` directory.

## Reference

Please cite our paper if you use this code.
```

```