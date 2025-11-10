## KeenKT

Welcome to the paper **"KeenKT:Knowledge Mastery-State Disambiguation for Knowledge Tracing"** .

## Experiment Environment
- python 3.10+
- torch 2.0+
- torch_geometric 2.4+
- scikit-learn 1.4+
- pandas 2.2.0+
- tqdm

## Dataset
we use datasets including :

Assist2009(https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010)

Algebra2005 (https://pslcdatashop.web.cmu.edu/KDDCup/)

Bridge2006 (https://pslcdatashop.web.cmu.edu/KDDCup/)

NIPS34 (https://eedi.com/projects/neurips-education-challenge)

ASSISTments2015 (https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data)

POJ(https://drive.google.com/drive/folders/1LRljqWfODwTYRMPw6wEJ_mMt1KZ4xBDk)

## Data Preparation

```
cd train
python data_preprocess.py --dataset_name=assist2015
```

## Run Your Model

You could run KeenKT as follows command 

```
CUDA_VISIBLE_DEVICES=0 python python KeenKT-train.py --dataset_name=assist2015
```

## Evaluate Your Model

Now, let’s use `predict.py` to evaluate the model performance on the testing set.

```
python predict.py --save_dir=saved_model/YourModelPath
```

--save_dir is the save path of your trained model that you can find in your training log


