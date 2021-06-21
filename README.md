# Few-Shot-Intent-Detection


Few-Shot-Intent-Detection is a repository designed for few-shot intent detection with/without Out-of-Scope (OOS) intents. It includes popular challenging intent detection datasets and baselines. For more details of the new released OOS datasets, please check our [paper](https://arxiv.org/abs/2106.04564).


**---Note: We will update the repository soon---**

## Intent detection datasets

We process data based on previous published resources, all the data are in the same format as [DNNC](https://github.com/salesforce/DNNC-few-shot-intent). 


| Dataset      	| Description  | #Train | #Valid | #Test 	|  Processed Data Link| 
|--------------	|------	|------	|------	|---------------	|------	|
| [BANKING77](https://arxiv.org/abs/2003.04807)      	| one banking domain with 77 intents  |8622|1540| 3080  	|  [Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/BANKING77)                  	|
| [CLINC150](https://www.aclweb.org/anthology/D19-1131/)        | 10 domains and 150 intents |15000| 3000	| 4500 	| [Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/CLINC150)|                                              	| Link	|
| [HWU64](https://arxiv.org/abs/1903.05566)        | personal assistant with 64 intents and several domains                                                 |8954| 1076	| 1076 	|  [Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/HWU64)	|
| [SNIPS](https://arxiv.org/pdf/1805.10190.pdf)        |snips voice platform with 7 intents   |13084| 700	| 700 	|  [Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/SNIPS)	|
| [ATIS](https://ieeexplore.ieee.org/document/5700816)        |airline travel information system   |4478| 500	| 893 	|  [Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/SNIPS)	|



## Intent detection datasets with OOS queries


What is OOS queires:

`OOD-OOS`: i.e., out-of-domain OOS. General out-of-scope queries which are not supported by the dialog systems, also called out-of-domain OOS. For instance, requesting an online NBA/TV show service in a banking system.

`ID-OOS`: i.e., in-domain OOS. Out-of-scope queries which are more related to the in-scope intents, which makes the intent detection task more challenging. For instance, requesting a banking service that is not supported by the banking system.

| Dataset      	| Description  | #Train | #Valid | #Test 	|#OOD-OOS-Train |#OOD-OOS-Valid|#OOD-OOS-Test| #ID-OOS-Train |#ID-OOS-Valid|#ID-OOS-Test| Processed Data Link| 
|--------------	|------	|------	|------	|---------------	|------	|------	|------	|------	|------	|------|------	|
| [CLINC150](https://www.aclweb.org/anthology/D19-1131/)        | A dataset with general OOS-OOS queries |15000| 3000	| 4500  |	100| 100|1000| -|-|-|[Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/CLINC150)|
| [CLINC-Single-Domain-OOS](https://arxiv.org/abs/2106.04564)        | Two domains with both general OOS-OOS queries and ID-OOS queries |500| 500	| 500  |-	| 200|1000| -|400|350|[Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/CLINC-Single-Domain-OOS)|                                             
| [BANKING77-OOS](https://arxiv.org/abs/2106.04564)        | One banking domain with both general OOS-OOS queries and ID-OOS queries |5905| 1506	| 2000  |-	| 200|1000| 2062|530|1080|[Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/BANKING77-OOS)|      


Briefly describe the [BANKING77-OOS](https://arxiv.org/abs/2106.04564) dataset. 

*  A dataset with a single banking domain, includes both general Out-of-Scope (OOD-OOS) queries and In-Domain but Out-of-Scope (ID-OOS) queries, where ID-OOS queries are semantically similar intents/queries with in-scope intents.  BANKING77 originally includes 77 intents. BANKING77-OOS includes 50 in-scope intents in this dataset, and the ID-OOS queries are built up based on 27 held-out semantically similar in-scope intents.

Briefly describe the [CLINC-Single-Domain-OOS](https://arxiv.org/abs/2106.04564) dataset. 

*  A dataset with two separate domains, i.e., the  "Banking''  domain and the "Credit cards''  domain with both general Out-of-Scope (OOD-OOS) queries and In-Domain but Out-of-Scope (ID-OOS) queries, where ID-OOS queries are semantically similar intents/queries with in-scope intents. Each domain in CLINC150 originally includes 15 intents. Each domain in the new dataset includes ten in-scope intents in this dataset, and the ID-OOS queries are built up based on five held-out semantically similar in-scope intents.

Both datasets can be used to conduct intent detection with and without OOD-OOS and ID-OOS queries


You can easily load the processed data:
```python
class IntentExample:
    def __init__(self, text, label, do_lower_case):
        self.original_text = text
        self.text = text
        self.label = label

        if do_lower_case:
            self.text = self.text.lower()
        
def load_intent_examples(file_path, do_lower_case=True):
    examples = []

    with open('{}/seq.in'.format(file_path), 'r', encoding="utf-8") as f_text, open('{}/label'.format(file_path), 'r', encoding="utf-8") as f_label:
        for text, label in zip(f_text, f_label):
            e = IntentExample(text.strip(), label.strip(), do_lower_case)
            examples.append(e)

    return examples
```

More details can check [code for load data and do random sampling for few-shot learning](https://github.com/salesforce/DNNC-few-shot-intent/blob/master/train_classifier.py#L127).

## State-of-the art models and baselines


**[DNNC](https://www.aclweb.org/anthology/2020.emnlp-main.411/)**

Download pre-trained RoBERTa NLI checkpoint: 
```bash
wget https://storage.googleapis.com/sfr-dnnc-few-shot-intent/roberta_nli.zip
```
Access to public code: [Link](https://github.com/salesforce/DNNC-few-shot-intent)

**[CONVERT](https://www.aclweb.org/anthology/2020.nlp4convai-1.5/)**

Download pre-trained checkpoint: 
```bash
wget https://github.com/connorbrinton/polyai-models/releases/download/v1.0/model.tar.gz
```

Access to public code:
```bash
wget https://github.com/connorbrinton/polyai-models/archive/refs/tags/v1.0.zip
```


**[CONVBERT](https://arxiv.org/abs/2009.13570)** 

Download pre-trained checkpoints: 

Step-1: install [AWS CL2](https://aws.amazon.com/cli/): e.g., install [MacOS PKG](https://awscli.amazonaws.com/AWSCLIV2.pkg)

Step-2: 
```bash
aws s3 cp s3://dialoglue/ --no-sign-request `Your_folder_name` --recursive
```
Then the checkpoints are downloaded into  `Your_folder_name`

## Few-shot intent detection results:

**5-shot learning**

Coming soon

**10-shot learning**

| Model      	| BANKING77  | CLICN150 | HWU64 | 
|--------------	|------	|------	|------	|
|[RoBERTa+Classifier](https://www.aclweb.org/anthology/2020.emnlp-main.411/) | 84.27 | 91.55 | 82.89 |
|[USE+CONVERT](https://www.aclweb.org/anthology/2020.nlp4convai-1.5/)        | 85.19 | 93.26 | 85.83 | 
|[CONVBERT+MLM](https://arxiv.org/abs/2009.13570)       | 83.99 | 92.75 | 84.52 |
|[DNNC](https://www.aclweb.org/anthology/2020.emnlp-main.411/)               | 86.71 | 93.76 | 84.72 | 

`Note:` All the results are reported by the paper authors.

## Citation

Please cite our paper if you use above resources in your work:

```bibtex
@article{zhang2021pretrained,
  title={Are Pretrained Transformers Robust in Intent Classification? A Missing Ingredient in Evaluation of Out-of-Scope Intent Detection},
  author={Zhang, Jian-Guo and Hashimoto, Kazuma and Wan, Yao and Liu, Ye and Xiong, Caiming and Yu, Philip S},
  journal={arXiv preprint arXiv:2106.04564},
  year={2021}
}
```
