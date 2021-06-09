# Few-Shot-Intent-Detection


Few-Shot-Intent-Detection is a repository designed for few-shot intent detection with/without Out-of-Scope (OOS) intents. It includes popular challenging intent detection datasets and baselines.


**---Note: We will update the repository soon---**

## Intent detection datasets

We process data based on previous published resources, all the data are in the same format as [DNNC](https://github.com/jianguoz/DNNC-few-shot-intent). 


| Dataset      	| Description  | #Train | #Valid | #Test 	|  Processed Data Link| 
|--------------	|------	|------	|------	|---------------	|------	|
| [BANKING77](https://arxiv.org/abs/2003.04807)      	| one banking domain with 77 intents  |8622|1540| 3080  	|  [Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/BANKING77)                  	|
| [CLINC150](https://www.aclweb.org/anthology/D19-1131/)        | 10 domains and 150 intents |15000| 3000	| 4500 	| [Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/CLINC150)|                                              	| Link	|
| [HWU64](https://arxiv.org/abs/1903.05566)        | personal assistant with 64 intents and several domains                                                 |8954| 1076	| 1076 	|  [Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/HWU64)	|
| [SNIPS](https://arxiv.org/pdf/1805.10190.pdf)        |snips voice platform with 7 intents   |13084| 700	| 700 	|  [Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/SNIPS)	|
| [ATIS](https://ieeexplore.ieee.org/document/5700816)        |airline travel information system   |4478| 500	| 893 	|  [Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/SNIPS)	|



## Intent detection datasets with OOS queries

| Dataset      	| Description  | #Train | #Valid | #Test 	|#OOD-OOS-Train |#OOD-OOS-Valid|#OOD-OOS-Test| #ID-OOS-Train |#ID-OOS-Valid|#ID-OOS-Test| Processed Data Link| 
|--------------	|------	|------	|------	|---------------	|------	|------	|------	|------	|------	|------|------	|
| [CLINC150](https://www.aclweb.org/anthology/D19-1131/)        | A dataset with general OOS-OOS queries |15000| 3000	| 4500  |	100| 100|1000| -|-|-|[Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/CLINC150)|
| [CLINC-Single-Domain-OOS](https://arxiv.org/abs/2106.04564)        | Two domains with both general OOS-OOS queries and ID-OOS queries |500| 500	| 500  |-	| 200|1000| -|400|350|[Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/CLINC-Single-Domain-OOS)|                                             
| [BANKING77-OOS](https://arxiv.org/abs/2106.04564)        | One banking domain with both general OOS-OOS queries and ID-OOS queries |5905| 1506	| 2000  |-	| 200|1000| 2062|530|1080|[Link](https://github.com/jianguoz/Few-Shot-Intent-Detection/tree/main/Datasets/BANKING77-OOS)|      


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

More details can check [code for load data and do random sampling for few-shot learning](https://github.com/jianguoz/DNNC-few-shot-intent/blob/master/train_classifier.py#L127).

## State-of-the art models and baselines

  
