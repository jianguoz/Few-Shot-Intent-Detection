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
<!-- | [CLINC150](https://www.aclweb.org/anthology/D19-1131/)        	| 20K  	| popular personal assistant queries                                                  	| CC-BY-SA 3.0               	|
| [Restaurant8k](https://arxiv.org/abs/2005.08866) 	| 8.2K 	| restaurant booking domain queries                                                   	| CC-BY-4.0                  	|
| [DSTC8 SGD](https://arxiv.org/abs/1909.05855)    	| 20K  	| multi-domain, task-oriented conversations   between a human and a virtual assistant 	| CC-BY-SA 4.0 International 	|
| [TOP](https://arxiv.org/abs/1810.07942)          	| 45K  	| compositional queries for hierachical   semantic representations                    	| CC-BY-SA                   	|
| [MultiWOZ 2.1](https://arxiv.org/abs/1907.01669) 	| 12K  	| multi-domain dialogues with multiple turns                                              	| MIT                        	|

 -->

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

## Intent detection datasets with OOS queries


## State-of-the art models and baselines

  
