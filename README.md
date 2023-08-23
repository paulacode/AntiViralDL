# AntiViralDL

This is a tensorflow implementation with a computational framework for predicting virus-drug associations using self-supervised learning as described in our paper：

“AntiViralDL: Computational antiviral Drug Repurposing Using Graph Neural Network and Self Supervised Learning”



## Requirements
* TensorFlow 1.14/2.4
* python 3.7
* numpy 1.19
* pandas 1.1
* scikit-learn 1.0
* scipy 1.5

## Run the demo

```bash
python main.py
```

## Data
Known virus-drug associations from two different sources: the DrugVirus.info 2 database and the US FDA database.

The DrugVirus.info 2 database contains 1,519 virus-drug associations between 231 drugs and 153 viruses and we manually collect current FDA-approved drugs for the treatment of viral infectious diseases,including  142 virus-drug associations between 111 drugs and 16 viruses.

Overall, we obtained a merged dataset comprising 1,648 virus-drug associations between 158 viruses and 336 drugs after removing all redundant records.


## Cite

Please cite our paper if you use this code in your own work:

“AntiViralDL: Computational antiviral Drug Repurposing Using Graph Neural Network and Self Supervised Learning”
