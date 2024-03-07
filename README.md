#  Tackling Long Code Search with Splitting, Encoding, and Aggregating

[LREC-COLING 2024] This is the official PyTorch implementation for the paper: ["Tackling Long Code Search with Splitting, Encoding, and Aggregating"](https://arxiv.org/abs/2208.11271).



## Environment

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install all the required packages.

```sh
conda create -n SEA python==3.8
conda activate SEA

download this_project

cd SEA
pip install -r requirements.txt
pip install git+https://github.com/casics/spiral.git
```



## Data prepare

### Dataset

We follow [GraphCodeBERT](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch) pre-process progress for CodeSearchNet. The answer of each query is retrieved from the whole development and testing code corpus instead of 1,000 candidate codes. We put the dataset files on `~/VisualSearch/GraphCodeBERT_dataset`.

```sh
mkdir ~/VisualSearch
mkdir ~/VisualSearch/GraphCodeBERT_dataset
```

Please refer to GraphCodeBERT for data-preprocess: https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch#data-preprocess



## Training and evaluation on six language of CodeSearchNet (Supplementary for Table 7 of the paper)

### SEA-ASTSplitting with a window size of 32 and Attention (one layer) + Mean fusion method.



```sh
langs=(ruby python javascript php go java)
encoder_name_or_paths=("microsoft/graphcodebert-base" "microsoft/unixcoder-base" "roberta-base" "microsoft/codebert-base")
train_batch_size=48
eval_batch_size=64
window_size=32
step=16
nl_length=128
code_length=256
num_train_epochs=15
split_type="ast_subtree"
AttentionCommonType="one_layer"


for lang in ${langs[*]}
do
for encoder_name_or_path in ${encoder_name_or_paths[*]}
do
python train_LongCode.py  \
--num_workers 40 --train_batch_size ${train_batch_size} --TrainModel GraphCodeBERTMultiCodeFusion \
--device cuda --learning_rate 2e-5 --lang ${lang} --window_setting WindowSize_${window_size},step_${step} \
--seed 5 --encoder_name_or_path ${encoder_name_or_path} --nl_length ${nl_length} \
--code_length ${code_length} --eval_batch_size ${eval_batch_size} --num_train_epochs ${num_train_epochs} \
  --split_type ${split_type} --AttentionWithAve --AttentionCommonType ${AttentionCommonType} 
done
done
```

On two V100 GPU, we get following results:



**MRR**:

| Model / Method       | Ruby                     | Javascript               | Go                      | Python                  | Java                     | Php                      | Overall                  |
| -------------------- | ------------------------ | ------------------------ | ----------------------- | ----------------------- | ------------------------ | ------------------------ | ------------------------ |
| RoBERTa              | 0.587                    | 0.517                    | 0.850                   | 0.587                   | 0.599                    | 0.560                    | 0.617                    |
| UniXcoder            | 0.586                    | 0.603                    | 0.881                   | 0.695                   | 0.687                    | 0.644                    | 0.683                    |
| CodeBERT             | 0.679                    | 0.620                    | 0.882                   | 0.672                   | 0.676                    | 0.628                    | 0.693                    |
| GraphCodeBERT        | 0.703                    | 0.644                    | 0.897                   | 0.692                   | 0.691                    | 0.649                    | 0.713                    |
| SEA+RoBERTa       | 0.651 (10.9%$\uparrow$) | 0.593 (14.6%$\uparrow$) | 0.879 (3.5%$\uparrow$) | 0.633 (7.9%$\uparrow$) | 0.666 (11.1%$\uparrow$) | 0.647 (15.6%$\uparrow$) | 0.678 (10.0%$\uparrow$) |
| SEA+UniXcoder     | 0.648 (10.7%$\uparrow$) | 0.692 (14.8%$\uparrow$) | 0.896 (1.8%$\uparrow$) | 0.707 (1.7%$\uparrow$) | 0.739 (7.5%$\uparrow$)  | 0.712 (10.5%$\uparrow$) | 0.732 (7.3%$\uparrow$)  |
| SEA+CodeBERT      | 0.742 (9.3%$\uparrow$)  | 0.696 (12.3%$\uparrow$) | 0.905 (2.6%$\uparrow$) | 0.714 (6.2%$\uparrow$) | 0.732 (8.3%$\uparrow$)  | 0.711 (13.2%$\uparrow$) | 0.750 (8.3%$\uparrow$)  |
| SEA+GraphCodeBERT | 0.776 (10.4%$\uparrow$) | 0.742 (15.2%$\uparrow$) | 0.921 (2.7%$\uparrow$) | 0.754 (8.9%$\uparrow$) | 0.768 (11.1%$\uparrow$) | 0.748 (15.3%$\uparrow$) | 0.785 (10.1%$\uparrow$) |



**R@1**

| Model / Method    | Ruby | Javascript | Go   | Python | Java | Php  | Overall |
| ----------------- | ---- | ---------- | ---- | ------ | ---- | ---- | ------- |
| RoBERTa (code)    | 52.4 | 45.2       | 81.1 | 51.1   | 52.8 | 46.7 | 54.9    |
| UniXcoder         | 47.4 | 49.7       | 82.5 | 59.1   | 59.0 | 54.0 | 58.6    |
| CodeBERT          | 58.3 | 51.4       | 83.7 | 57.4   | 58.0 | 52.1 | 60.2    |
| GraphCodeBERT     | 60.7 | 53.8       | 85.8 | 59.4   | 59.2 | 54.5 | 62.2    |
| SEA+RoBERTa       | 54.3 | 49.0       | 82.4 | 52.3   | 56.4 | 53.7 | 58.0    |
| SEA+UniXcoder     | 54.0 | 59.3       | 84.4 | 60.1   | 64.4 | 60.7 | 63.8    |
| SEA+CodeBERT      | 64.7 | 59.8       | 85.7 | 61.1   | 63.8 | 60.7 | 66.0    |
| SEA+GraphCodeBERT | 68.8 | 64.7       | 87.9 | 65.8   | 68.1 | 64.9 | 70.0    |



**R@5**

| Model / Method    | Ruby | Javascript | Go   | Python | Java | Php  | Overall |
| ----------------- | ---- | ---------- | ---- | ------ | ---- | ---- | ------- |
| RoBERTa (code)    | 76.1 | 71.6       | 93.0 | 75.6   | 77.0 | 71.5 | 77.5    |
| UniXcoder         | 71.8 | 73.4       | 94.8 | 82.2   | 80.5 | 77.4 | 80.0    |
| CodeBERT          | 80.0 | 75.2       | 94.4 | 79.2   | 79.6 | 75.3 | 80.6    |
| GraphCodeBERT     | 82.4 | 77.4       | 95.4 | 81.3   | 81.7 | 83.2 | 83.6    |
| SEA+RoBERTa       | 78.7 | 71.3       | 94.6 | 76.5   | 78.9 | 78.1 | 79.7    |
| SEA+UniXcoder     | 77.8 | 81.1       | 96.1 | 83.6   | 85.7 | 84.1 | 84.7    |
| SEA+CodeBERT      | 86.0 | 82.0       | 96.4 | 84.1   | 85.0 | 84.0 | 86.2    |
| SEA+GraphCodeBERT | 89.1 | 86.1       | 97.3 | 87.2   | 87.6 | 86.9 | 89.0    |



**R@10**

| Model / Method    | Ruby | Javascript | Go   | Python | Java | Php  | Overall |
| ----------------- | ---- | ---------- | ---- | ------ | ---- | ---- | ------- |
| RoBERTa (code)    | 82.1 | 79.4       | 95.2 | 81.9   | 83.1 | 78.3 | 83.3    |
| UniXcoder         | 78.4 | 79.9       | 96.8 | 87.2   | 85.9 | 83.6 | 85.3    |
| CodeBERT          | 85.3 | 81.4       | 96.2 | 85.0   | 85.2 | 81.4 | 85.8    |
| GraphCodeBERT     | 87.2 | 83.4       | 97.2 | 86.6   | 86.5 | 78.5 | 86.6    |
| SEA+RoBERTa       | 84.5 | 77.7       | 96.4 | 82.6   | 83.8 | 84.4 | 84.9    |
| SEA+UniXcoder     | 84.1 | 86.8       | 97.6 | 88.8   | 89.8 | 89.6 | 89.5    |
| SEA+CodeBERT      | 89.8 | 87.0       | 97.7 | 89.4   | 89.1 | 89.6 | 90.4    |
| SEA+GraphCodeBERT | 92.0 | 90.2       | 98.4 | 91.7   | 91.1 | 91.6 | 92.5    |



### SEA-TokenSplitting with a window size of 256 and Attention (one layer) + Mean fusion method.

As shown in Table 5 of the paper, the search performance of SEA-TokenSplitting with a window size of 256 and Attention (one layer) + Mean fusion method is a little lower than the optimal configuration. We also provide the codes and results of this method.

```shell
langs=(ruby python javascript php go java)
encoder_name_or_paths=("microsoft/graphcodebert-base" "microsoft/unixcoder-base" "roberta-base" "microsoft/codebert-base")
train_batch_size=48
eval_batch_size=64
window_size=32
step=16
nl_length=128
code_length=256
num_train_epochs=15
split_type="token"
AttentionCommonType="one_layer"



for lang in ${langs[*]}
do
for encoder_name_or_path in ${encoder_name_or_paths[*]}
do
python train_LongCode.py  \
--num_workers 40 --train_batch_size ${train_batch_size} --TrainModel GraphCodeBERTMultiCodeFusion \
--device cuda --learning_rate 2e-5 --lang ${lang} --window_setting WindowSize_${window_size},step_${step} \
--seed 5 --encoder_name_or_path ${encoder_name_or_path} --nl_length ${nl_length} \
--code_length ${code_length} --eval_batch_size ${eval_batch_size} --num_train_epochs ${num_train_epochs} \
  --split_type ${split_type} --AttentionWithAve --AttentionCommonType ${AttentionCommonType} 
done
done
```

On two V100 GPU, we get following results:



**MRR**:

| Model / Method       | Ruby                     | Javascript               | Go                      | Python                  | Java                    | Php                      | Overall                 |
| -------------------- | ------------------------ | ------------------------ | ----------------------- | ----------------------- | ----------------------- | ------------------------ | ----------------------- |
| RoBERTa              | 0.587                    | 0.517                    | 0.850                   | 0.587                   | 0.599                   | 0.560                    | 0.617                   |
| UniXcoder            | 0.586                    | 0.603                    | 0.881                   | 0.695                   | 0.687                   | 0.644                    | 0.683                   |
| CodeBERT             | 0.679                    | 0.620                    | 0.882                   | 0.672                   | 0.676                   | 0.628                    | 0.693                   |
| GraphCodeBERT        | 0.703                    | 0.644                    | 0.897                   | 0.692                   | 0.691                   | 0.649                    | 0.713                   |
| SEA+RoBERTa       | 0.637 (8.4%$\uparrow$)  | 0.578 (11.8%$\uparrow$) | 0.872 (2.6%$\uparrow$) | 0.625 (6.5%$\uparrow$) | 0.651 (8.8%$\uparrow$) | 0.640 (14.3%$\uparrow$) | 0.667 (8.2%$\uparrow$) |
| SEA+UniXcoder     | 0.652 (11.3%$\uparrow$) | 0.644 (6.9%$\uparrow$)  | 0.886 (0.6%$\uparrow$) | 0.737 (6.1%$\uparrow$) | 0.750 (9.2%$\uparrow$) | 0.729 (13.1%$\uparrow$) | 0.733 (7.4%$\uparrow$) |
| SEA+CodeBERT      | 0.733 (7.9%$\uparrow$)  | 0.652 (5.2%$\uparrow$)  | 0.908 (2.9%$\uparrow$) | 0.707 (5.1%$\uparrow$) | 0.740 (9.5%$\uparrow$) | 0.707 (12.6%$\uparrow$) | 0.739 (6.6%$\uparrow$) |
| SEA+GraphCodeBERT | 0.775 (10.3%$\uparrow$) | 0.660 (2.6%$\uparrow$)  | 0.909 (1.3%$\uparrow$) | 0.723 (4.4%$\uparrow$) | 0.743 (7.5%$\uparrow$) | 0.728 (12.1%$\uparrow$) | 0.756 (6.1%$\uparrow$) |



**R@1**

| Model / Method       | Ruby | Javascript | Go   | Python | Java | Php  | Overall |
| -------------------- | ---- | ---------- | ---- | ------ | ---- | ---- | ------- |
| RoBERTa (code)       | 52.4 | 45.2       | 81.1 | 51.1   | 52.8 | 46.7 | 54.9    |
| UniXcoder            | 47.4 | 49.7       | 82.5 | 59.1   | 59.0 | 54.0 | 58.6    |
| CodeBERT             | 58.3 | 51.4       | 83.7 | 57.4   | 58.0 | 52.1 | 60.1    |
| GraphCodeBERT        | 60.7 | 53.8       | 85.8 | 59.4   | 59.2 | 54.5 | 62.2    |
| SEA+RoBERTa       | 52.8 | 39.4       | 81.3 | 51.5   | 54.6 | 52.6 | 55.4    |
| SEA+UniXcoder     | 54.2 | 42.9       | 82.0 | 63.6   | 65.8 | 62.9 | 61.9    |
| SEA+CodeBERT      | 63.7 | 44.0       | 86.0 | 60.3   | 64.7 | 60.2 | 63.1    |
| SEA+GraphCodeBERT | 68.4 | 43.2       | 86.0 | 62.0   | 64.8 | 62.8 | 64.5    |



**R@5**

| Model / Method       | Ruby | Javascript | Go   | Python | Java | Php  | Overall |
| -------------------- | ---- | ---------- | ---- | ------ | ---- | ---- | ------- |
| RoBERTa (code)       | 76.1 | 71.6       | 93.0 | 75.6   | 77.0 | 71.5 | 77.5    |
| UniXcoder            | 71.8 | 73.4       | 94.8 | 82.2   | 80.5 | 77.4 | 80.0    |
| CodeBERT             | 80.0 | 75.2       | 94.4 | 79.2   | 79.6 | 75.3 | 80.6    |
| GraphCodeBERT        | 82.4 | 77.4       | 95.4 | 81.3   | 81.7 | 83.2 | 83.6    |
| SEA+RoBERTa       | 76.7 | 76.3       | 94.5 | 75.9   | 78.2 | 77.8 | 79.9    |
| SEA+UniXcoder     | 78.7 | 86.0       | 96.5 | 86.2   | 86.6 | 85.2 | 86.5    |
| SEA+CodeBERT      | 84.6 | 87.1       | 96.8 | 83.4   | 85.3 | 83.6 | 86.8    |
| SEA+GraphCodeBERT | 89.1 | 89.2       | 96.8 | 85.0   | 86.1 | 85.3 | 88.6    |



**R@10**

| Model / Method       | Ruby | Javascript | Go   | Python | Java | Php  | Overall |
| -------------------- | ---- | ---------- | ---- | ------ | ---- | ---- | ------- |
| RoBERTa (code)       | 82.1 | 79.4       | 95.2 | 81.9   | 83.1 | 78.3 | 83.3    |
| UniXcoder            | 78.4 | 79.9       | 96.8 | 87.2   | 85.9 | 83.6 | 85.3    |
| CodeBERT             | 85.3 | 81.4       | 96.2 | 85.0   | 85.2 | 81.4 | 85.8    |
| GraphCodeBERT        | 87.2 | 83.4       | 97.2 | 86.6   | 86.5 | 78.5 | 86.6    |
| SEA+RoBERTa       | 83.1 | 84.1       | 96.3 | 82.5   | 83.7 | 84.3 | 85.7    |
| SEA+UniXcoder     | 83.9 | 92.4       | 98.4 | 90.9   | 90.4 | 90.5 | 91.1    |
| SEA+CodeBERT      | 88.9 | 92.4       | 98.1 | 88.7   | 89.8 | 89.1 | 91.2    |
| SEA+GraphCodeBERT | 91.9 | 95.1       | 98.0 | 90.0   | 90.0 | 90.2 | 92.5    |



## Cite

If you find SEA useful for your research or development, please cite the following papers: [SEA](https://arxiv.org/abs/2208.11271).

```
@inproceedings{tian2023eulernet,
  title = {Tackling Long Code Search with Splitting, Encoding, and Aggregating},
  author = {Fan Hu, Yanlin Wang, Lun Du, Hongyu Zhang, Shi Han, Dongmei Zhang, Xirong Li},
  booktitle = {LREC-COLING},
  year = {2024},
}
```
