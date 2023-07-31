# FEDformer


The method, termed as Frequency Enhanced Decomposed Transformer (FEDformer), is more efficient than standard Transformer with a linear

complexity to the sequence length. 


## All you need is here!!


#### Kaggle or run locally option:


- If you want to run on kaggle, you can find a ready notebook for this called alldata+custom_using_fedformer_(kaggle_run).ipynb

- If you have your custom server or you need to run locally, you have a notebook is ready for that called alldata+custom_using_fedformer_(Queen's_Server_run).ipynb.


#### Generating output in files or in cell option:


- If you want to generate output in cell , you can find a ready notebook for this called alldata+custom_using_fedformer_(kaggle_run).ipynb.

- If you want to generate output in files, you have a notebook is ready for that called alldata+custom_using_fedformer_(Queen's_Server_run).ipynb.



## Installation


Use the package manager [pip](https://pip.pypa.io/en/stable/) to install einops and git to clone repo of the model and dataset.


```bash

!pip install einops

!git clone https://github.com/cure-lab/LTSF-Linear

!git clone https://github.com/zhouhaoyi/ETDataset.git

```


## Usage

- If you use kaggle run you don't have to do anything. Just upload the notebook and run it on kaggle. Find it in a notebook called alldata+custom_using_fedformer_(kaggle_run).ipynb.

- If you use another server, do installation, change the directory path and you need to add a new data in data_factory file. You can find it in the first cells. Find it in a notebook called alldata+custom_using_fedformer_(Queen's_Server_run).ipynb.


- LINE ADDED IN data_factory.py file 


```python

'Energy_consumption_Evaluation_': Dataset_ETT_hour

```


SAMPLE OF SCRIPT 


```python

pre_lens = [96, 192, 336, 720]


for pre_len in pre_lens:

       # ETTm1

        os.system(f"python -u /kaggle/working/LTSF-Linear/FEDformer/run.py "

                  f"--is_training 1 "

                  f"--root_path /kaggle/working/ETDataset/ETT-small "

                  f"--data_path ETTm1.csv "

                  f"--task_id ETTm1 "

                  f"--model FEDformer "

                  f"--data ETTm1 "

                  f"--features M "

                  f"--seq_len 96 "

                  f"--label_len 48 "

                  f"--pred_len {pre_len} "

                  f"--e_layers 2 "

                  f"--d_layers 1 "

                  f"--factor 3 "

                  f"--enc_in 7 "

                  f"--dec_in 7 "

                  f"--c_out 7 "

                  f"--des 'Exp' "

                  f"--d_model 512 "

                  f"--itr 3")

```

## The Results :
[Results](https://drive.google.com/drive/folders/1zSrlBxoTDQo6BlEm9lxzc1FBkKLMwGAa?usp=sharing)

## Reference


- [Paper of FEDformer](https://arxiv.org/pdf/2201.12740.pdf)


- [Repo of FEDformer](https://arxiv.org/pdf/2201.12740.pdf)
