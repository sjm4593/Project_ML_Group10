# Modified Hinglish Sentiment Classifier

<img src="./Hinglish-Logo.png" width="500">

## Environment setup

```bash
conda create -n hinglish python=3.8
```

```bash
conda activate hinglish
```

```bash
conda install --file conda_requirements.txt
```

```bash
pip install -r pip_requirements.txt
```

```bash
conda install -y pytorch==1.6.0 cudatoolkit=10.2 -c pytorch -c nvidia
```

## Data and Model files

Run this in a terminal and save these in the root directory
```python
from hinglishutils import get_files_from_gdrive
#%%
get_files_from_gdrive("https://drive.google.com/file/d/1-Ki6v1a1jF79qx22gM6JlX1NVD4txTdn/view?usp=sharing", 
                      "train_lm.txt")

get_files_from_gdrive("https://drive.google.com/file/d/1-MRU7w2_la36qopO8Ob4BoCynOAZc0sZ/view?usp=sharing", 
                      "dev_lm.txt")

get_files_from_gdrive("https://drive.google.com/file/d/1-NqiU-tL5hW59MFtUXh1exivRokZKfs7/view?usp=sharing", 
                      "test_lm.txt")
#%%
get_files_from_gdrive("https://drive.google.com/file/d/1k4N0JlVOP-crIcCtC6ZI5Va8X3s2-r_D/view?usp=sharing", 
                      "test_labels_hinglish.txt")
#%%
get_files_from_gdrive("https://drive.google.com/file/d/1-FykBMdD7erRhr9370thtySNm6QvnQAA/view?usp=sharing", 
                      "train.json")

get_files_from_gdrive("https://drive.google.com/file/d/1-F6o4lSub2D-_iCoNPvxxnCiPQ82VJjG/view?usp=sharing", 
                      "test.json")

get_files_from_gdrive("https://drive.google.com/file/d/1-Esp4UtIZwX44eI8qndngweKZ6p9GLKT/view?usp=sharing", 
                      "valid.json")

get_files_from_gdrive("https://drive.google.com/file/d/17wFvtj9tfp4QI6FrErAyqL9H1s5-lZkR/view?usp=sharing", 
                      "final_test.json")
```

Download the pretrained models directly from Google drive, and extract the tarballs in the root directory using:

https://drive.google.com/file/d/1-0bVrbhQ3nJhwmgIdhuL-ws4V9zuFpMF/view?usp=sharing
https://drive.google.com/file/d/1I1JXDg8ZzuuzXMN1X986oCeOjSxqZj7C/view?usp=sharing
https://drive.google.com/file/d/1TTJzXi0dWYHVCrZM8vWoZzKWPfIF0ErB/view?usp=sharing

```bash
tar -xvf filename.tar.gz
```

Run a small version using:

```bash
python adaboost_run.py --output_dir=bert --model_type=bert --model_name_or_path=bert-base-multilingual-cased --do_train --train_data_file=ada_data_small.txt --do_eval --eval_data_file=ada_data_small.txt --mlm  --num_train_epochs 1 --save_total_limit 2 --n_models 5 --overwrite_output_dir
```

For the full training, run (replace GPU_INDICES_HERE with the GPU indices you want to use with CUDA):

```bash
CUDA_VISIBLE_DEVICES=GPU_INDICES_HERE python adaboost_run.py --output_dir=bert --model_type=distilbert  --do_train --model_name_or_path=distilbert-base-cased --train_data_file=train_lm.txt --do_eval --eval_data_file=test_lm.txt --mlm  --num_train_epochs 5 --save_total_limit 10 --n_models 1 --overwrite_output_dir --per_gpu_train_batch_size=4
```