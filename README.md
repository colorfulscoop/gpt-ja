# GPT-based Japanese model for 🤗 Transformers

This repository is for GPT-based Japanese model trained on Japanese Wikipedia dataset.

Current support models are:

* [GPT2](https://huggingface.co/transformers/model_doc/gpt2.html)
* [GPT Neo](https://huggingface.co/transformers/model_doc/gpt_neo.html)

Model summary:

| 🤗 Model Hub | Data | Revision | Code | Total params | Test set PPL | vocab_size | n_ctx | n_layer | n_head | n_embd | Epochs | Training time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [colorfulscoop/gpt2-small-ja](https://hf.co/colorfulscoop/gpt2-small-ja) | jawiki_20210820 | 20210820.1.0 | [ef927e1](https://github.com/colorfulscoop/gpt-ja/tree/ef927e1717dfb34c810c9227bbcfcf438f01be9d) | 110M | 29.13 | 32,000 | 1,024 | 12 | 12 | 768 | 30 | 15 days |
| | jawiki_20210301 | 20210301.1.0 | - | 110M | - | 32,000 | 1,024 | 12 | 12 | 768 | 30 | - |

Data summary:

| Id | Corpus | #tokens in train set | #tokens in valid set | #tokens in test set |
| --- | --- | --- | --- | --- |
| jawiki_20210820 | Japanese Wikipedia on 20210820 | 540M | 13M | 13M |

**Note:** a same tokenizer is used if models are trained on same data.

Sample usage:

```py
>>> import transformers
>>> pipeline = transformers.pipeline("text-generation", "models/gpt2-small", revision="20210820.1.0")
>>> pipeline("統計的機械学習でのニューラルネットワーク", do_sample=True)
[{'generated_text': '統計的機械学習でのニューラルネットワークの解析は、多くのアルゴリズムの完全な実装をもたらした。これらの'}]
```


Although this repository is being maintained to make all the models work in the latest master branch,
please check out the specific commit hash of training run to reproduce the result.

## Training details

Training model was conducted on the following environment.

* OS: Ubuntu 18.04.5 LTS
* GPU:  RTX 2080 Ti x1

## Environment preparation

```sh
$ docker container run --gpus all --ipc=host --rm -it -v $(pwd):/work -w /work nvidia/cuda:11.1-devel-ubuntu20.04 bash
(container)$ apt update && apt install -y python3 python3-pip git wget
(container)$ pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
(container)$ pip3 install -r requirements.txt
```

### Data preparation

Check the latest date in the list from https://dumps.wikimedia.org/jawiki/ .

```sh
(container)$ bash src/get_jawiki.sh 20210820 input
```

Finally generated data can be found under `input` directory.

```sh
(container)$ ls -1 input/20210820/{train,valid,test}.txt
input/20210820/test.txt
input/20210820/train.txt
input/20210820/valid.txt
```

### Train tokenizer

Train SentencePiece model in the same container used in data peparation.

```sh
(container)$ python3 src/train_tokenizer.py --train_file input/20210820/train.txt --model_dir models/gpt2-small
```

### Train model

Run training with the config file:

```sh
(container)$ python3 src/train.py train --config input/gpt2-small.json
...
255999it [10:21:51,  7.03it/s]{'epoch': 30, 'batch': 256000, 'step': 493108, 'train_loss': 0.190585415356369, 'lr': 0.0001}
263236it [10:39:12,  6.86it/s]
6788it [10:28, 10.81it/s]
{'epoch': 30, 'valid_loss': 3.417723441833458, 'valid_ppl': 30.49990112587307, 'save_model': True}
```

### Test

```sh
(container)$ python3 src/train.py test --config input/gpt2-small.json
6793it [09:16, 12.20it/s]
{'test_loss': 3.371613106758486, 'test_ppl': 29.125471679484484}
```


### Export Tensorflow model

```py
(container)$ pip install tensorflow
(container)$ python3
>>> from transformers import TFGPT2LMHeadModel
>>> model = TFGPT2LMHeadModel.from_pretrained("models/gpt2-small", from_pt=True)
>>> model.save_pretrained("models/gpt2-small")
```

### Upload to 🤗 Model Hub

Follow [official document](https://huggingface.co/transformers/model_sharing.html) to upload model.

#### Prepare environment

Prepare git lfs. In a MacOS environment, git lfs can be installed as follows.

```sh
$ brew install git-lfs
$ git lfs install
Updated git hooks.
Git LFS initialized.
```

Then clone the repository.

```sh
$ git clone https://huggingface.co/colorfulscoop/gpt2-small-ja release/gpt2-small-ja
```

#### Copy model to release directory

```sh
$ cp models/gpt2-small/* release/gpt2-small-ja/
cp: models/gpt2-small/spm is a directory (not copied).
$ cd release/gpt2-small-ja
```

Then, modify `config.json` to specify default generation values by following diff.

```sh
   "unk_token_id": 1,
   "use_cache": true,
-  "vocab_size": 32000
+  "vocab_size": 32000,
+  "top_k": 50,
+  "top_p": 0.95,
+  "do_sample": true
 }
```

Commit changes to git.

```sh
$ git add .
```

#### Release

```sh
$ git push origin
```
