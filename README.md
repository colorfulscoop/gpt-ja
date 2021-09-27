# GPT-based Japanese model for 🤗 Transformers

This repository is for GPT-based Japanese model trained on Japanese Wikipedia dataset.

Current support models are:

* [GPT2]()
* [GPT Neo](https://huggingface.co/transformers/model_doc/gpt_neo.html)

Model summary:

| Model in 🤗 Model Hub | Revision | Data | Total params | vocab_size | n_ctx | n_layer | n_head | n_embd | Epochs | Test PPL | Training time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [colorfulscoop/gpt2-small-ja](https://hf.co/colorfulscoop/gpt2-small-ja) | v20210820.1.0 | jawiki_20210820 | 110M | 32,000 | 1,024 | 12 | 12 | 768 | 30 | | 15 days |

Data summary:

| Id | Corpus | #tokens in train set | #tokens in valid set | #tokens in test set |
| --- | --- | --- | --- | --- |
| jawiki_20210820 | Japanese Wikipedia on 20210820 | 540M | 13M | |

**Note:** a same tokenizer is used if models are trained on same data.

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
(container)$ python3 src/train_tokenizer.py --train_file input/20210820/train.txt --model_dir models/small-v2
```


### Train model

Run training with the config file:

```sh
(container)$ python3 src/train.py train --config input/config-small-v2.json
```

### Test

Once your model is trained, use `test.py` script to measure loss and PPL metrics.
You can specify a config file and checkpoint which PyTorch Lightning automatically saves.

```sh
$ python test.py --config lightning_logs/version_0/config.yaml --ckpt_path lightning_logs/version_0/checkpoints/epoch\=2-step\=8.ckpt
```

### Upload to 🤗 Model Hub

