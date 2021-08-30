# GPT-2 Japanese model for 🤗 Transformers

This repository is for GPT-2 Japanese model trained on Japanese Wikipedia dataset.

Model details are as follows. Please check out more details of each model from each document in Model Hub.

| Model in 🤗 Model Hub| n_ctx | n_layer | n_head | n_embd |
| --- | --- | --- | --- | --- |
| [gpt2-small-ja](https://hf.co/colorfulscoop/gpt2-small-ja) | 1024 | 12 | 12 | 768 |

Simple usage;

```sh
>>> import transformers
>>> transformers.pipeline(...)
```

Following document shows how to reproduce model training.

## Training details

Training model was conducted on the following environment.

* OS: Ubuntu 18.04.5 LTS
* GPU:  RTX 2080 Ti x1

### Data preparation

```sh
$ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.6-slim-buster bash
(container)$ apt update && apt install -y wget git
```

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

```sh
(container)$ exit
```

### Train tokenizer

Train SentencePiece model in the same container used in data peparation.

```sh
(container)$ pip install -r requirements.txt
(container)$ python src/train_tokenizer.py --train_file input/20210820/train.txt --model_dir models/small-v2
```

```sh
(container)$ exit
```

### Train model

Install PyTorch with CUDA 11.1 and dependent packages.

```sh
$ docker container run --gpus all --ipc=host --rm -it -v $(pwd):/work -w /work nvidia/cuda:11.1-devel-ubuntu20.04 bash
(container)$ apt update && apt install -y python3 python3-pip git
(container)$ pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
(container)$ pip3 install -r requirements.txt
```

Training script uses [PyTorch Lightning CLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html).

First, create your config file and copy it for modification.

```sh
(container)$ python3 src/trainer.py --print_config >input/default_config.yaml
(container)$ cp input/default_config.yaml input/config-small-v2.yaml
```

Second, modify the config file to set up your parameters for training.  Following parameters are recommended to set up.

**For trainer parameters:**

| params | what to set | example |
| --- | --- | --- |
| trainer.seed_everything | Set an int value as a seed for reproducibility | 1000 |
| trainer.max_epochs | Set the number of epochs | 10 |
| trainer.deterministic | Set true to ensure reproducibility while training on GPU | true |
| trainer.gpu | | 1 |
| [trainer.precision](https://pytorch-lightning.readthedocs.io/en/stable/advanced/amp.html) | Set 16 for 16-bit training if while training on GPU | 16 |
| [trainer.accumulate_grad_batches](https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#accumulate-gradients) | Set the number of batches to calculate gradient for updating parameters | 16 |
| [trainer.gradient_clip_val](https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#gradient-clipping) | Set a value to clip gradient | 1.0 |

Following setting might be useful when you need to monitor values in TensorBoard:

```yaml
trainer:
  callbacks:
    - class_path: pytorch_lightning.callbacks.LearningRateMonitor
    - class_path: pytorch_lightning.callbacks.GPUStatsMonitor
```

**For model parameters:**

The default parameters for GPT2 is for small model. You can specify `model.block_size`, `model.n_layer`, `model.n_head`, `model.n_embd` parameters to change the network size.

| Parameter | Description | Example Value |
| --- | --- | --- |
| model.tokenizer_model | Set GPT2 tokenizer model path | models/small-v2/ |
| model.train_file | Set text file for train your language model | models/small-v2 |
| model.valid_file | Set text file for validate your language model | input/20210820/valid.txt |
| model.test_file | Set text file for test your language model | input/20210820/test.txt |
| model.block_size | Set context size of GPT2 model | 1024 |
| model.n_layer | Set the number of layers of GPT2 model | 12 |
| model.n_head | Set the number of attention head of GPT2 model | 12 |
| model.n_embd | Set the embedding dimension of GPT2 model | 768 |

Then run training with the config file:

```sh
(container)$ python3 src/trainer.py --config input/config-small-v2.yaml
```

While training, you can check log via TensorBoard

```sh
$ docker container run -p 6006:6006 -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.4.1-gpu tensorboard --logdir lightning_logs --host 0.0.0.0
```

### Test

Once your model is trained, use `test.py` script to measure loss and PPL metrics.
You can specify a config file and checkpoint which PyTorch Lightning automatically saves.

```sh
$ python test.py --config lightning_logs/version_0/config.yaml --ckpt_path lightning_logs/version_0/checkpoints/epoch\=2-step\=8.ckpt
```

### Export model

Finally `export_model.py` exports transformers' models under a directory specified by `--output_dir`.
This script also saves your tokenizer model.

```sh
$ python export_model.py --config lightning_logs/version_0/config.yaml --ckpt_path lightning_logs/version_0/checkpoints/epoch\=2-step\=8.ckpt --output_dir model
```

### Upload to 🤗 Model Hub