# GPT-2 Japanese model for HuggingFace's transformers

## Data preparation

```sh
$ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.6-slim-buster bash
(container)$ apt update && apt install -y wget git
```

Check the latest date in the list from https://dumps.wikimedia.org/jawiki/ .

```sh
(container)$ cd src
(container)$ bash src/get_jawiki.sh 20210820 input
```

Finally generated data can be found under `input` directory.

```sh
(container)$ ls input/20210301
test.txt  train.txt  valid.txt
```

```sh
(container)$ exit
```

### Train tokenizer

Train SentencePiece model.

```sh
$ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.6-slim-buster bash
(container)$ pip install -r requirements.txt
(container)$ python train_tokenizer.py --train_file data/jawiki/20210301/data/train.txt
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

```sh
$ python trainer.py --print_config > default_config.yaml
```

### Train

Training script uses [PyTorch Lightning CLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html).

First, create your config file.

```sh
$ python trainer.py --print_config > default_config.yaml
```

Second, modify the default_config.yaml file to set up your parameters for training.
Following parameters are recommended to set up.

**For trainer parameters:**

| params | what to set | example |
| --- | --- | --- |
| trainer.seed_everything | Set an int value as a seed for reproducibility | 1000 |
| trainer.max_epochs | Set the number of epochs | 10 |
| trainer.deterministic | Set true to ensure reproducibility while training on GPU | true |
| [trainer.precision](https://pytorch-lightning.readthedocs.io/en/stable/advanced/amp.html) | Set 16 for 16-bit training if while training on GPU | 16 |
| [trainer.accumulate_grad_batches](https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#accumulate-gradients) | Set the number of batches to calculate gradient for updating parameters | 16 |
| [trainer.gradient_clip_val](https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#gradient-clipping) | Set a value to clip gradient | 1 |

Following setting might be useful when you need to monitor values in TensorBoard:

```yaml
trainer:
  callbacks:
    - class_path: pytorch_lightning.callbacks.LearningRateMonitor
    - class_path: pytorch_lightning.callbacks.GPUStatsMonitor
```

**For model parameters:**

The default parameters for GPT2 is for small model. You can specify `model.block_size`, `model.n_layer`, `model.n_head`, `model.n_embd` parameters to change the network size.

| model.tokenizer_model | Set GPT2 tokenizer model on [Hugging Face Model Hub](https://huggingface.co/models) | colorfulscoop/gpt2-small-ja |
| --- | --- | --- |
| model.train_file | Set text file for train your language model | data/train.txt |
| model.valid_file | Set text file for validate your language model | data/valid.txt |
| model.test_file | Set text file for test your language model | data/test.txt |
| model.block_size | Set context size of GPT2 model | 1024 |
| model.n_layer | Set the number of layers of GPT2 model | 12 |
| model.n_head | Set the number of attention head of GPT2 model | 12 |
| model.n_embd | Set the embedding dimension of GPT2 model | 768 |

Assume that the modified config file is saved as `config.yaml`

Then run training with the config file:

```sh
$ python trainer.py --config config.yaml
```

While training, you can check log via TensorBoard

```sh
docker container run -p 6006:6006 -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.4.1-gpu tensorboard --logdir lightning_logs --host 0.0.0.0
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

This allows you to load your model from transformers library as usual way.

```sh
>>> import transformers
>>> transformers.AutoTokenizer.from_pretrained("model")
>>> transformers.AutoModelForCausalLM.from_pretrained("model")
```
