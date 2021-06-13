---
language: ja
datasets: wikipedia
widget:
- text: "統計的推定を使うことで、"
license: cc-by-sa-3.0
---

# GPT-2 small Japanese model

This repository contains a pretrained SentencePiece tokenizer model and GPT-2 small model trained on Japanese Wikipedia dataset.

## Training data

[Japanese Wikipedia](https://ja.wikipedia.org/wiki/Wikipedia:データベースダウンロード) dataset which is released under [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/) is used for training both the tokenizer and GPT-2 model as of March 1st, 2021.
The dataset is splitted into three subsets - train, valid and test. Both of tokenizer and model are trained with the train split.

## Model description

The model architecture is the same as GPT-2 small model (n_ctx: 1024, n_embd 768, n_head: 12, n_layer: 12) except for a vocabulary size.
The vocabulary size is set to 32,000 instead of an original size of 50,257.
`transformers.GPT2LMHeadModel` is used for training.

## Tokenizer description

[SentencePiece](https://github.com/google/sentencepiece) tokenizer is used as a tokenizer for this model.

In a training, the tokenizer model is trained with 10,000,000 samples which are extracted from the train split of training data.
The vocabulary size is set to 32,000. A `add_dummy_prefix` option is set to `True` because words are not separated by whitespaces in Japanese.

After training, the model is imported to `transformers.BERTGenerationTokenizer` because it supports SentencePiece models and it does not add any special tokens as default, which is useful expecially for a text generation task.

## Training

The model is trained on the train split for 30 epochs with batch size 32 and 1024 tokens for each sample (i.e. 32,768 tokens are processed in each batch).
Each epoch contains around 16,000 steps.
Adam optimizer is used. The learning rate linearly increased from `0` to `1e-4` in first 10,000 steps, then decreased to `0`.
A clip norm is also used to set to `1.0`.

Trainind was conducted on Ubuntu 18.04.5 LTS with one RTX 2080 Ti.

After training completed, test set PPL was reached to 28.47.

All the code to train tokenizer and GPT-2 models are available in [a GitHub repository](https://github.com/colorfulscoop/gpt2-ja).

## Usage

First, install dependecies.

```sh
$ pip install transformers==4.4.2 torch==1.8.0 sentencepiece==0.1.95
```

Then load the pretrained tokenizer and GPT-2 model, and call a `generate` method.

```sh
>>> import transformers, torch
>>> tokenizer = transformers.AutoTokenizer.from_pretrained("colorfulscoop/gpt2-small-ja")
>>> model = transformers.AutoModelForCausalLM.from_pretrained("colorfulscoop/gpt2-small-ja")
>>> input = tokenizer.encode("統計的推定を使うことで、", return_tensors="pt")
>>> output = model.generate(input, do_sample=True, top_p=0.95, top_k=50, num_return_sequences=3)
>>> tokenizer.batch_decode(output)
['統計的推定を使うことで、より精密に測定できる場合もある。すなわち、誤差があるか、測定', '統計的推定を使うことで、2億5000万年の間に人類の居住に不可欠な金属元素と', '統計的推定を使うことで、データの正確さによって、推定結果から予測された推定量を決定する']
```

**Note:** The default model configuration `config.json` sets some generation parameters with `do_sample=True`, `top_k=50`, `top_p=0.95`. Please reset these parameters when you need to set different parameters.

## Versions

| Version | Commit hash |
| --- | --- |
| latest | main |
| v1 | 697bc101c1bb775d8635509f54e66b51d633fd94 |

All models are using the same tokenizer model as v1.
If you use a previous model, please specify the corresponded revision to `AutoModelForCausalLM.from_pretrained` method.


```sh
>>> import transformers
>>> tokenizer = transformers.AutoTokenizer.from_pretrained("colorfulscoop/gpt2-small-ja")
>>> model = transformers.AutoModelForCausalLM.from_pretrained("colorfulscoop/gpt2-small-ja", revision="697bc101c1bb775d8635509f54e66b51d633fd94")
```

## License

All the models included in this repository are licensed under [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

**Disclaimer:** The model potentially has possibility that it generates similar texts in the training data, texts not to be true, or biased texts. Use of the model is at your sole risk. Colorful Scoop makes no warranty or guarantee of any outputs from the model. Colorful Scoop is not liable for any trouble, loss, or damage arising from the model output.

**Author:** Colorful Scoop
