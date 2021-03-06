import torch
import pydantic
from typing import Optional, List, Any
import transformers
import tqdm
import json
import numpy as np
import math
import random


class FileIterator:
    def __init__(self, filename):
        self._filename = filename

    def __iter__(self):
        with open(self._filename) as fd:
            for line in fd:
                yield line.strip("\n")


class BlockDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterator, tokenizer, block_size, drop_last=True):
        super().__init__()
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._iterator = iterator
        self._drop_last = drop_last

    @classmethod
    def from_file(cls, filename, tokenizer, block_size):
        iterator = FileIterator(filename=filename)
        return cls(
            iterator=iterator,
            tokenizer=tokenizer,
            block_size=block_size
        )

    def __iter__(self):
        """
            Yields (List[int])
        """
        ids = []
        for text in self._iterator:
            ids.extend(self._tokenizer.encode(text))
            while len(ids) >= self._block_size+1:
                yield {"input_ids": ids[:self._block_size],
                       "labels": ids[1:self._block_size+1]}
                ids = ids[self._block_size:]
        if not self._drop_last:
            yield {"input_ids": ids[:-1],
                   "labels": ids[1:]}

    @classmethod
    def collate_fn(cls, item):
        """Collate function for DataLoader
        Args:
            item (List[dict[str, List[int]]]): BlockDataset のイテレータが返す辞書のリスト
        Returns:
            (dict[str, torch.Tensor]):
        """
        keys = item[0].keys()
        dic = {
            key: torch.tensor([x[key] for x in item])
            for key in keys
        }
        return dic


def build_dataloader(
    filename, block_size, tokenizer, batch_size,
    shuffle_buffer_size, prefetch_factor, num_workers,
):
    dataset = BlockDataset.from_file(
        block_size=block_size,
        tokenizer=tokenizer,
        filename=filename
    )
    if shuffle_buffer_size:
        dataset = torch.utils.data.BufferedShuffleDataset(
            dataset,
            buffer_size=shuffle_buffer_size,
        )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=BlockDataset.collate_fn,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )
    return loader


def forward(model, item, loss_fn, device):
    """1バッチ毎のロスの計算を行う。

    item は DataLoader が返す辞書オブジェクトで `input_ids` と `labels` キーからなる。
    各々さずは (batch_size, input_len) となる。
    """
    # テンソルの to はインプレースではないので代入しないといけないということであっている？
    src, tgt = item["input_ids"], item["labels"]

    # [*4] テンソルを対象デバイスに移す。
    # テンソルの `to` はモジュールの `to` と異なりインプレースでデバイスに移らず、
    # 移動した先の新しいテンソルを返すので、必ず代入を行うこと
    src = src.to(device=device)
    tgt = tgt.to(device=device)

    # ロスを計算する
    output = model(input_ids=src)
    logits = output.logits  # shape: (batch_size, input_len, vocab_size)
    loss = loss_fn(
        input=logits.view(-1, logits.shape[-1]),
        target=tgt.view(-1)
    )
    return loss


def progress_bar(seq, show: bool):
    return tqdm.tqdm(seq) if show else seq


def train(
    config,
    model,
    tokenizer,
    optimizer,
    scheduler,
    loss_fn,
    train_dataloader,
    valid_dataloader,
    device
):
    model.to(device=device)

    # Setup scaler for SMP
    # Please refer to https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

    # variables to use in log
    num_steps = 0

    # keep best model loss
    best_val_loss = float("infinity")

    for epoch in range(1, config.epochs+1):
        # [*1] 学習モード
        model.train()

        for train_batch_idx, item in progress_bar(enumerate(train_dataloader, start=1), show=config.show_progress_bar):
            # ロスの計算グラフを構築する
            # forward 関数は、検証時にも利用するため別の関数で後で定義する
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                loss = forward(model, item, loss_fn, device)
                loss = loss / config.accumulation_steps

            # 勾配を計算し、その結果をテンソルの.gradに保存する
            scaler.scale(loss).backward()

            if train_batch_idx % config.accumulation_steps == 0:
                # 勾配に従ってオプティマイザに登録したパラメータ (required_grad=Trueのテンソル) を更新
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                # [*2] 勾配の初期化Go
                optimizer.zero_grad()

                num_steps += 1

            # エポックのロス計算は、勾配計算を行わないため計算グラフを構築する必要はない。
            # 計算グラフを構築しないために item を使ってテンソルの中身を取り出して計算している。
            # item を使わないと計算グラフをバッチのループ毎に作り続けそれを train_loss にキープし続けるため、
            # メモリを大量に消費してしまう

            # ログの出力
            if train_batch_idx % (config.accumulation_steps * config.log_steps) == 0:
                batch_log = dict(
                    epoch=epoch,
                    batch=train_batch_idx,
                    step=num_steps,
                    train_loss=loss.item(),
                    lr=optimizer.param_groups[0]['lr'],
                )
                print(batch_log)

        # [*1] 検証モード
        model.eval()
        # [*3] 推論モードでは勾配計算しないので計算グラフを作成する必要がない。
        #      `torch.no_grad()` コンテキスト内のテンソルの計算では計算グラフは構築されない。
        with torch.no_grad():
            val_loss = 0
            for val_batch_idx, item in progress_bar(enumerate(valid_dataloader, start=1), show=config.show_progress_bar):
                loss = forward(model, item, loss_fn, device)
                val_loss += loss.item()

                # 次の行の assert で計算グラフが構築されていないことが確認できる。
                # assert loss.grad is None

        # Update best validation loss
        val_loss_per_batch = val_loss/val_batch_idx
        save_model = True

        if val_loss_per_batch < best_val_loss:
            best_val_loss = val_loss_per_batch
        else:
            if config.save_best_model:
                save_model = False

        if save_model:
            model.save_pretrained(config.output_path)
            tokenizer.save_pretrained(config.output_path)

        epoch_log = dict(
            epoch=epoch,
            valid_loss=val_loss_per_batch,
            valid_ppl=math.exp(val_loss_per_batch),
            save_model=save_model,
        )
        print(epoch_log)


def set_reproducibility(seed: int = None, deterministic: bool = False):
    """
    Refer to the document for details
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    torch.use_deterministic_algorithms(deterministic)


class TrainConfig(pydantic.BaseModel):
    model_type: str

    # Required parameters
    output_path: str
    tokenizer_model: str
    train_file: str
    valid_file: str
    test_file: str

    # [Training options]
    epochs: int = 1
    batch_size: int = 2
    prefetch_factor: int = 10
    workers: int = 1
    shuffle_buffer_size: int = 1000
    lr: float = 1e-4
    warmup_steps: int = 0
    training_steps: Optional[int] = None
    use_amp: bool = False
    accumulation_steps: int = 1
    show_progress_bar: bool = True
    log_steps: int = 100
    seed: Optional[int] = None
    deterministic: bool = False
    save_best_model: bool = True


class GPT2TrainConfig(TrainConfig):
    # [Model config]
    # for small
    n_ctx: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    # for medium -> n_layer=24, n_head=16, n_embd=1024
    # for large  -> n_layer=36, n_head=20, n_embd=5120
    # for XL     -> n_layer=48, n_head=24, n_embd=6400

    block_size: int = 1024


class GPTNeoTrainConfig(TrainConfig):
    # [Model config]
    # for small
    attention_types: List[Any] = [[[["global", "local"], 6]]]
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 1024

    block_size: int = 1024


def load_config(config_file):
    json_dic = json.load(open(config_file))
    model_type = json_dic["model_type"]
    assert model_type in ["gpt2", "gpt_neo"]

    if model_type == "gpt2":
        model_cls = GPT2TrainConfig
    elif model_type == "gpt_neo":
        model_cls = GPTNeoTrainConfig

    config = model_cls.parse_file(config_file)
    return config


def init_model(config, tokenizer):
    model_type = config.model_type

    if model_type == "gpt2":
        model_config = transformers.GPT2Config(
            vocab_size=len(tokenizer),
            tokenizer_class=tokenizer.__class__.__name__,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            sep_token_id=tokenizer.sep_token_id,
            cls_token_id=tokenizer.cls_token_id,
            unk_token_id=tokenizer.unk_token_id,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            n_ctx=config.n_ctx,
        )
        model = transformers.GPT2LMHeadModel(model_config)
    elif model_type == "gpt_neo":
        model_config = transformers.GPTNeoConfig(
            vocab_size=len(tokenizer),
            tokenizer_class=tokenizer.__class__.__name__,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            sep_token_id=tokenizer.sep_token_id,
            cls_token_id=tokenizer.cls_token_id,
            unk_token_id=tokenizer.unk_token_id,
            attention_types=config.attention_types,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
        )
        model = transformers.GPTNeoForCausalLM(model_config)

    return model


class Trainer:
    def train(self, config):
        config = load_config(config)

        # Set Reproducibility
        set_reproducibility(seed=config.seed, deterministic=config.deterministic)

        # Define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_model)

        # Initialize model
        model = init_model(config, tokenizer)

        # Load data
        train_dataloader = build_dataloader(
            filename=config.train_file,
            block_size=config.block_size,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            shuffle_buffer_size=config.shuffle_buffer_size,
            prefetch_factor=config.prefetch_factor,
            num_workers=config.workers,
        )
        valid_dataloader = build_dataloader(
            filename=config.valid_file,
            block_size=config.block_size,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            shuffle_buffer_size=None,
            prefetch_factor=config.prefetch_factor,
            num_workers=config.workers,
        )

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # Scheduler
        if config.training_steps:
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.training_steps
            )
        else:
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.warmup_steps,
            )

        # Loss function
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        train(
            config=config,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            device=device,
        )

    def test(self, config):
        config = load_config(config)

        # Define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = transformers.AutoTokenizer.from_pretrained(config.output_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(config.output_path)

        # Load data
        test_dataloader = build_dataloader(
            filename=config.test_file,
            block_size=config.block_size,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            shuffle_buffer_size=None,
            prefetch_factor=config.prefetch_factor,
            num_workers=config.workers,
        )

        # Loss function
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        model.to(device=device)
        model.eval()

        with torch.no_grad():
            test_loss = 0
            for batch_idx, item in progress_bar(enumerate(test_dataloader, start=1), show=config.show_progress_bar):
                loss = forward(model, item, loss_fn, device)
                test_loss += loss.item()

        test_loss = test_loss / batch_idx
        test_log = dict(
            test_loss=test_loss,
            test_ppl=math.exp(test_loss),
        )
        print(test_log)


if __name__ == "__main__":
    import fire

    fire.Fire(Trainer)
