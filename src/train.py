from re import M
import torch
import pydantic
from typing import Optional
import transformers
import tqdm


class BlockDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator, tokenizer, block_size, drop_last=True):
        super().__init__()
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._generator = generator
        self._drop_last = drop_last

    @classmethod
    def from_texts(cls, texts, tokenizer, block_size):
        """
        Args:
            tokenizer (transformers.AutoTokenizer)
            texts (List[str])
            block_size (int)
        """
        return cls(
            generator=lambda: texts,
            tokenizer=tokenizer,
            block_size=block_size
        )

    @classmethod
    def from_file(cls, filename, tokenizer, block_size):
        return cls(
            generator=lambda: (line.strip("\n") for line in open(filename)),
            tokenizer=tokenizer,
            block_size=block_size
        )

    def __iter__(self):
        """
            Yields (List[int])
        """
        ids = []
        for text in self._generator():
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

            # エポックのロス計算は、勾配計算を行わないため計算グラフを構築する必要はない。
            # 計算グラフを構築しないために item を使ってテンソルの中身を取り出して計算している。
            # item を使わないと計算グラフをバッチのループ毎に作り続けそれを train_loss にキープし続けるため、
            # メモリを大量に消費してしまう

            # ログの出力
            if train_batch_idx % (100*config.accumulation_steps) == 0:
                batch_log = dict(
                    epoch=epoch,
                    batch=train_batch_idx,
                    step=train_batch_idx / config.accumulation_steps,
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
            for val_batch_idx, item in enumerate(valid_dataloader, start=1):
                loss = forward(model, item, loss_fn, device)
                val_loss += loss.item()

                # 次の行の assert で計算グラフが構築されていないことが確認できる。
                # assert loss.grad is None

        epoch_log = dict(
            epoch=epoch,
            valid_loss=val_loss/val_batch_idx,
        )
        print(epoch_log)


class TrainConfig(pydantic.BaseModel):
    # Required parameters
    tokenizer_model: str
    train_file: str
    valid_file: str

    # [Model config]
    # for small
    n_ctx: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    # for medium -> n_layer=24, n_head=16, n_embd=1024
    # for large  -> n_layer=36, n_head=20, n_embd=5120
    # for XL     -> n_layer=48, n_head=24, n_embd=6400

    # [Training options]
    epochs: int = 1
    batch_size: int = 2
    prefetch_factor: int = 10
    workers: int = 1
    shuffle_buffer_size: int = 1000
    lr: float = 1e-4
    warmup_steps: int = 0
    steps: Optional[int] = None
    use_amp: bool = False
    accumulation_steps: int = 1
    show_progress_bar: bool = False


class Trainer:
    def train(self, **train_args):
        config = TrainConfig(**train_args)

        # Define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_model)

        # Prepare model
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

        # Load data
        train_dataloader = build_dataloader(
            filename=config.train_file,
            block_size=config.n_ctx,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            shuffle_buffer_size=config.shuffle_buffer_size,
            prefetch_factor=config.prefetch_factor,
            num_workers=config.workers,
        )
        valid_dataloader = build_dataloader(
            filename=config.train_file,
            block_size=config.n_ctx,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            shuffle_buffer_size=config.shuffle_buffer_size,
            prefetch_factor=config.prefetch_factor,
            num_workers=config.workers,
        )

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # Scheduler
        if config.steps:
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=config.steps
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
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            device=device
        )


if __name__ == "__main__":
    import fire

    fire.Fire(Trainer)
