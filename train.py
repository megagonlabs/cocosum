import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import set_seed

from model import Summarizer, CoCoTripModule

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = Summarizer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=765)
    parser.add_argument("--task", type=str, default="cont")
    parser.add_argument("--ckpt", type=str, default=None)

    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    if args.ckpt is None:
        summarizer = Summarizer(**vars(args))
    else:
        from pathlib import Path

        ckpt_path = str(next(Path(args.ckpt).glob("*.ckpt")))
        summarizer = Summarizer.load_from_checkpoint(ckpt_path, **vars(args))
    datamodule = CoCoTripModule(tokenizer=summarizer.tokenizer, **vars(args))

    checkpoint_callback = ModelCheckpoint(monitor="val_rouge-12lf",
                                          verbose=True,
                                          save_top_k=1,
                                          mode="max",
                                          filename="{epoch}-{val_rouge-12lf:.3f}")
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(summarizer, datamodule=datamodule)
    trainer.test()
