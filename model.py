import json
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import rouge
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, LEDTokenizerFast
from transformers.trainer_pt_utils import LabelSmoother

from led import LEDForConditionalGeneration
from reader import CoCoTrip


def load_train(train_file, tokenizer):
    max_length = tokenizer.model_max_length
    model_inputs = []
    for ins in tqdm(list(map(json.loads, open(train_file))), desc="Loading...", ncols=80):
        input_ids = torch.tensor([x for r in tokenizer(ins["src"]).input_ids for x in r])
        token_type_ids = torch.zeros_like(input_ids)
        if "counter" in ins:
            counter_ids = torch.tensor([x for r in tokenizer(ins["counter"]).input_ids for x in r])
            counter_type_ids = torch.ones_like(counter_ids)
        else:
            counter_ids = counter_type_ids = torch.tensor([0])

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(ins["tgt"], max_length=max_length, truncation=True).input_ids
        labels = [(x if x != tokenizer.pad_token_id else -100) for x in labels]
        labels = torch.tensor(labels)
        model_inputs.append([input_ids, token_type_ids, counter_ids, counter_type_ids, labels, ins])
    return model_inputs


def load_dev_test(file_path, tokenizer, task: str = "cont"):
    model_inputs = []
    for ins in json.load(open(file_path)):
        for key in "ab":
            input_ids = torch.tensor([x for r in tokenizer(ins[f"entity_{key}_reviews"]).input_ids for x in r])
            token_type_ids = torch.zeros_like(input_ids)
            c_key = "b" if key == "a" else "a"
            counter_ids = torch.tensor([x for r in tokenizer(ins[f"entity_{c_key}_reviews"]).input_ids for x in r])
            counter_type_ids = torch.ones_like(counter_ids)
            task_key = f"entity_{key}_summary" if task == "cont" else "common_summary"
            labels = torch.tensor([0])
            model_inputs.append([input_ids, token_type_ids, counter_ids, counter_type_ids, labels, ins[task_key]])
    return model_inputs


class CoCoTripModule(pl.LightningDataModule):
    def __init__(self, train_path: str, tokenizer, task: str = "cont", **kwargs):
        super().__init__()
        self.train_path = Path(train_path)
        self.data_dir = self.train_path.parent
        self.tokenizer = tokenizer
        self.task = task

    def setup(self, stage: Optional[str] = None) -> None:
        train = load_train(self.train_path, self.tokenizer)
        val = load_dev_test(self.data_dir / "dev.json", self.tokenizer, task=self.task)
        test = load_dev_test(self.data_dir / "test.json", self.tokenizer, task=self.task)
        self.data = {"train": train, "val": val, "test": test}

    def _get_dataloader(self, dataset_split, is_train: bool = False) -> DataLoader:
        dataset = CoCoTrip(dataset_split, pad_token_id=self.tokenizer.pad_token_id)
        return DataLoader(dataset, collate_fn=dataset.collate_fn, shuffle=is_train)

    def train_dataloader(self):
        return self._get_dataloader(self.data["train"], is_train=True)

    def val_dataloader(self):
        return self._get_dataloader(self.data["val"])

    def test_dataloader(self):
        return self._get_dataloader(self.data["test"])


class Summarizer(pl.LightningModule):
    def __init__(self,
                 model_name: str = "allenai/led-base-16384",
                 max_output_len: int = 256,
                 lr: float = 2e-5,
                 weight_decay: float = 0.001,
                 max_steps: int = 200000,
                 warmup: int = 5000,
                 epsilon: float = 0.1,
                 use_pair: bool = False,
                 default_root_dir: str = "tmp",
                 **kwargs):
        super().__init__()
        self.tokenizer = LEDTokenizerFast.from_pretrained(model_name)
        self.model = LEDForConditionalGeneration.from_pretrained(model_name)
        self.label_smoother = LabelSmoother(epsilon=epsilon) if epsilon > 0. else None

        self.rouge = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True,
                                 stemming=True, ensure_compatibility=True)

        self.max_output_len = max_output_len
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.warmup = warmup
        self.use_pair = use_pair
        self.default_root_dir = Path(default_root_dir) if default_root_dir is not None else None
        self.save_hyperparameters()

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor = None,
                output_ids: torch.Tensor = None):
        return self.model(input_ids,
                          token_type_ids=token_type_ids,
                          attention_mask=(input_ids != self.tokenizer.pad_token_id),
                          global_attention_mask=self._set_global_attention_mask(input_ids),
                          labels=output_ids, use_cache=False)

    def encode(self,
               input_ids: torch.Tensor,
               token_type_ids: torch.Tensor,
               counter_ids: torch.Tensor,
               counter_type_ids: torch.Tensor,
               expanded_return_idx: torch.Tensor):
        if self.use_pair:
            input_ids = torch.cat((input_ids, counter_ids), dim=-1)
            token_type_ids = torch.cat((token_type_ids, counter_type_ids), dim=-1)
        input_ids, token_type_ids = input_ids.to(self.device), token_type_ids.to(self.device)
        encoder = self.model.get_encoder()
        encoder_outputs = encoder.forward(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=(input_ids != self.tokenizer.pad_token_id),
            global_attention_mask=self._set_global_attention_mask(input_ids))
        last_hidden_state = encoder_outputs.last_hidden_state.index_select(0, expanded_return_idx.to(self.device))
        encoder_outputs["last_hidden_state"] = last_hidden_state
        return {"encoder_outputs": encoder_outputs}

    def partial_forward(self,
                        decoder_input_ids: torch.Tensor,
                        past=None,
                        encoder_outputs=None):
        model_inputs = self.model.prepare_inputs_for_generation(
            decoder_input_ids, past=past, use_cache=True, encoder_outputs=encoder_outputs)
        outputs = self.model(**model_inputs, return_dict=True)
        return outputs

    def post_process(self,
                     outputs_list,
                     model_kwargs_list,
                     beam_idx):
        new_model_kwargs_list = []
        for output, model_kwargs in zip(outputs_list, model_kwargs_list):
            new_model_kwargs_list.append(self.model._update_model_kwargs_for_generation(output, model_kwargs))
        for model_kwargs in new_model_kwargs_list:
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self.model._reorder_cache(model_kwargs["past"], beam_idx)
        return new_model_kwargs_list

    def _set_global_attention_mask(self, input_ids):
        """Configure the global attention pattern based on the task"""
        # Local attention everywhere - no global attention
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)

        # Global attention on periods
        global_attention_mask[input_ids == self.tokenizer.bos_token_id] = 1
        return global_attention_mask

    def training_step(self, batch, batch_nb):
        input_ids, token_type_ids, counter_ids, counter_type_ids, output_ids, entry = batch
        if self.use_pair:
            input_ids = torch.cat((input_ids, counter_ids), dim=-1)
            token_type_ids = torch.cat((token_type_ids, counter_type_ids), dim=-1)

        outputs = self.forward(input_ids=input_ids, token_type_ids=token_type_ids, output_ids=output_ids)
        if self.label_smoother is None:
            loss = outputs.loss
        else:
            loss = self.label_smoother(outputs, output_ids)
        self.log("loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup,
                                                    num_training_steps=self.max_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _evaluation_step(self, batch, batch_nb):
        predictions = self.predict_step(batch, batch_nb)
        return [{"prediction": pred, "reference": ref} for pred, ref in zip(predictions, batch[-1])]

    def validation_step(self, batch, batch_nb):
        return self._evaluation_step(batch, batch_nb)

    def test_step(self, batch, batch_nb):
        return self._evaluation_step(batch, batch_nb)

    def predict_step(self, batch, batch_nb, dataloader_idx: int = None, no_repeat_ngram_size: int = 3):
        input_ids, token_type_ids, counter_ids, counter_type_ids, _, entry = batch
        if self.use_pair:
            input_ids = torch.cat((input_ids, counter_ids), dim=-1)
            token_type_ids = torch.cat((token_type_ids, counter_type_ids), dim=-1)
        generated_ids = self.model.generate(input_ids,
                                            token_type_ids=token_type_ids,
                                            attention_mask=(input_ids != self.tokenizer.pad_token_id),
                                            global_attention_mask=self._set_global_attention_mask(input_ids),
                                            decoder_start_token_id=self.tokenizer.bos_token_id,  # def
                                            use_cache=True, max_length=self.max_output_len, num_beams=4,
                                            no_repeat_ngram_size=no_repeat_ngram_size)
        predictions = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        return predictions

    def _evaluation_epoch_end(self, split, outputs):
        hyp = [o["prediction"] for outs in outputs for o in outs]
        ref = [o["reference"] for outs in outputs for o in outs]

        scores = {}
        results = self.rouge.get_scores(hyp, ref)
        for metric_name in ("rouge-1", "rouge-2", "rouge-l"):
            for key in "fpr":
                val = results[metric_name][key]
                name = f"{metric_name}{key}"
                self.log(f"{split}_" + name, val, on_epoch=True, prog_bar=key == "f")
                scores[name] = val
        for key in "fpr":
            val = sum(results[metric_name][key] for metric_name in ("rouge-1", "rouge-2", "rouge-l"))
            self.log(f"{split}_rouge-12l{key}", val, on_epoch=True)

    def validation_epoch_end(self, outputs):
        return self._evaluation_epoch_end("val", outputs)

    def test_epoch_end(self, outputs):
        return self._evaluation_epoch_end("test", outputs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Summarizer")
        parser.add_argument("train_path", type=str, )
        parser.add_argument("--model_name", type=str, default="allenai/led-base-16384")
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--weight_decay", type=float, default=0.001)
        parser.add_argument("--warmup", type=int, default=1000)
        parser.add_argument("--epsilon", type=float, default=0.1)
        parser.add_argument("--use_pair", action="store_true")
        return parent_parser
