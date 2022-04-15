import json
import math
from collections import defaultdict
from pathlib import Path

import fire
import torch
from huggingface_hub import hf_hub_url, cached_download
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.generation_utils import top_k_top_p_filtering, BeamSearchScorer

from model import Summarizer, load_dev_test
from reader import CoCoTrip

AVAILABLE_MODELS = {"megagonlabs/cocosum-cont-self",
                    "megagonlabs/cocosum-cont-few",
                    "megagonlabs/cocosum-comm-self",
                    "megagonlabs/cocosum-comm-few"}


class Generator:
    def __init__(self,
                 model_checkpoint: str,
                 counter_model_checkpoint: str = None,
                 alpha: float = 1.0,
                 top_p: float = 1.0,
                 do_moe: bool = False,
                 do_ens_tgt: bool = False,
                 do_ens_cnt: bool = False,
                 do_ens_tgt_moe: bool = False,
                 do_ens_cnt_moe: bool = False,
                 ens_method: str = "prop"):
        assert not (do_ens_tgt and do_ens_tgt_moe)
        assert not (do_ens_cnt and do_ens_cnt_moe)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Contrastive
        if model_checkpoint in AVAILABLE_MODELS:
            model_url = hf_hub_url(str(model_checkpoint), filename="best.model")
            ckpt_path = cached_download(url=model_url)
        else:
            ckpt_path = str(next(Path(model_checkpoint).glob("*.ckpt")))
        self.model = Summarizer.load_from_checkpoint(ckpt_path)
        self.model.to(self.device).eval()
        self.tokenizer = self.model.tokenizer
        if counter_model_checkpoint is not None:
            if model_checkpoint in AVAILABLE_MODELS:
                model_url = hf_hub_url(str(model_checkpoint), filename="best.model")
                ckpt_path = cached_download(url=model_url)
            else:
                ckpt_path = str(next(Path(counter_model_checkpoint).glob("*.ckpt")))
            self.c_model = Summarizer.load_from_checkpoint(ckpt_path)
            self.c_model.to(self.device).eval()
        else:
            self.c_model = None

        self.alpha = alpha
        self.top_p = top_p

        self.do_moe = do_moe
        self.do_ens_tgt = do_ens_tgt
        self.do_ens_cnt = do_ens_cnt
        self.do_ens_tgt_moe = do_ens_tgt_moe
        self.do_ens_cnt_moe = do_ens_cnt_moe
        self.ens_method = ens_method

    @torch.no_grad()
    def generate(self,
                 input_ids: torch.Tensor,
                 token_type_ids: torch.Tensor,
                 counter_ids: torch.tensor,
                 counter_type_ids: torch.Tensor,
                 max_output_len: int = 256,
                 min_output_len: int = 20,
                 beam_size: int = 4):
        assert len(input_ids) == 1, f"batch size must be 1 but {len(input_ids), input_ids.shape}"
        batch_size = 1
        bos_token_id = self.tokenizer.bos_token_id
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        counter_ids = counter_ids.to(self.device)
        counter_type_ids = counter_type_ids.to(self.device)

        decoder_input_ids = torch.tensor([[bos_token_id]] * input_ids.shape[0], device=self.device)
        expanded_return_idx = (
            torch.arange(decoder_input_ids.shape[0]).view(-1, 1).repeat(1, beam_size).view(-1).to(input_ids.device)
        )
        decoder_input_ids = decoder_input_ids.index_select(0, expanded_return_idx)

        # Encoding
        target_model_kwargs = [self.model.encode(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            counter_ids=counter_ids,
            counter_type_ids=counter_type_ids,
            expanded_return_idx=expanded_return_idx)]
        if self.do_ens_tgt or self.do_ens_tgt_moe:
            target_model_kwargs.append(self.model.encode(
                input_ids=counter_ids,
                token_type_ids=1 - counter_type_ids,  # FLIP type ids
                counter_ids=input_ids,
                counter_type_ids=1 - token_type_ids,
                expanded_return_idx=expanded_return_idx))

        if self.c_model:
            counter_model_kwargs = [self.c_model.encode(
                input_ids=counter_ids,
                token_type_ids=1 - counter_type_ids,  # FLIP type ids
                counter_ids=input_ids,
                counter_type_ids=1 - token_type_ids,
                expanded_return_idx=expanded_return_idx)]
            if self.do_ens_cnt or self.do_ens_cnt_moe:
                counter_model_kwargs.append(self.c_model.encode(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    counter_ids=counter_ids,
                    counter_type_ids=counter_type_ids,
                    expanded_return_idx=expanded_return_idx))
        else:
            counter_model_kwargs = []

        logits_processor = self.model.model._get_logits_processor(
            repetition_penalty=None,
            bad_words_ids=None,
            no_repeat_ngram_size=3,  # ngram block
            encoder_no_repeat_ngram_size=None,
            encoder_input_ids=None,
            min_length=min_output_len,
            max_length=max_output_len,
            eos_token_id=self.tokenizer.eos_token_id,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            prefix_allowed_tokens_fn=None,
            num_beams=beam_size,
            num_beam_groups=None,
            diversity_penalty=None,
            remove_invalid_values=None)

        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=beam_size,
            device=self.device)
        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * beam_size,))
        cur_len = decoder_input_ids.shape[-1]
        while True:
            target_outputs, target_scores = [], []
            for model_kwargs in target_model_kwargs:
                output = self.model.partial_forward(decoder_input_ids, **model_kwargs)
                target_outputs.append(output)
                target_scores.append(torch.log_softmax(output.logits[:, -1, :], dim=-1))
            if len(target_scores) > 1:
                if self.do_ens_tgt:  # poe
                    target_scores = torch.sum(torch.stack(target_scores, dim=0), dim=0) / len(target_scores)
                else:  # moe
                    target_scores = torch.logsumexp(torch.stack(target_scores, dim=0) - math.log(len(target_scores)),
                                                    dim=0)
            else:
                target_scores = target_scores[0]
            target_scores = logits_processor(decoder_input_ids, target_scores)

            counter_outputs, counter_scores = [], []
            for model_kwargs in counter_model_kwargs:
                output = self.c_model.partial_forward(decoder_input_ids, **model_kwargs)
                counter_outputs.append(output)
                counter_scores.append(torch.log_softmax(output.logits[:, -1, :], dim=-1))
            if len(counter_scores) > 1:
                if self.do_ens_cnt:  # poe
                    counter_scores = torch.sum(torch.stack(counter_scores, dim=0), dim=0) / len(counter_scores)
                else:  # moe
                    counter_scores = torch.logsumexp(torch.stack(counter_scores, dim=0) - math.log(len(target_scores)),
                                                     dim=0)
            elif len(counter_scores) == 1:
                counter_scores = counter_scores[0]
            else:
                counter_scores = None

            next_token_scores = self.aggregate(target_scores, counter_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, beam_size * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                decoder_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            decoder_input_ids = torch.cat([decoder_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            target_model_kwargs = self.model.post_process(target_outputs, target_model_kwargs, beam_idx)
            if len(counter_model_kwargs):
                counter_model_kwargs = self.c_model.post_process(counter_outputs, counter_model_kwargs, beam_idx)

            cur_len = cur_len + 1
            stopping_criteria = self.model.model._get_stopping_criteria(
                max_length=max_output_len, max_time=None, max_new_tokens=None, start_length=cur_len
            )
            if beam_scorer.is_done or stopping_criteria(decoder_input_ids, None):
                break
        sequence_outputs = beam_scorer.finalize(
            decoder_input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=max_output_len,
        )
        return self.tokenizer.batch_decode(sequence_outputs["sequences"], skip_special_tokens=True)

    def aggregate(self,
                  target_scores: torch.Tensor,
                  counter_scores: torch.Tensor = None):
        main_scores = top_k_top_p_filtering(target_scores, top_p=self.top_p)
        if self.do_moe:
            main_scores = torch.exp(main_scores)
            if counter_scores is not None:
                if self.ens_method == "prop":
                    main_scores = main_scores + self.alpha * (torch.exp(target_scores) / torch.exp(counter_scores))
                else:
                    main_scores = main_scores + self.alpha * torch.exp(counter_scores)
            main_scores = torch.log(main_scores)
        else:
            if counter_scores is not None:
                if self.ens_method == "prop":
                    main_scores = main_scores + self.alpha * (target_scores - counter_scores)
                else:
                    main_scores = main_scores + self.alpha * counter_scores
        logits = torch.log_softmax(main_scores, dim=-1)
        return logits


def run(data_dir: str,
        task: str,
        output_dir: str,
        model_checkpoint: str,
        counter_model_checkpoint: str = None,
        beam_size: int = 4,
        max_output_len: int = 128,
        alpha: float = 1.0,
        top_p: float = 1.0,
        do_moe: bool = False,
        do_ens_tgt: bool = False,
        do_ens_cnt: bool = False,
        do_ens_tgt_moe: bool = False,
        do_ens_cnt_moe: bool = False,
        ens_method: str = "prop"):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    g = Generator(model_checkpoint=model_checkpoint,
                  counter_model_checkpoint=counter_model_checkpoint,
                  alpha=alpha, top_p=top_p, do_moe=do_moe,
                  do_ens_tgt=do_ens_tgt, do_ens_cnt=do_ens_cnt,
                  do_ens_tgt_moe=do_ens_tgt_moe, do_ens_cnt_moe=do_ens_cnt_moe,
                  ens_method=ens_method)
    outputs = defaultdict(list)
    for split in ("dev", "test"):
        raw = load_dev_test(data_dir / f"{split}.json", g.tokenizer, task=task)
        data = CoCoTrip(raw)
        for batch in tqdm(DataLoader(data, batch_size=1, collate_fn=data.collate_fn), desc=split, ncols=80):
            input_ids, token_type_ids, counter_ids, counter_type_ids, output_ids, entry = batch
            decoded_outputs = g.generate(input_ids,
                                         token_type_ids,
                                         counter_ids=counter_ids,
                                         counter_type_ids=counter_type_ids,
                                         max_output_len=max_output_len,
                                         beam_size=beam_size)
            outputs[split].append([{"prediction": out, "reference": e} for out, e in zip(decoded_outputs, entry)])
    outputs = {key: [v for vs in val for v in vs] for key, val in outputs.items()}
    json.dump(outputs, open(output_dir / "outputs.json", "w"))


if __name__ == '__main__':
    fire.Fire(run)
