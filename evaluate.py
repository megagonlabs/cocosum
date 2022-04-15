import json
from collections import defaultdict, Counter
from pathlib import Path

import bert_score
import fire
import pandas as pd
import rouge

pd.options.display.float_format = '{:,.2f}'.format

EVALUATOR = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True,
                        stemming=True, ensure_compatibility=True)
MODEL_TYPE = "microsoft/deberta-xlarge-mnli"
SCORER = bert_score.BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)


def stem(x):
    return Counter(EVALUATOR.stem_tokens(EVALUATOR.tokenize_text(x.lower())))


def scoring(gold, cont, comm):
    scores = {}
    intra = defaultdict(list)
    numerator, denom = [], []

    hyp_cont, ref_cont = [], []
    hyp_comm, ref_comm = [], []
    for x in gold:
        ref_a, ref_b = tuple(x["entity_a_summary"]), tuple(x["entity_b_summary"])
        hyp_a, hyp_b = cont[ref_a], cont[ref_b]
        hyp_cont.extend([hyp_a, hyp_b]), ref_cont.extend([list(ref_a), list(ref_b)])
        ref_c = tuple(x["common_summary"])
        hyp_c = comm[ref_c]

        hyp_comm.append(hyp_c), ref_comm.append(list(ref_c))

        # Intra
        for metric, vs in EVALUATOR.get_scores(hyp_a, hyp_b).items():
            for k, v in vs.items():
                intra["_".join(("intra", metric, k))].append(v)

        # Distinct
        s_c = stem(hyp_c)
        s_a, s_b = stem(hyp_a), stem(hyp_b)

        numerator.append(sum((s_a & s_b).values()) + sum((s_a & s_c).values()) + sum((s_b & s_c).values()) - 2 * sum(
            (s_a & s_b & s_c).values()))
        denom.append(sum((s_a | s_b | s_c).values()))
    scores.update({k: sum(v) / len(v) for k, v in intra.items()})

    for metric, vs in EVALUATOR.get_scores(hyp_cont, ref_cont).items():
        for k, v in vs.items():
            scores["_".join(("cont", metric, k))] = v
    for metric, vs in EVALUATOR.get_scores(hyp_comm, ref_comm).items():
        for k, v in vs.items():
            scores["_".join(("comm", metric, k))] = v

    for key, val in zip("prf", SCORER.score(hyp_cont, ref_cont)):
        scores[f"cont_bert_score_{key}"] = val.mean().item()
    for key, val in zip("prf", SCORER.score(hyp_comm, ref_comm)):
        scores[f"comm_bert_score_{key}"] = val.mean().item()
    for key, val in zip("prf", SCORER.score(hyp_cont[::2], hyp_cont[1::2])):
        scores[f"intra_bert_score_{key}"] = val.mean().item()

    inter = defaultdict(list)
    for i, h in enumerate(hyp_comm):
        hyp_inter = [(h, r) for r in hyp_comm[:i] + hyp_comm[i + 1:]]
        for metric, vs in EVALUATOR.get_scores(*zip(*hyp_inter)).items():
            for k, v in vs.items():
                inter["_".join(("inter", metric, k))].append(v)
        for key, val in zip("prf", SCORER.score(*zip(*hyp_inter))):
            inter[f"inter_bert_score_{key}"].append(val.mean().item())
    scores.update({k: sum(v) / len(v) for k, v in inter.items()})
    scores["distinct"] = sum(1 - n / d for n, d in zip(numerator, denom)) / len(numerator)
    return scores


def run(data_dir, cont_path, comm_path, output_path=None):
    data_dir = Path(data_dir)
    cont = json.load(open(cont_path))
    comm = json.load(open(comm_path))

    cont = {k: {tuple(x["reference"]): x["prediction"] for x in v} for k, v in cont.items()}
    comm = {k: {tuple(x["reference"]): x["prediction"] for x in v} for k, v in comm.items()}
    scores = {}
    for split in ("dev", "test"):
        gold = json.load(open(data_dir / f"{split}.json"))
        scores[split] = scoring(gold, cont[split], comm[split])

    print(100 * pd.DataFrame(scores).sort_index())
    if output_path is not None:
        with open(output_path, "w") as file:
            print(json.dumps(scores), file=file)


if __name__ == '__main__':
    fire.Fire(run)
