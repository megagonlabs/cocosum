import json
import os
import re
import string
import unicodedata
from multiprocessing import Pool, cpu_count
from pathlib import Path

import fire
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

PRINTABLE = set(string.printable)


def strip_text(s: str) -> str:
    # https://stackoverflow.com/a/518232/2809427
    # https://stackoverflow.com/a/8689826
    s = re.sub("!{2,}", "!", s)
    s = re.sub("\?{2,}", "?", s)
    s = re.sub(",{2,}", ".", s)
    s = re.sub("\.{2,}", ".", s)
    s = re.sub("-{2,}", "-", s)
    return re.sub(" +", " ", "".join(c for c in unicodedata.normalize("NFD", s)
                                     if unicodedata.category(c) != "Mn" and c in PRINTABLE))


def download(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    if not (data_dir / "json").exists():
        os.system(f"wget -P {data_dir} https://www.cs.virginia.edu/~hw5x/Data/LARA/TripAdvisor/TripAdvisorJson.tar.bz2")
        os.system(f"tar -xjvf {data_dir}/TripAdvisorJson.tar.bz2 -C {data_dir}")


def train_dev_test(data_dir):
    data_dir = Path(data_dir)
    anno = json.load(open(data_dir / "anno.json"))
    for split in anno:
        processed = []
        for ins in anno[split]:
            for key in "ab":
                data = json.load(open(data_dir / "json" / (ins[f"entity_{key}"] + ".json")))
                reviews, uid = [], []
                for x in data["Reviews"]:
                    if x["ReviewID"] in ins[f"entity_{key}_uid"]:
                        reviews.append(strip_text(x["Content"]))
                        uid.append(x["ReviewID"])
                uid, reviews = zip(*sorted(zip(uid, reviews), key=lambda z: -len(nltk.word_tokenize(z[1]))))
                ins[f"entity_{key}_reviews"] = list(reviews)
                ins[f"entity_{key}_uid"] = list(uid)
            processed.append(dict(ins))
        if split == "train":
            with open(data_dir / "few_cont.jsonl", "w") as file:
                for ins in processed:
                    for key in "ab":
                        for tgt in ins[f"entity_{key}_summary"]:
                            print(json.dumps({"src": ins[f"entity_{key}_reviews"], "tgt": tgt}), file=file)
            with open(data_dir / "few_comm.jsonl", "w") as file:
                for ins in processed:
                    for key in "ab":
                        for tgt in ins[f"common_summary"]:
                            print(json.dumps({"src": ins[f"entity_{key}_reviews"], "tgt": tgt}), file=file)
            with open(data_dir / "few_comm_pair.jsonl", "w") as file:
                for ins in processed:
                    for key in "ab":
                        c_key = "b" if key == "a" else "a"
                        for tgt in ins[f"common_summary"]:
                            print(json.dumps({"src": ins[f"entity_{key}_reviews"],
                                              "counter": ins[f"entity_{c_key}_reviews"],
                                              "tgt": tgt}), file=file)
        else:
            json.dump(processed, open(data_dir / f"{split}.json", "w"))


def pseudo_train(data_dir):
    data_dir = Path(data_dir)
    anno = json.load(open(data_dir / "anno.json"))
    entity_to_ignore = {x[f"entity_{k}"] for vs in anno.values() for x in vs for k in "ab"}  # Used in train/dev/test

    files = [fp for fp in (data_dir / "json").glob("*.json") if fp.stem not in entity_to_ignore]
    contrastive, common = [], []
    with Pool(cpu_count()) as p, tqdm(desc="Create", ncols=80, total=len(files)) as prog:
        for cont, comm in p.imap_unordered(build_pseudo, files):
            contrastive.extend(cont), common.extend(comm)
            prog.update()

    print("Contrastive: ", len(contrastive))
    print("Common: ", len(common))

    with open(data_dir / "train_cont_all.jsonl", "w") as file:
        contrastive = sorted(contrastive, key=lambda x: x["dist"])
        for ins in contrastive:
            print(json.dumps(ins), file=file)
    with open(data_dir / "train_comm_all.jsonl", "w") as file:
        common = sorted(common, key=lambda x: x["dist"])
        for ins in common:
            print(json.dumps(ins), file=file)


def build_pseudo(fp):
    data = json.load(open(fp))
    reviews, comm, cont = set(), set(), set()
    for ins in data["Reviews"]:
        if "showReview" in ins["Content"]:
            continue
        review = strip_text(ins["Content"])
        length = len(nltk.word_tokenize(review))
        if 50 <= length <= 100:
            reviews.add(review)
        elif 100 <= length <= 150:
            cont.add(review)  # Use for both input and output
        elif review[0].isupper() and 15 <= length <= 50:
            comm.add(review)
    if len(reviews | cont) < 9:
        return [], []
    tfidf = TfidfVectorizer()
    tfidf.fit(list(reviews | cont | comm))
    reviews = list(reviews | cont)
    cont, comm = [x for x in cont if x[0].isupper()], list(comm)
    src_vec = tfidf.transform(reviews)
    nn = NearestNeighbors(n_neighbors=9).fit(src_vec)  # 8 reviews + 1 summary

    contrastive = []
    if cont:
        dist, src_ind = nn.kneighbors(tfidf.transform(cont))  # dist: l2 dist of normalized vectors
        dist = dist[:, 1:].sum(axis=1)
        for i, indices in enumerate(src_ind):
            tgt = cont[i]
            src = [reviews[j] for j in indices]
            src = [x for x in src if x != tgt]
            contrastive.append({
                "src": sorted(src, key=lambda x: -len(nltk.word_tokenize(x)))[:8],
                "tgt": tgt, "dist": dist[i]})
    common = []
    if comm:
        dist, src_ind = nn.kneighbors(tfidf.transform(comm))  # dist: l2 dist of normalized vectors
        dist = dist[:, :8].sum(axis=1)  # Only top8
        for i, indices in enumerate(src_ind[:, :8]):
            common.append({
                "src": sorted([reviews[j] for j in indices if i != j], key=lambda x: -len(nltk.word_tokenize(x))),
                "tgt": comm[i], "dist": dist[i]})
    return contrastive, common


def pairing(data_dir):
    data_dir = Path(data_dir)
    data = list(map(json.loads, open(data_dir / "train_comm_all.jsonl")))
    tgt = [x["tgt"] for x in data]
    tgt_vec = TfidfVectorizer().fit_transform(tgt)
    sim = np.argsort((tgt_vec @ tgt_vec.T).toarray(), axis=1)
    with open(data_dir / "train_comm_pair.jsonl", "w") as file:
        for i, x in enumerate(sim[:, ::-1]):
            for j in range(len(x)):
                if i != x[j]:
                    ins = data[i]
                    ins["counter"] = data[x[j]]["src"]
                    break
            print(json.dumps(ins), file=file)


def run(data_dir: str = "./data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    download(data_dir)  # Download TripAdvisor reviews
    train_dev_test(data_dir)
    pseudo_train(data_dir)
    pairing(data_dir)


if __name__ == '__main__':
    fire.Fire(run)
