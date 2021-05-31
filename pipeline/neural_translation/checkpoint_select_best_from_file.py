#!/usr/bin/python
import sys
import shutil
import pandas
import numpy as np
from collections import defaultdict


# Extracting best checkpoints
def best_by_ref_lang(scores, lang):
    # We extract only the name to language of interest's BLEU
    name_to_scores = {n: s[lang] for n,s in scores.items()}
    return max(name_to_scores, key=lambda key: name_to_scores[key])


def best_by_mean(scores):
    # We extract the name to mean BLEU
    name_to_scores = {n: np.mean(list(s.values())) for n,s in scores.items()}
    return max(name_to_scores, key=lambda key: name_to_scores[key])


def best_by_min(scores):
    # We extract the name to min BLEU
    name_to_scores = {n: np.min(list(s.values())) for n,s in scores.items()}
    return max(name_to_scores, key=lambda key: name_to_scores[key])


def best_by_all(scores):
    # Inversion of dict: we match lang to {cp: bleu}, then rank all cp for each lang
    lang_to_scores = defaultdict(dict)
    for name, v in scores.items():
        for lang, bleu in v.items():
            lang_to_scores[lang][name] = bleu
    ranked_by_lang = {l: sorted(s, key=lambda key: s[key]) for l, s in lang_to_scores.items()}

    # We compute the score of each cp, and pick the one with max
    cp_scored = {cp: sum([v.index(cp) for v in ranked_by_lang.values()]) for cp in scores.keys()}
    return max(cp_scored, key=lambda key: cp_scored[key])


# Application to current values
if __name__ == "__main__":
    work_dir = sys.argv[1]
    cp_dir = sys.argv[2]

    df = pandas.read_csv(f"{work_dir}/bleu/all_bleu.csv",
                         sep=" ", names=["name", "l_in", "l_out", "bleu"])
    df = df[~df["name"].isin(["checkpoint_best.pt", "checkpoint_last.pt"])]

    names = df["name"].unique().tolist()
    scores = {name:{} for name in names}

    langs = []
    for ix, row in df.iterrows():
        if row["name"] in ["checkpoint_best.pt", "checkpoint_last.pt"]:
            continue
        scores[row["name"]][f"{row.l_in}_{row.l_out}"] = row.bleu
        langs.append(f"{row.l_in}_{row.l_out}")

    langs = list(set(langs))

    print("Best by mean:", best_by_mean(scores))
    print("Best by min:", best_by_min(scores))
    print("Best by all:", best_by_all(scores))
    for lang in langs:
        print(f"Best by {lang}:", best_by_ref_lang(scores, lang))

    shutil.copyfile(f"{work_dir}/{cp_dir}/{best_by_mean(scores)}",
                    f"{work_dir}/{cp_dir}/checkpoint_best.pt")
