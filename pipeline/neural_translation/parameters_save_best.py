#!/usr/bin/python
import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from checkpoint_select_best_from_file import best_by_min, best_by_all, best_by_mean, best_by_ref_lang


def get_all_params_dict(dir, model_type, langs, seeds):
    model_names = [name for name in os.listdir(dir)
                   if name.split("_")[0] == model_type]

    scores = defaultdict(lambda : {lang: [] for lang in langs}) #{model_name:  for model_name in model_names}
    for model_name in model_names:
        for seed in seeds:
            try:
                df = pd.read_csv(
                    os.path.join(dir, model_name, seed, "bleu", "best_bleu_synthesis.csv"),
                    sep=" ", index_col=False,
                    names=["l_in", "l_out", "bleu", "extended_bleu"]
                )
                for ix, row in df.iterrows():
                    if f"{row.l_in}-{row.l_out}" in langs:
                        scores[model_name][f"{row.l_in}-{row.l_out}"].append(row.bleu)
            except FileNotFoundError as e:
                print(e)

    scores_mean = {model: {lang: np.mean(v) for lang, v in lang_v.items()} for model, lang_v in scores.items()}
    scores_std = {model: {lang: np.std(v) for lang, v in lang_v.items()} for model, lang_v in scores.items()}

    return scores_mean, scores_std


def get_best_param(dir, model_type, num_params, langs, seeds):
    scores_mean, scores_std = get_all_params_dict(dir, model_type, langs, seeds)

    best_all = best_by_all(scores_mean)
    while any([x > 4 for x in scores_std[best_all].values()]):
        del scores_mean[best_all]
        best_all = best_by_all(scores_mean)

    # TODO: simplify using unpacking and starred expression
    # Best all returns name_param1_param.._paramN
    if num_params > 1:
        return [best_all.split("_")[i + 1] for i in range(num_params)]
    else:
        return best_all.split("_")[1]


def f_parameter_line(name_of_param, val_of_param, is_string=False):
    if is_string:
        return f"{name_of_param} = '{val_of_param}'\n"
    return f"{name_of_param} = {val_of_param}\n"


if __name__ == "__main__":
    # Application to current values
    name = sys.argv[1]
    param_dir = sys.argv[2]
    work_dir = sys.argv[3]
    seeds = sys.argv[4]
    seeds = list(seeds.split(" "))
    langs = sys.argv[5]
    langs = list(langs.split(","))

    with open(os.path.join(param_dir, "reference_file.py"), "a+") as f:
        if name == "emb_vs_hid":
            best_rnn_emb_dim, best_rnn_hid_dim = get_best_param(work_dir, "bigru", 2, langs, seeds)
            f.write(f_parameter_line("best_rnn_emb_dim", best_rnn_emb_dim))
            f.write(f_parameter_line("best_rnn_hid_dim", best_rnn_hid_dim))
            best_tra_emb_dim, best_tra_hid_dim = get_best_param(work_dir, "transformer", 2, langs, seeds)
            f.write(f_parameter_line("best_tra_emb_dim", best_tra_emb_dim))
            f.write(f_parameter_line("best_tra_hid_dim", best_tra_hid_dim))

        elif name == "batch_vs_lr":
            best_rnn_bs, best_rnn_lr = get_best_param(work_dir, "bigru", 2, langs, seeds)
            f.write(f_parameter_line("best_rnn_bs", best_rnn_bs))
            f.write(f_parameter_line("best_rnn_lr", best_rnn_lr))
            best_tra_bs, best_tra_lr = get_best_param(work_dir, "transformer", 2, langs, seeds)
            f.write(f_parameter_line("best_tra_bs", best_tra_bs))
            f.write(f_parameter_line("best_tra_lr", best_tra_lr))

        elif name == "nb_layers":
            best_rnn_nb_layers = get_best_param(work_dir, "bigru", 1, langs, seeds)
            f.write(f_parameter_line("best_rnn_nb_layers", best_rnn_nb_layers))
            best_tra_nb_layers = get_best_param(work_dir, "transformer", 1, langs, seeds)
            f.write(f_parameter_line("best_tra_nb_layers", best_tra_nb_layers))

        elif name == "compare_attentions":
            best_attention = get_best_param(work_dir, "bigru", 1, langs, seeds)
            f.write(f_parameter_line("best_attention", best_attention, is_string=True))

        elif name == "compare_heads":
            best_heads = get_best_param(work_dir, "transformer", 1, langs, seeds)
            f.write(f_parameter_line("best_heads", best_heads))

        else:
            raise Exception("Experience name does not exist")