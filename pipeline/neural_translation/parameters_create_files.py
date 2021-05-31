#!/usr/bin/python
import sys
import os

# Application to current values
name = sys.argv[1]
param_dir = sys.argv[2] #"/Users/cfourrie/documents/almanach/software/cognate_analysis/pipeline/neural_translation/parameters"
sys.path.append(param_dir)


def param_line(name_of_param, val_of_param, param_string=False):
    # TODO: use
    if param_string:
        return f"{name_of_param}='{val_of_param}'\n"
    return f"{name_of_param}={val_of_param}\n"


try:
    import reference_file as ref
except ModuleNotFoundError:
    if name == "emb_vs_hid":
        print("There is no reference module yet!")
        ref = None
    else:
        raise Exception("There is no reference module, but you need it for the experience you want!")


best_rnn_emb_dim = ref.best_rnn_emb_dim if hasattr(ref, 'best_rnn_emb_dim') else 0
best_rnn_hid_dim = ref.best_rnn_hid_dim if hasattr(ref, 'best_rnn_hid_dim') else 0
best_tra_emb_dim = ref.best_tra_emb_dim if hasattr(ref, 'best_tra_emb_dim') else 0
best_tra_hid_dim = ref.best_tra_hid_dim if hasattr(ref, 'best_tra_hid_dim') else 0
best_rnn_bs = ref.best_rnn_bs if hasattr(ref, 'best_rnn_bs') else 0
best_rnn_lr = ref.best_rnn_lr if hasattr(ref, 'best_rnn_lr') else 0
best_tra_bs = ref.best_tra_bs if hasattr(ref, 'best_tra_bs') else 0
best_tra_lr = ref.best_tra_lr if hasattr(ref, 'best_tra_lr') else 0
best_rnn_nb_layers = ref.best_rnn_nb_layers if hasattr(ref, 'best_rnn_nb_layers') else 0
best_tra_nb_layers = ref.best_tra_nb_layers if hasattr(ref, 'best_tra_nb_layers') else 0
best_attention = ref.best_attention if hasattr(ref, 'best_attention') else 0
best_heads = ref.best_heads if hasattr(ref, 'best_heads') else 0


if name == "emb_vs_hid":
    try:
        os.mkdir(os.path.join(param_dir, "emb_vs_hid"))
    except FileExistsError:
        print("Careful, you might overwrite stuff")
    for emb_dim in [8, 12, 16, 20, 24]:
        for hid_dim in [18, 36, 54, 72]:
            for model_type in ["bigru", "transformer"]:
                with open(os.path.join(param_dir, "emb_vs_hid", f"{model_type}_{emb_dim}_{hid_dim}"), "w") as f:
                    f.write(param_line("model_type", model_type, param_string=True))
                    if model_type == "bigru":
                        f.write(param_line("attention", "luong-dot", param_string=True))
                    f.write(param_line("enc_layer", 1))
                    f.write(param_line("enc_emb_dim", emb_dim))
                    f.write(param_line("enc_hid_dim", hid_dim))
                    if model_type == "transformer":
                        f.write(param_line("enc_heads", 1))
                    f.write(param_line("dec_layer", 1))
                    f.write(param_line("dec_emb_dim", emb_dim))
                    f.write(param_line("dec_hid_dim", hid_dim))
                    if model_type == "transformer":
                        f.write(param_line("dec_heads", 1))
                    f.write(param_line("batch_size", 64))
                    f.write(param_line("dropout", 0.2))
                    f.write(param_line("learning_rate", 0.005))
                    f.write(param_line("epoch", 20))
                    f.write(param_line("share_encoder", ''))
                    f.write(param_line("share_decoder", ''))


if name == "batch_vs_lr":
    try:
        os.mkdir(os.path.join(param_dir, "batch_vs_lr"))
    except FileExistsError:
        print("Careful, you might overwrite stuff")
    for batch_size in [10, 30, 65, 100]:
        for learning_rate in [0.001, 0.005, 0.01]:
            for model_type, best_emb_dim, best_hid_dim in [
                ("bigru", best_rnn_emb_dim, best_rnn_hid_dim),
                ("transformer", best_tra_emb_dim, best_tra_hid_dim)
            ]:
                with open(os.path.join(param_dir, "batch_vs_lr", f"{model_type}_{batch_size}_{learning_rate}"), "w") as f:
                    f.write(param_line("model_type", model_type, param_string=True))
                    if model_type == "bigru":
                        f.write(param_line("attention", "luong-dot", param_string=True))
                    f.write(param_line("enc_layer", 1))
                    f.write(param_line("enc_emb_dim", best_emb_dim))
                    f.write(param_line("enc_hid_dim", best_hid_dim))
                    if model_type == "transformer":
                        f.write(param_line("enc_heads", 1))
                    f.write(param_line("dec_layer", 1))
                    f.write(param_line("dec_emb_dim", best_emb_dim))
                    f.write(param_line("dec_hid_dim", best_hid_dim))
                    if model_type == "transformer":
                        f.write(param_line("dec_heads", 1))
                    f.write(param_line("batch_size", batch_size))
                    f.write(param_line("dropout", 0.2))
                    f.write(param_line("learning_rate", learning_rate))
                    f.write(param_line("epoch", 20))
                    f.write(param_line("share_encoder", ''))
                    f.write(param_line("share_decoder", ''))


if name == "nb_layers":
    try:
        os.mkdir(os.path.join(param_dir, "nb_layers"))
    except FileExistsError:
        print("Careful, you might overwrite stuff")
    for nb_layers in [1, 2, 4]:
        for model_type, best_emb_dim, best_hid_dim, best_bs, best_lr in [
            ("bigru", best_rnn_emb_dim, best_rnn_hid_dim, best_rnn_bs, best_rnn_lr),
            ("transformer", best_tra_emb_dim, best_tra_hid_dim, best_tra_bs, best_tra_lr)
        ]:
            with open(os.path.join(param_dir, "nb_layers", f"{model_type}_{nb_layers}"), "w") as f:
                f.write(param_line("model_type", model_type, param_string=True))
                if model_type == "bigru":
                    f.write(param_line("attention", "luong-dot", param_string=True))
                f.write(param_line("enc_layer", nb_layers))
                f.write(param_line("enc_emb_dim", best_emb_dim))
                f.write(param_line("enc_hid_dim", best_hid_dim))
                if model_type == "transformer":
                    f.write(param_line("enc_heads", 1))
                f.write(param_line("dec_layer", nb_layers))
                f.write(param_line("dec_emb_dim", best_emb_dim))
                f.write(param_line("dec_hid_dim", best_hid_dim))
                if model_type == "transformer":
                    f.write(param_line("dec_heads", 1))
                f.write(param_line("batch_size", best_bs))
                f.write(param_line("dropout", 0.2))
                f.write(param_line("learning_rate", best_lr))
                f.write(param_line("epoch", 20))
                f.write(param_line("share_encoder", ''))
                f.write(param_line("share_decoder", ''))


if name == "compare_attentions":
    try:
        os.mkdir(os.path.join(param_dir, "compare_attentions"))
    except FileExistsError:
        print("Careful, you might overwrite stuff")
    for attention in ["none", "luong-dot", "luong-general",
                     "luong-concat", "bahdanau-dot", "bahdanau-concat",
                     "bahdanau-general", "bahdanau"]:
        with open(os.path.join(param_dir, "compare_attentions",
                               f"bigru_{attention}"),
                  "w") as f:
            f.write(param_line("model_type", 'bigru', param_string=True))
            f.write(param_line("attention", attention, param_string=True))
            f.write(param_line("enc_layer", best_rnn_nb_layers))
            f.write(param_line("enc_emb_dim", best_rnn_emb_dim))
            f.write(param_line("enc_hid_dim", best_rnn_hid_dim))
            f.write(param_line("dec_layer", best_rnn_nb_layers))
            f.write(param_line("dec_emb_dim", best_rnn_emb_dim))
            f.write(param_line("dec_hid_dim", best_rnn_hid_dim))
            f.write(param_line("batch_size", best_rnn_bs))
            f.write(param_line("dropout", 0.2))
            f.write(param_line("learning_rate", best_rnn_lr))
            f.write(param_line("epoch", 20))
            f.write(param_line("share_encoder", ''))
            f.write(param_line("share_decoder", ''))


if name == "compare_heads":
    try:
        os.mkdir(os.path.join(param_dir, "compare_heads"))
    except FileExistsError:
        print("Careful, you might overwrite stuff")
    for nb_heads in [1, 2, 3, 4]:
        with open(os.path.join(param_dir, "compare_heads",
                               f"transformer_{nb_heads}"),
                  "w") as f:
            f.write(param_line("model_type", 'transformer', param_string=True))
            f.write(param_line(f"enc_layer", best_tra_nb_layers))
            f.write(param_line(f"enc_emb_dim", best_tra_emb_dim))
            f.write(param_line(f"enc_hid_dim", best_tra_hid_dim))
            f.write(param_line(f"enc_heads", nb_heads))
            f.write(param_line(f"dec_layer", best_tra_nb_layers))
            f.write(param_line(f"dec_emb_dim", best_tra_emb_dim))
            f.write(param_line(f"dec_hid_dim", best_tra_hid_dim))
            f.write(param_line(f"dec_heads", nb_heads))
            f.write(param_line(f"batch_size", best_tra_bs))
            f.write(param_line("dropout", 0.2))
            f.write(param_line("learning_rate", best_tra_lr))
            f.write(param_line("epoch", 20))
            f.write(param_line("share_encoder", ''))
            f.write(param_line("share_decoder", ''))

