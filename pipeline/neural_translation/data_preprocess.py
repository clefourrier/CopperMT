#!/bin/python3
import subprocess
import sys, os


def copy_dicts(langs_list, langs_enc_list, langs_dec_list, raw_data_dir, data_dir, data_bin):
    try:
        os.makedirs(f"{data_dir}/{data_bin}")
    except FileExistsError:
        print("File already exists")
    for langs in langs_list + langs_enc_list + langs_dec_list:
        tail = subprocess.Popen(["tail", "-n", "+4", f"{raw_data_dir}/m_{langs}.vocab"], 
            stdout=subprocess.PIPE)
        cut = subprocess.Popen(["cut", "-f1"], 
            stdin=tail.stdout, 
            stdout=subprocess.PIPE)
        subprocess.Popen(["sed", "s/$/ 100/g"], 
            stdin=cut.stdout, 
            stdout=open(f"{data_dir}/{data_bin}/mf_{langs}.vocab", "w")).wait()
        tail.wait()
        cut.wait()


def preprocess(langs_enc_list, langs_dec_list, all_l_pairs, split_data_dir, data_dir, data_bin):
    for l_pair in all_l_pairs:
        l_in, l_out = l_pair.split("-")

        dict_path = {"in_dict": "", "out_dict": ""}
        for l_cur, encdec_list, name, cur_dict_name in [
            (l_in, langs_enc_list, "encoder", "in_dict"),
            (l_out, langs_dec_list, "decoder", "out_dict")
        ]:
            # Encoder (possibly shared) language
            encdec_langs = [l_group for l_group in encdec_list if l_cur in l_group]
            if len(encdec_langs) == 1:
                dict_path[cur_dict_name] = f"{data_dir}/{data_bin}/mf_{encdec_langs[0]}.vocab"
            elif len(encdec_langs) == 0:
                dict_path[cur_dict_name] = f"{data_dir}/{data_bin}/mf_{l_cur}.vocab"
            else:
                raise Exception(f"One language is present in several shared {name} language groups : {';'.join(encdec_langs)}")

        print(l_in, dict_path["in_dict"], l_out, dict_path["out_dict"])
        subprocess.Popen(["fairseq-preprocess", f"--source-lang={l_in}", f"--target-lang={l_out}", 
                         f"--trainpref={split_data_dir}/train.{l_in}-{l_out}",
                         f"--validpref={split_data_dir}/valid.{l_in}-{l_out}",
                         #f"--testpref={split_data_dir}/test.{l_in}-{l_out}",
                         f"--destdir={data_dir}/{data_bin}", f"--task=multilingual_translation", 
                         f"--srcdict={dict_path['in_dict']}", 
                         f"--tgtdict={dict_path['out_dict']}"]
                         ).wait()
        if l_in != l_out:
            subprocess.Popen(["fairseq-preprocess", f"--source-lang={l_in}", f"--target-lang={l_out}", 
                             f"--testpref={split_data_dir}/test.{l_in}-{l_out}",
                             f"--destdir={data_dir}/{data_bin}", f"--task=multilingual_translation", 
                             f"--srcdict={dict_path['in_dict']}", 
                             f"--tgtdict={dict_path['out_dict']}",
                             "--only-source"]
                             ).wait()

if __name__ == "__main__":
    args_dict=dict(arg.split("=") for arg in sys.argv[1:])

    # Group in clean args
    langs_list = (args_dict["-l"] if "-l" in args_dict.keys() else args_dict["--langs-list"]).split(",")
    all_l_pairs = (args_dict["-lp"] if "-lp" in args_dict.keys() else args_dict["--all-langs-pair"]).split(",")
    data_dir = args_dict["-d"] if "-d" in args_dict.keys() else args_dict["--data-dir"] 
    split_data_dir = args_dict["-o"] if "-o" in args_dict.keys() else args_dict["--orig-data-dir"] 
    raw_data_dir = args_dict["-r"] if "-r" in args_dict.keys() else args_dict["--raw-data-dir"] 
    try:
        langs_enc_list = args_dict["-le"] if "-le" in args_dict.keys() else args_dict["--langs-enc-list"] 
        langs_enc_list = langs_enc_list.split(",")
    except KeyError:
        langs_enc_list = []

    try:
        langs_dec_list = args_dict["-ld"] if "-ld" in args_dict.keys() else args_dict["--langs-dec-list"] 
        langs_dec_list = langs_dec_list.split(",")
    except KeyError:
        langs_dec_list = []

    try:
        finetuning = bool(args_dict["-f"] if "-f" in args_dict.keys() else args_dict["--finetuning"])
    except KeyError:
        finetuning = False
    data_bin="data-bin-finetune" if finetuning else "data-bin"
    
    copy_dicts(langs_list, langs_enc_list, langs_dec_list, 
        raw_data_dir, data_dir, data_bin)
    preprocess(langs_enc_list, langs_dec_list, all_l_pairs, 
        split_data_dir, data_dir, data_bin)
    