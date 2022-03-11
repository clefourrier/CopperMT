#!/bin/python3
import subprocess
import sys


def train_segm(langs_list, langs_enc_list, langs_dec_list, all_l_pairs, vocab_size, model_type, data_dir):
	for langs in set(langs_list + langs_enc_list + langs_dec_list):
		langs = langs.split("-")
		input_data = ""
		for lang in langs:
			for l_pair in all_l_pairs:
				if lang in l_pair.split("-"):
					input_data += f"{data_dir}/train.{l_pair}.{lang},"
		input_data = input_data[:-1]

		# We store the long and special chars here
		chars = [] 
		for file in input_data.split(","):
			with open(file, "r") as f:
				for line in f:
					chars.extend(line.replace("\n", "").split(" "))
		chars = list(set(chars))

		subprocess.run(
			["spm_train", f"--input={input_data}", f"--model_prefix={data_dir}/m_{'-'.join(langs)}",
			f"--vocab_size={vocab_size}", "--character_coverage=1.0", f"--model_type={model_type}",
			f"--user_defined_symbols={','.join(chars)}",
			"--add_dummy_prefix=false"
			]
		)
		print(["spm_train", f"--input={input_data}", f"--model_prefix={data_dir}/m_{'-'.join(langs)}",
			f"--vocab_size={vocab_size}", "--character_coverage=1.0", f"--model_type={model_type}",
			f"--user_defined_symbols={','.join(chars)}",
			"--add_dummy_prefix=false"
			])

if __name__ == "__main__":
	args_dict=dict(arg.split("=") for arg in sys.argv[1:])

	# Group in clean args
	langs_list = (args_dict["-l"] if "-l" in args_dict.keys() else args_dict["--langs-list"]).split(",")
	all_l_pairs = args_dict["-lp"] if "-lp" in args_dict.keys() else args_dict["--all-langs-pair"] 
	all_l_pairs = all_l_pairs.split(",")
	vocab_size = args_dict["-v"] if "-v" in args_dict.keys() else args_dict["--vocab-size"] 
	model_type = args_dict["-t"] if "-t" in args_dict else args_dict["--model-type"] 
	data_dir = args_dict["-d"] if "-d" in args_dict else args_dict["--data-dir"] 
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


	train_segm(langs_list, langs_enc_list, langs_dec_list, 
		all_l_pairs, vocab_size, model_type, data_dir)
