import os, sys
from pathlib import Path
import pandas as pd
import subprocess
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from data.management.from_file.utils import clean_phones

path_to_yamtg = "/home/cfourrie/documents/software/YaMTG/YaMTG_2.1.csv" # Your path to YAMTG here 
out_path = "/home/cfourrie/documents/software/cognate-prediction/inputs/raw_data/romance_monolingual/" # Your output path here
langs_of_interest_orig = sorted(["lat", "spa", "ita"])
langs_of_interest = sorted(["la", "es", "it"])

extract = False
phonetize = True


if extract:
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass

    # Sources in the file next to this one
    df_yamtg = pd.read_csv(path_to_yamtg,
                            sep='\t',
                            names=["lang1", "lexeme1", "lang2", "lexeme2", "link"],
                            dtype={"lang1": str, "lexeme1": str,
                                   "lang2": str, "lexeme2": str, "link": str}).set_index("link")

    print("Data read")
    for lang in langs_of_interest_orig:
        cur_list = []
        for i in [1, 2]:
            cur_frame = df_yamtg[df_yamtg[f"lang{i}"] == lang]
            for index, row in cur_frame.iterrows():
                words = row[f"lexeme{i}"]
                if isinstance(words, str) and not any(x in words for x in [".", "(", ")"]):
                    words = words.split(" ")[0]
                    cur_list.append(words)
            print(lang, i, "list read")

        print(lang, "done")

        lang = {"lat": "la", "spa": "es", "ita": "it", "ron": "ro", "fra": "fr"}[lang]
        with open(os.path.join(out_path, f"orig{lang}.{lang}"), "w+") as file:
            for word in cur_list:
                file.write(word + ".\n")

## Then, for each generated file, manually apply espeak:
if phonetize:
    for l_in in langs_of_interest:
        # We keep only uniques
        subprocess.Popen(
            ["awk", "{!seen[$0]++};END{for(i in seen) print i}",
             os.path.join(out_path, f"orig{l_in}.{l_in}")
             ], stdout=open(os.path.join(out_path, f"orig_set_{l_in}.{l_in}"), "w+")
        ).wait()

        # We remove weird words (containing other than alphabetic and the end dot)
        subprocess.Popen(
            ["awk", '/^[a-zA-Z]+.$/',
             os.path.join(out_path, f"orig_set_{l_in}.{l_in}")
             ], stdout=open(os.path.join(out_path, f"orig_set_alphab_{l_in}.{l_in}"), "w+")
        ).wait()

        subprocess.Popen(
            ["espeak", "-v", l_in, "--ipa", "-q",
             "-f", os.path.join(out_path, f"orig_set_alphab_{l_in}.{l_in}")
             ], stdout=open(os.path.join(out_path, f"orig_set_alphab_espeak_{l_in}.{l_in}"), "w+")
        ).wait()

        with open(os.path.join(out_path, f"orig_set_alphab_espeak_{l_in}.{l_in}"), "r") as old, \
                open(os.path.join(out_path, f"{l_in}_{l_in}.{l_in}"), "w+") as new:
            for line in old:
                new.write(clean_phones(list(line)))
