from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from utils.bleu import get_neural_bleu_predictions, \
    get_statistical_bleu_predictions


def extract_backtranslated_data(all_langs, path_origin_monolingual,
                 path_translated_monolingual, path_out_data, neural=True):
    # Back prediction is saved as a new dataset, data_gen
    for l_in, l_out in all_langs:
        if neural:
            sources, _, prediction = get_neural_bleu_predictions(
                path_translated_monolingual, l_in, l_out, 10)
        else:
            sources, _, prediction = get_statistical_bleu_predictions(
                path_origin_monolingual, path_translated_monolingual,
                l_in, l_out, 10, 10
            )

        sources = [" ".join(s) for s in sources]
        with open(f'{path_origin_monolingual}/{l_out}_{l_out}.{l_out}',
                  'r') as reference_file:
            comparision_data = ["".join(line.split("\n")[0]) for line in
                                reference_file]
        try:
            os.mkdir(f'{path_out_data}')
        except FileExistsError:
            pass

        comparision_data = ["".join(w[:-1]) for w in comparision_data]
        print("Checking values")
        with open(f'{path_out_data}/{l_out}_{l_in}.{l_out}', 'w+') as new_in_file:
            with open(f'{path_out_data}/{l_out}_{l_in}.{l_in}','w+') as new_out_file:
                for i, source in enumerate(sources):
                    if "<unk>" in source:
                        continue
                    for n in range(10):
                        pred = " ".join(prediction[n][i])
                        if pred in comparision_data:
                            new_in_file.write(pred + "\n")
                            new_out_file.write(source + "\n")
                            break
                    if i % 1000 == 0:
                        print(l_in, l_out, str(i))


# unused at the moment, equivalent to data_concatenate.sh
def extend_data(data, data_to_append, seeds):
    """ Extends an already shuffled data set data with data_to_append.
    Likely faster in pure shell. TODO
    """
    for seed in seeds:
        _, _, filenames = next(os.walk(f"{data}/{seed}"))

        # Append correct data for finetuning
        for filename in filenames:
            # Open in append mode
            with open(os.path.join(f"{data}/{seed}", filename), "a") as ref:
                with open(os.path.join(f'{data_to_append}/{seed}', filename), "r") as file_to_append:
                    ref.write(file_to_append.read())


if __name__ == "__main__":
    langs = sys.argv[1]
    langs = list([lg.split("-") for lg in langs.split(",")])
    path_origin_monolingual = sys.argv[2]
    path_translated_monolingual = sys.argv[3]
    path_out_data = sys.argv[4]
    try:
        neural = bool(int(sys.argv[5]))
    except Exception:
        neural = True

    extract_backtranslated_data(langs, path_origin_monolingual,
                                path_translated_monolingual,
                                path_out_data, neural)
