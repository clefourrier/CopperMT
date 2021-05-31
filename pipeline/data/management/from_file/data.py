import os
import itertools, warnings
import subprocess
from typing import List, Tuple
from .shuffling import ShuffleType, Shuffle
from .utils import remove_duplicates_in_list, remove_indices

class DatasetReader:
    """ Reads a datafile already split and with a moses format """

    def __init__(self, folder_path: str, langs: list,
                 name: str, to_phonetize: bool = True):
        """ From a path and arguments, transforms the file to readable content
        for the follow up

        :param path: file path
        """
        self.name = name
        self.path = folder_path  # For reference

        local_path = os.path.join(folder_path, "_".join(langs))
        self.data = self.__read_data_file(local_path, langs, to_phonetize)

    @staticmethod
    def __read_data_file(path: str, langs: list, to_phonetize: bool = False) -> dict:
        """Reads the data file and returns it as a dict {lang: word_list}
        :param path: file path

        We expect all line to be a tuple of tuple
        """
        result = {}
        length = []
        for lang in langs:
            with open(f"{path}.{lang}") as file:
                cur_result = []
                for i, line in enumerate(file):
                    if to_phonetize:  # must be phonetized
                        warnings.warn("\nThe phonetising process can be a bit slow.")
                        line_list = list(
                            subprocess.Popen(
                                ["espeak", "-v", lang, "--ipa", "-q", line], stdout=subprocess.PIPE
                            )
                            .communicate()[0]
                            .decode("utf8")
                            .strip()
                        )
                        line_list.insert(0, lang)
                        line_list.append("EOW")
                    else:  # is normal data
                        line_list = line.rstrip().split(" ")
                    cur_result.append(line_list)
                length.append(i + 1)
                result[lang] = cur_result

        if not all(x == length[0] for x in length):
            raise Exception(
                f"All files read should be of same length. Not the case " f"for: {','.join(langs)}",
                "USER_ERROR",
            )

        return result


class Dataset:
    def __init__(self, name, langs, data_dict, remove_duplicates):
        self.name = name
        self.langs = langs
        self.data = data_dict
        self.num_sentences = len(self.data[self.langs[0]])
        self.must_remove_duplicates = remove_duplicates

        if self.must_remove_duplicates:
            self.remove_duplicates()
        else:
            warnings.warn(f"Duplicates were not removed from the dataset {self.name}.")

    @classmethod
    def create_from_reader(cls, name: str, data_list: List[DatasetReader],
                           remove_duplicates: bool = True
                           ):
        common_langs = list(set.intersection(*[set(data.data.keys()) for data in data_list]))

        data = {}
        for lang in common_langs:
            data[lang] = [item for data in data_list for item in data.data[lang]]

        return cls(name, common_langs, data, remove_duplicates)

    def shuffle(self, seed: int, shuffle_type: ShuffleType = ShuffleType.PSEUDO_RANDOM,
                reference_lang: str = None):
        order_of_lists = []
        to_shuffle_together = []
        for lang, cur_data in self.data.items():
            order_of_lists.append(lang)
            to_shuffle_together.append(cur_data)

        if not reference_lang:
            reference_lang = order_of_lists[0]

        shuffled = Shuffle.shuffle(to_shuffle_together, shuffle_type,
                                   reference_lang, seed)

        for i, lang in enumerate(order_of_lists):
            self.data[lang] = shuffled[i]

    def split(self, pct: float) -> Tuple:
        """ Divides the dataset into two ** new ** Datasets,
        containing pct of the data and 1 - pct of the data, respectively

        :param pct: Percentage to divide the data
        :return: Two new datasets
        """
        cut_idx = int(pct * self.num_sentences)
        subdata_1 = {lang: cur_data[:cut_idx] for lang, cur_data in self.data.items()}
        subdata_2 = {lang: cur_data[cut_idx:] for lang, cur_data in self.data.items()}
        return (
            Dataset(self.name, self.langs, subdata_1, False),
            Dataset(self.name, self.langs, subdata_2, False),
        )

    def remove_duplicates(self):
        """ Removes all the entries of a given dataset from the current dataset

        :param test_set: Dataset containing the values to remove
        """
        # Creates the list of the languages that are common
        for lang in self.langs:
            indices_removed = remove_duplicates_in_list(self.data[lang])

            # We have to propagate the indices removal to all the other lists
            for other_lang in [x for x in self.langs if x != lang]:
                remove_indices(self.data[other_lang], indices_removed)

    def __getitem__(self, lang: str) -> list:
        """ Getter, returns the data of a given lang.
        Takes into source either the index of the lang or its name

        :param lang: str: name of the lang
        :return: A data object
        """
        if isinstance(lang, str):
            try:
                return self.data[lang]
            except KeyError:
                raise Exception(
                    f"The language you provided does not exist in this dataset."
                    f"\nCurrent languages are {str(self.langs)} and you provided {lang}",
                    "USER_ERROR",
                )
        else:
            raise Exception(
                f"Lang must be a language name (str). You provided an object of type {str(type(lang))}",
                "USER_ERROR",
            )

    def are_available_languages(self, lang: List[str]) -> bool:
        """ Checks if the language given as argument is included in the source languages """
        if isinstance(lang, str):
            lang = [lang]
        for cur_lang in lang:
            if cur_lang not in self.langs:
                return False
        return True

    def get_available_langs(self, langs: list = None) -> list:
        if not langs:
            return self.langs.copy()

        langs = [langs] if isinstance(langs, str) else langs
        return [lang for lang in langs if lang in self.langs]

    def save(self, l_in: str, l_out: str, file_path: str, file_name: str):
        x_file = open(os.path.join(file_path, f"{file_name}_{l_in}_{l_out}.{l_in}"), "w")
        y_file = open(os.path.join(file_path, f"{file_name}_{l_in}_{l_out}.{l_out}"), "w")

        if l_in in self.langs and l_out in self.langs:
            for i in range(len(self.data[l_in])):
                # For each, we remove sow and eow
                x_file.write(f"{' '.join(self.data[l_in][i])}\n")
                y_file.write(f"{' '.join(self.data[l_out][i])}\n")

        x_file.close()
        y_file.close()