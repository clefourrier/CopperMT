import subprocess
import itertools
import re
import os
from copy import copy

from .edit_distance import keep_best_edit_distance
from .language_info import available_espeak_languages
from data.management.from_file.shuffling import ShuffleType, Shuffle
from data.management.from_file.utils import clean_phones


class Word:
    """ Contains a word: lang, word, phonetization"""
    def __init__(self, word, lang):
        self.lang = lang
        self.lex = word
        self.phon = self.phonetize_word()
        if "(" in self.phon:
            # Appears when a word with wrong chars has not been phonetised properly
            self.phon = ""
            self.lex = ""

    def phonetize_word(self):
        # We phonetize the word
        try:
            phones = list(subprocess.Popen(["espeak", "-v", self.lang, "--ipa", "-q", self.lex],
                                      stdout=subprocess.PIPE).communicate()[0].decode('utf8').strip())
            return clean_phones(phones)

        except Exception:
            if self.lex != ' ':
                return clean_phones(list(self.lex))
            return ""

    def __str__(self):
        return f"Lexeme: {self.lex}, Phones: {self.phon}\n"

    def __eq__(self, other):
        return self.lex == other.lex and self.lang == other.lang

    def __hash__(self):
        return hash((self.lex, self.lang))


class WordSet:
    """One cognate set id, several languages and their lexemes/forms"""
    def __init__(self, set_id):
        self.set_id = set_id
        self.languages = {}

    def add_word(self, language, values):
        # We add a language to the base if not present
        if language not in self.languages.keys():
            self.languages[language] = []

        # We remove useless information todo: to check
        # and save the list of values
        values = values.split("(")[0]  # The parenthesis indicates grammatical info, such as (plur) and so forth
        values = values.replace("~", ",")
        values = values.split(",")  # for key: value1, value2, value3, we keep all the values
        if not isinstance(values, list):
            values = [values]

        for i, v in enumerate(values):
            values[i] = Word(v, language)

        self.languages[language].extend(values)

    def remove_duplicate_words(self):
        for language, lexemes in self.languages.items():
            if isinstance(lexemes, list):
                lexemes_simplified = list(set(lexemes))
                self.languages[language] = lexemes_simplified

    def contains_lang(self, languages: list):
        return set(languages) <= set(list(self.languages.keys()))

    def get_tuples(self, languages: list):
        """ Returns a tuple"""
        word_lists_per_languages = [self.languages[language] for language in languages]
        # We want to combine all words-langs
        word_combinations = list(itertools.product(*word_lists_per_languages))

        result = []
        for word_combination in word_combinations:
            phone_list = [word.phon for word in word_combination]
            lex_list = [word.lex for word in word_combination]

            if len(phone_list) == len(lex_list) == len(languages):
                if all(len(word) > 0 for word in lex_list):
                    result.append([tuple(phone_list), tuple(lex_list)])

        return result

    def __str__(self):
        result = f"ID: {str(self.set_id)} \n"
        for language, values in self.languages.items():
            result += f"{str(language)}: {' '.join([str(value) for value in values])} \n"
        return result

    def __len__(self):
        return len(self.languages.keys())

    def __copy__(self):
        cur_copy = WordSet(copy(self.set_id))
        cur_copy.languages = copy(self.languages)
        return cur_copy


class WordSetCollection:
    """Stores a bunch of word sets"""
    # todo: add get subset, add shuffle, add duplicate management
    def __init__(self):
        self.list_ids = []
        self.cognate_sets = {}
        self.last_ix = 0

    def add_word_set(self, lang_to_words: dict):
        # We define a new id and a new word set to store these values
        self.last_ix += 1
        if self.last_ix not in self.list_ids:
            self.list_ids.append(self.last_ix)
            self.cognate_sets[self.last_ix] = WordSet(self.last_ix)

        for lang, words in lang_to_words.items():
            for word_item in words:
                # If the word is a split character, we skip it
                if word_item in [",", "", "-"]:
                    continue

                # We split the word on non alphanumeric characters but not on space
                word_list = re.split('[[^\s]\W]+', str(word_item))

                for word in word_list:
                    # Small check to make sure it is not a proper noun
                    if word[0].islower():
                        self.cognate_sets[self.last_ix].add_word(lang, word)

    def clean(self, monolingual_data_removed = True):
        """ Only keeps the word sets with more than one language ^^ """
        list_ids = []
        result = {}
        for cs_id, cs in self.cognate_sets.items():
            # We remove cognate sets with only one lang
            if monolingual_data_removed and len(cs) <= 1:
                continue
            list_ids.append(cs_id)
            cs.remove_duplicate_words()
            result[cs_id] = cs
        self.list_ids = list_ids
        self.cognate_sets = result

    def generate_tuples(self, langs):
        tuple_list = []

        for cs_id, cs in self.cognate_sets.items():
            if cs.contains_lang(langs):
                cur_tuple_list = cs.get_tuples(langs)
                tuple_list.extend(cur_tuple_list)

        # Remove duplicates in tuples
        tuple_list.sort()
        tuple_list = keep_best_edit_distance(tuple_list)

        print(len(tuple_list), "cognate couples in ", langs)
        return tuple_list

    def save(self, folder_path: str, langs: list):
        file_name = "_".join(langs)
        tuple_list = self.generate_tuples(langs)
        file_list = [open(os.path.join(folder_path, f"{file_name}.{lang}"), "w+") for lang in langs]
        file_list_orig = [open(os.path.join(folder_path, f"orig.{file_name}.{lang}"), "w+") for lang in langs]
        for tuple_group in tuple_list:
            for lang_ix, __ in enumerate(langs):
                file_list[lang_ix].write(str(tuple_group[0][lang_ix]) + '\n')
                file_list_orig[lang_ix].write(tuple_group[1][lang_ix] + '\n')

        for file in file_list + file_list_orig:
            file.close()

    def __str__(self):
        result = ""
        for cognate_set in self.cognate_sets.values():
            result += str(cognate_set) + "\n"
        return result

    def __len__(self):
        return len(self.list_ids)

    def __copy__(self):
        cur_copy = WordSetCollection()
        cur_copy.cognate_sets = copy(self.cognate_sets)
        cur_copy.last_ix = self.last_ix
        return cur_copy
