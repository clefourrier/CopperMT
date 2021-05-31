import random
from enum import Enum


class ShuffleType(Enum):
    """
    Types of shuffling possible for the data set
    """

    NONE = 0
    ASCENDING = 1
    DESCENDING = 2
    PSEUDO_RANDOM = 3


class Shuffle:
    @staticmethod
    def shuffle(
        list_of_data: list, shuffle_type: ShuffleType,
        ix_lang_of_reference_for_shuffling: int, seed: int = 0
    ):
        random.seed(seed)

        data = []

        shuffle_type = shuffle_type
        # Shuffling
        if shuffle_type == ShuffleType.ASCENDING:
            data = Shuffle.__by_ascending_size(
                list_of_data, ix_lang_of_reference_for_shuffling
            )  # lang we sort accorded to
        elif shuffle_type == ShuffleType.DESCENDING:
            data = Shuffle.__by_descending_size(
                list_of_data, ix_lang_of_reference_for_shuffling
            )
        elif shuffle_type == ShuffleType.PSEUDO_RANDOM:
            data = Shuffle.__pseudo_randomly(list_of_data)
        elif shuffle_type == ShuffleType.NONE:
            data = list_of_data

        return data

    @staticmethod
    def __by_ascending_size(list_of_data: list, index_ref_lang: int):
        """ Creates batches where the data is ordered from the shortest sentence to the longest """
        list_tuples = zip(*sorted(zip(*list_of_data), key=lambda x: len(x[index_ref_lang])))
        return [list(elem) for elem in list_tuples]

    @staticmethod
    def __by_descending_size(list_of_data: list, index_ref_lang: int):
        """ Creates batches where the data is ordered from the longest sentence to the shortest """
        list_tuples = zip(
            *sorted(zip(*list_of_data), key=lambda x: len(x[index_ref_lang]), reverse=True)
        )
        return [list(elem) for elem in list_tuples]

    @staticmethod
    def __pseudo_randomly(list_of_data: list):
        """ Creates batches where the data is shuffled randomly """
        data = list(zip(*list_of_data))
        random.shuffle(data)
        return [list(elem) for elem in zip(*data)]
