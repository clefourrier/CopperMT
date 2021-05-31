def get_indices_of_duplicate_items_after_first(seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs[1:]


def remove_duplicates_in_list(data_list):
    duplicates = get_duplicates(data_list)

    indices_to_remove = []
    for dup in duplicates:
        indices_to_remove.extend(
            get_indices_of_duplicate_items_after_first(data_list, dup))

    remove_indices(data_list, indices_to_remove)
    return indices_to_remove


def get_duplicates(list_containing_duplicates):
    seen = set()
    duplicates = []
    for item in list_containing_duplicates:
        if item[0] not in seen:  # item is a list of one phonetized word
            seen.add(item[0])
        else:
            duplicates.append(item)
    return duplicates


def remove_subset(data_list, subset):
    indices_to_remove = []
    for x in subset.data:
        try:
            index = data_list.index(x)
        except ValueError:
            continue

        indices_to_remove.append(index)

    remove_indices(data_list, indices_to_remove)
    return indices_to_remove


def remove_indices(data_list, indices: list):
    # We want the indices to be in reversed order to pop values from the back of the list,
    # in order not to affect the remaining indices
    indices = list(set(indices))
    indices.sort(reverse=True)
    for index in indices:
        data_list.pop(index)


def get_all_phones_to_frequency(path: str, langs_groups: list, unique_lang: str = None):
    phones = {}
    count = 0
    for lang_group in langs_groups:
        if unique_lang:
            if unique_lang in lang_group:
                with open(os.path.join(path,
                                       f"{'_'.join(lang_group)}.ipa.{unique_lang}")
                          ) as file:
                    for line in file:
                        line = eval(line)
                        for char in line:
                            count += 1
                            try:
                                phones[char] += 1
                            except KeyError:
                                phones[char] = 1
        else:
            for lang in lang_group:
                with open(os.path.join(path,
                                       f"{'_'.join(lang_group)}.ipa.{lang}")
                          ) as file:
                    for line in file:
                        line = eval(line)
                        for char in line:
                            count += 1
                            try:
                                phones[char] += 1
                            except KeyError:
                                phones[char] = 1

    phones = {p: c/count for p, c in phones.items()}
    return phones


def get_indices_of_duplicate_items_after_first(seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs[1:]


def clean_phones(phones):
    # We remove extra characters
    phones = [item for item in phones
              if item not in ["'", "'", "‘", 'ˈ', "'", '', ' ~', '-', '~', '-',
                              '.',
                              "ˈ", '̩', '̝, ', " ", '', ' ', ' ', '̃', '̊ ',
                              'ˌ', '̊', '̝']]

    if phones[0] == " ":
        phones = phones[1:]

    # We identify the double consonnants
    while any(i == j for i, j in zip(phones, phones[1:])):  # if there is a double consonnant
        n_phones = []
        save_this_char = True
        for i, (char, char_next) in enumerate(zip(phones, phones[1:] + ["END"])):
            if save_this_char:
                n_phones.append(char)
            else:
                if n_phones[-1] != ":":
                    n_phones.append(":")
                save_this_char = True

            if char_next == char:  # indicates whether to save new char or not
                save_this_char = False

        phones = n_phones


    # We concatenate the long phones
    concat_dict = {':': ':', "ː": ':'}
    for concat_item, concat_value in concat_dict.items():
        while concat_item in phones:
            i = phones.index(concat_item)
            phones.pop(i)
            phones[i - 1] = phones[i - 1] + concat_value

    # We transform the small letters into big letters
    small_letters = {'ʲ': 'j', 'ʰ': 'h'}
    phones = [
        phone if phone not in small_letters.keys() else small_letters[phone] for
        phone in phones]

    return " ".join(phones)