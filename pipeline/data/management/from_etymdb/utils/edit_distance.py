

def levenshtein(s, t):
    ''' From Wikipedia article; Iterative with two matrix rows. '''
    if s == t:
        return 0
    elif len(s) == 0:
        return len(t)
    elif len(t) == 0:
        return len(s)
    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)
    for i, __ in enumerate(v0):
        v0[i] = i
    for i, __ in enumerate(s):
        v1[0] = i + 1
        for j, __ in enumerate(t):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j, __ in enumerate(v0):
            v0[j] = v1[j]

    return v1[len(t)]


# Find a way to remove approximate duplicates too
def keep_best_edit_distance_on_key(list_tuples, sort_key):
    if len(list_tuples) == 0:
        return list_tuples

    end_list = []
    list_tuples.sort(key=lambda t: t[1][sort_key])  # We sort on full words (0 for source, 1 for ouput)

    best_pair = list_tuples[0]
    cur_key = best_pair[1][sort_key]
    best_levenshtein = levenshtein(best_pair[1][0], best_pair[1][1])

    for pair in list_tuples[1:]:
        key = pair[1][sort_key]
        value = pair[1][abs(1 - sort_key)]  # We look at the second pair (the original words)
        # We only keep the first apparition of a cognate tuple
        if key != cur_key:
            end_list.append(best_pair)  # We save the best pair of the previous pair set
            best_levenshtein = levenshtein(key, value)  # New levenshtein distance
            best_pair = pair  # New best pair
            cur_key = key  # We change the current key
        else:  # We are on the same polish word
            cur_levenshtein = levenshtein(key, value)
            if cur_levenshtein < best_levenshtein:
                best_levenshtein = cur_levenshtein
                best_pair = pair

    end_list.append(best_pair)  # we save the last best pair
    return end_list


def keep_best_edit_distance(list_tuples):
    res = list_tuples
    for i in range(len(list_tuples[0])):
        res = keep_best_edit_distance_on_key(res, i)
    return res

