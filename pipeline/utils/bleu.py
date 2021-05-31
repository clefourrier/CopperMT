import sacrebleu
import math


def bleu_score(predictions: list, expected: list) -> float:
    """ Adaptation of the bleu score computations to manage batches

    :param predictions: List of sentences (lists of words)
    :param expected: List of end sentences (which are list of words)
    :return: Bleu score
    """
    predictions = [
        [" ".join(word[1:-1] if word[-1] == "EOW" else word[1:]) for word in cur_words]
        for cur_words in predictions
    ]  # Removing sow, eow
    expected = [" ".join(item[1:-1]) for item in expected]  # Removing sow, eow
    # should be opposite sense, but does it change a lot?
    return sacrebleu.corpus_bleu(expected, predictions, tokenize="none").score


def bleu_score_with_tokens(predictions: list, expected: list) -> float:
    """ For moses
    Adaptation of the bleu score computations to manage batches, for data where it is not necessary to remove
     start or end of word tokens

    :param predictions: List of sentences (lists of words)
    :param expected: List of end sentences (which are list of words)
    :return: Bleu score
    """
    predictions = [[" ".join(word) for word in cur_words] for cur_words in predictions]
    expected = [" ".join(item) for item in expected]

    return sacrebleu.corpus_bleu(expected, predictions, tokenize="none").score


def get_neural_bleu(path, l_in, l_out, n_best, print_res=True, print_info=""):
    _, target, prediction = get_neural_bleu_predictions(path, l_in, l_out, n_best)
    bleu = bleu_score_with_tokens(prediction, target)
    if print_res:
        print(f"{path} {print_info} {str(n_best)} {l_in} {l_out} {str(bleu)}")
    return bleu


def get_neural_bleu_predictions(path, l_in, l_out, n_best, get_confidence=False):
    # Storage
    source = []
    target = []
    prediction = []
    confidence = []
    cur_prediction = []
    cur_confidence = []
    with open(
            f'{path}/bleu/bleu_checkpoint_best_{l_in}_{l_out}.{l_out}', 'r') as file:
        for i, line in enumerate(file):
            line = line.split("\t")
            # Actual source
            if "S-" in line[0]:
                word = line[1].strip(' ').split()
                source.append(word)
                # We reinitialize the cur_prediction list
                if len(cur_prediction) > 0:
                    prediction.append(cur_prediction)
                    confidence.append(cur_confidence)
                    cur_prediction = []
                    cur_confidence = []
            # Actual target
            if "T-" in line[0]:
                word = line[1].strip(' ').split()
                target.append(word)
            # Hypothesis
            if "H-" in line[0] and len(cur_prediction) <= n_best:
                word = line[2].strip(' ').split()
                cur_prediction.append(word)
                cur_confidence.append(math.exp(float(line[1])))

        prediction.append(cur_prediction)
        confidence.append(cur_confidence)
        prediction = [[bor[n] for bor in prediction] for n in range(n_best)]

    if get_confidence:
        return source, target, prediction, confidence
    return source, target, prediction


def get_statistical_bleu(path_data, path, l_in, l_out, n_best, cur_n_best, print_res=False):
    _, target, prediction = get_statistical_bleu_predictions(path_data, path, l_in,
                                                             l_out, n_best, cur_n_best)
    bleu = bleu_score_with_tokens(prediction, target)
    if print_res:
        print(f"{path} {str(cur_n_best)} {l_in} {l_out} {str(bleu)}")

    return bleu


def get_statistical_bleu_predictions(path_data, path, l_in, l_out, n_best, cur_n_best, get_confidence=False):
    target = []
    try:
        with open(f'{path_data}/test_{l_in}_{l_out}.{l_out}', 'r') as file:
            for i, line in enumerate(file):
                target.append(line.split())
    except FileNotFoundError:
        pass

    source = []
    with open(f'{path_data}/test_{l_in}_{l_out}.{l_in}', 'r') as file:
        for i, line in enumerate(file):
            source.append(line.split())

    prediction = []
    confidence = []
    cur_ix = -1
    cur_prediction = []
    cur_confidence = []
    with open(f'{path}/{l_in}_{l_out}/out/'
              f'test_{l_in}_{l_out}_nbest_{str(n_best)}.{l_out}', 'r') as file:
        for i, line in enumerate(file):
            line = line.split("|||")
            ix = int(line[0])
            word = line[1].strip(' ').split()

            if cur_ix != ix:
                if cur_ix != -1:
                    while len(cur_prediction) < cur_n_best:
                        cur_prediction.append(cur_prediction[-1])
                        cur_confidence.append(cur_confidence[-1])
                    prediction.append(cur_prediction)
                    confidence.append(cur_confidence)
                cur_prediction = [word]
                cur_confidence = [math.exp(float(line[-1]))]
                cur_ix = ix
            else:
                cur_prediction.append(word)
                cur_confidence.append(math.exp(float(line[-1])))
        # Management of last prediction
        while len(cur_prediction) < cur_n_best:
            cur_prediction.append(cur_prediction[-1])
            cur_confidence.append(cur_confidence[-1])
        prediction.append(cur_prediction)
        confidence.append(cur_confidence)

    prediction = [[bor[n] for bor in prediction] for n in range(cur_n_best)]

    if get_confidence:
        return source, target, prediction, confidence

    return source, target, prediction