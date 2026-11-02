from soynlp.hangle import decompose
from soynlp.lemmatizer import conjugate


def conjugate_as_present(stem):

    eomis_0 = ["ㄴ다", "ㄴ다고", "고있는"]
    eomis_1 = ["는다", "는다고", "고있는"]

    cho, jung, jong = decompose(stem[-1])
    if jong == " ":
        return _conjugate(stem, eomis_0)
    else:
        return _conjugate(stem, eomis_1)


def conjugate_as_imperative(stem):

    eomis_0 = ["어라"]
    eomis_1 = ["아라"]

    cho, jung, jong = decompose(stem[-1])
    if jung == "ㅓ" or jung == "ㅕ":
        return _conjugate(stem, eomis_0)
    else:
        return _conjugate(stem, eomis_1)


def conjugate_as_pleasure(stem):

    eomis = ["자", "ㄹ까", "ㄹ까봐", "까", "까봐", "을까", "을까봐"]
    return _conjugate(stem, eomis)


def _conjugate(stem, eomis):
    return {surface for eomi in eomis for surface in conjugate(stem, eomi)}


def rule_classify(stem):
    adj_suffixs = {"같", "답", "롭", "만하", "스럽", "시럽", "이", "아니"}
    verb_suffixs = {"거리", "당하", "당허", "시키"}

    lastone = stem[-1]
    lasttwo = stem[-2:]
    if (lastone in adj_suffixs) or (lasttwo in adj_suffixs):
        return "Adjective"
    elif (lastone in verb_suffixs) or (lasttwo in verb_suffixs):
        return "Verb"
    return None
