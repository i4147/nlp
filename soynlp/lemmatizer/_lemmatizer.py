# -*- encoding:utf8 -*-

from soynlp.hangle import compose, decompose
from ._conjugation import conjugate, conjugate_chat


class Lemmatizer:
    def __init__(self, stems, endings, predefined=None):
        self._stems = stems
        self._endings = endings
        self._initialize()
        if predefined:
            self._predefined.update(predefined)

    def _initialize(self):
        self._predefined = {"불어": ("붇다", "불다"), "그래": ("그렇다",)}

    def lemmatize(self, word, check_only_stem=False):
        candidates = set()
        for i in range(1, len(word) + 1):
            l, r = word[:i], word[i:]
            for stem, ending in lemma_candidate(l, r, self._predefined):
                if stem in self._stems:
                    if check_only_stem:
                        candidates.add((stem, ending))
                    elif ending in self._endings:
                        candidates.add((stem, ending))
        return candidates

    def candidates(self, word):
        candidates = set()
        for i in range(1, len(word) + 1):
            l = word[:i]
            r = word[i:]
            candidates.update(self.lemma_candidate(l, r, self._predefined))
        return candidates


def debug_message(message, l, r):
    print("{}: {} + {}".format(message, l, r))


def lemma_candidate_chat(l, r, predefined=None, debug=False):
    def add_lemma(stem, ending):
        candidates.add((stem, ending))

    def character_is_emoticon(c):
        return c in set("ㄷㅂㅅㅇㅋㅎ")

    candidates = lemma_candidate(l, r, predefined, debug)
    l_last = decompose(l[-1])

    if not r and character_is_emoticon(l_last[2]):
        l_ = l[:-1] + compose(l_last[0], l_last[1], " ")
        if debug:
            debug_message("마지막 종성이 이모티콘으로 의심되는 경우", l_, "()")
        candidates.update(lemma_candidate(l_, r, predefined, debug))

    return candidates


def lemma_candidate(l, r, predefined=None, debug=False):
    def add_lemma(stem, ending):
        candidates.add((stem, ending))

    candidates = {(l, r)}
    word = l + r

    l_last = decompose(l[-1])
    l_last_ = compose(l_last[0], l_last[1], " ")
    l_front = l[:-1]
    r_first = decompose(r[0]) if r else ("", "", "")
    r_first_ = compose(r_first[0], r_first[1], " ") if r else " "
    r_end = r[1:]

    if l_last[2] == "ㄹ" and r_first[0] == "ㅇ":
        l_stem = l_front + compose(l_last[0], l_last[1], "ㄷ")
        add_lemma(l_stem, r)
        if debug:
            debug_message("ㄷ 불규칙 활용", l_stem, r)

    if (l_last[2] == "ㄹ") and (r_first_ == "러" or r_first_ == "라"):
        l_stem = l_front + compose(l_last[0], l_last[1], " ") + "르"
        r_canon = compose("ㅇ", r_first[1], r_first[2]) + r_end
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("르 불규칙 활용", l_stem, r_canon)

    if l_last[2] == " ":
        l_stem = l_front + compose(l_last[0], l_last[1], "ㅂ")
        if r_first_ == "워" or r_first_ == "와":
            r_canon = compose("ㅇ", "ㅏ" if r_first_ == "와" else "ㅓ", r_first[2] if r_first[2] else " ") + r_end
        elif r_end and r_end[0] == "려":
            r_canon = compose("ㅇ", "ㅜ", r_first[2] if r_first[2] else " ") + r_end
        else:
            r_canon = r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("ㅂ 불규칙 활용", l_stem, r_canon)

    if l_last[2] == "ㄴ" or l_last[2] == "ㄹ" or l_last[2] == "ㅁ" or l_last[2] == "ㅂ" or l_last[2] == "ㅆ":
        for jongsung in " ㄹㅂㅎ":
            if l_last[2] == jongsung:
                continue
            l_stem = l_front + compose(l_last[0], l_last[1], jongsung)
            r_canon = l_last[2] + r
            add_lemma(l_stem, r_canon)
            if debug:
                debug_message("어미의 첫글자가 종성일 경우 (%s)" % jongsung, l_stem, r_canon)

    if (l_last[2] == " " and l[-1] != "벗") and (r_first[0] == "ㅇ"):
        l_stem = l_front + compose(l_last[0], l_last[1], "ㅅ")
        add_lemma(l_stem, r)
        if debug:
            debug_message("ㅅ 불규칙 활용", l_stem, r)

    if l_last_ == "퍼":
        l_stem = l_front + "푸"
        r_canon = compose("ㅇ", l_last[1], l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("우 불규칙 활용 (퍼)", l_stem, r_canon)

    if l_last[1] == "ㅝ":
        l_stem = l_front + compose(l_last[0], "ㅜ", " ")
        r_canon = compose("ㅇ", "ㅓ", l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("우 불규칙 활용", l_stem, r_canon)

    if l_last[1] == "ㅘ":
        l_stem = l_front + compose(l_last[0], "ㅗ", " ")
        r_canon = compose("ㅇ", "ㅏ", l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("오 불규칙 활용", l_stem, r_canon)

    if l_last[1] == "ㅓ" or l_last[1] == "ㅏ":
        l_stem = l_front + compose(l_last[0], "ㅡ", " ")
        r_canon = compose("ㅇ", l_last[1], l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("ㅡ 탈락 불규칙 활용 (꺼)", l_stem, r_canon)

    if l_last[2] == " " and r_first[0] == "ㅇ" and (r_first[1] == "ㅏ" or r_first[1] == "ㅓ"):
        l_stem = l + "으"
        r_canon = r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("ㅡ 탈락 불규칙 활용 (모으)", l_stem, r_canon)

    if l_last[0] == "ㅎ" and l_last[1] == "ㅐ":
        l_stem = l_front + "하"
        r_canon = compose("ㅇ", "ㅏ", l_last[2]) + r
        add_lemma(l_stem, r_canon)
        if debug:
            debug_message("여 불규칙 활용", l_stem, r_canon)

    if l_last[2] == " " or l_last[2] == "ㄴ" or l_last[2] == "ㄹ" or l_last[2] == "ㅂ" or l_last[2] == "ㅆ":
        if l_last[1] == "ㅏ" or l_last[1] == "ㅓ":
            l_stem = l_front + compose(l_last[0], l_last[1], "ㅎ")
            r_canon = r if l_last[2] == " " else l_last[2] + r
            add_lemma(l_stem, r_canon)
            if debug:
                debug_message("ㅎ 탈락 불규칙 활용", l_stem, r_canon)
        if (l_last[1] == "ㅐ") or (l_last[1] == "ㅔ"):
            if len(l) >= 2 and l[-2] == "그" and l_last[0] == "ㄹ":
                l_stem = l_front + "렇"
            else:
                l_stem = l_front + compose(l_last[0], "ㅓ" if l_last[1] == "ㅔ" else "ㅏ", "ㅎ")
            r_canon = compose("ㅇ", "ㅓ" if l_last[1] == "ㅔ" else "ㅏ", l_last[2]) + r
            add_lemma(l_stem, r_canon)
            if debug:
                debug_message("ㅎ 축약 불규칙 활용", l_stem, r_canon)

    if (l_last[2] == "ㅆ" or l_last[2] == "ㅅ" or l_last[2] == " ") and (l_last[1] == "ㅕ"):
        if ((l_last[0] == "ㅇ") and (l_last[1] == "ㅕ")) or not (l_last[0] == "ㅇ"):
            l_stem = l_front + compose(l_last[0], "ㅣ", " ")
            r_canon = compose("ㅇ", "ㅓ", l_last[2]) + r
            add_lemma(l_stem, r_canon)
            if debug:
                debug_message("이었 -> 였 규칙 활용", l_stem, r_canon)

    if predefined and (l, r) in predefined:
        for stem in predefined[(l, r)]:
            candidates.add(stem)
            if debug:
                debug_message("Predefined", l_stem, r_canon)

    candidates_ = set()
    for stem, eomi in candidates:
        if not eomi:
            continue
        if decompose(eomi[0])[2] == "ㅎ":
            continue
        surfaces = conjugate(stem, eomi)
        if word in surfaces:
            candidates_.add((stem, eomi))
    return candidates_
