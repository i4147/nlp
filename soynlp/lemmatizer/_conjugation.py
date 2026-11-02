# -*- encoding:utf8 -*-

from soynlp.hangle import compose, decompose

positive_moum = set("ㅏㅑㅗㅛ")
negative_moum = set("ㅓㅕㅜㅠ")
neuter_moum = set("ㅡㅣ")
pos_to_neg = {"ㅏ": "ㅓ", "ㅑ": "ㅕ", "ㅗ": "ㅜ", "ㅛ": "ㅠ"}
neg_to_pos = {"ㅓ": "ㅏ", "ㅕ": "ㅑ", "ㅜ": "ㅗ", "ㅠ": "ㅛ"}


def conjugate_chat(stem, ending, enforce_moum_harmoney=False, debug=False):
    if not ending:
        return {stem}

    candidates = conjugate(stem, ending, enforce_moum_harmoney, debug)

    l_len = len(stem)
    l_last = list(decompose(stem[-1]))
    l_last_ = stem[-1]
    r_first = list(decompose(ending[0]))

    if r_first[1] == " " and r_first[0] != " ":
        l = stem[:-1] + compose(l_last[0], l_last[1], r_first[0])
        r = ending[1:]
        surface = l + r
        candidates.add(surface)
        if r_first[1] != " ":
            candidates.add(stem + ending)
        if debug:
            print("어미의 첫 글자가 자음인 경우: {}".format(surface))

    return candidates


def conjugate(stem, ending, enforce_moum_harmoney=False, debug=False):

    assert ending

    l_len = len(stem)
    l_last = list(decompose(stem[-1]))
    l_last_ = stem[-1]
    r_first = list(decompose(ending[0]))

    if enforce_moum_harmoney:
        if (l_last[2] != "ㅂ" and l_last[1] in positive_moum) and (r_first[0] == "ㅇ" and r_first[1] in negative_moum):
            r_first[1] = neg_to_pos[r_first[1]]
            ending = compose(*r_first) + ending[1:]
        if (l_last[2] != "ㅂ" and l_last[1] in negative_moum) and (r_first[0] == "ㅇ" and r_first[1] in positive_moum):
            r_first[1] = pos_to_neg[r_first[1]]
            ending = compose(*r_first) + ending[1:]
        if (l_last[1] in neuter_moum) and (r_first[1] in positive_moum):
            r_first[1] = pos_to_neg[r_first[1]]
            ending = compose(*r_first) + ending[1:]

    r_first_ = compose(r_first[0], r_first[1], " ") if r_first[1] != " " else ending[0]

    candidates = set()

    if debug:
        print("l_last = {}".format(l_last))
        print("r_first = {}".format(r_first))

    if ending[0] == "다":
        surface = stem + ending
        candidates.add(surface)
        if debug:
            print("'다'로 시작하는 어미: {}".format(surface))

    if l_last[2] == "ㄷ" and r_first[0] == "ㅇ":
        l = stem[:-1] + compose(l_last[0], l_last[1], "ㄹ")
        surface = l + ending
        candidates.add(surface)
        candidates.add(stem + ending)
        if debug:
            print("ㄷ 불규칙: {}".format(surface))

    if (l_last_ == "르" and stem[-2:] != "푸르") and (r_first_ == "아" or r_first_ == "어") and l_len >= 2:
        c0, c1, c2 = decompose(stem[-2])
        l = stem[:-2] + compose(c0, c1, "ㄹ")
        r = compose("ㄹ", r_first[1], r_first[2]) + ending[1:]
        surface = l + r
        candidates.add(surface)
        if debug:
            print("르 불규칙: {}".format(surface))

    if l_last[2] == "ㅂ":
        l = stem[:-1] + compose(l_last[0], l_last[1], " ")
        if r_first_ == "어" or r_first_ == "아":
            if l_len >= 2 and (l_last_ == "답" or l_last_ == "곱" or l_last_ == "깝" or l_last_ == "롭"):
                c1 = "ㅝ"
            elif r_first[1] == "ㅗ":
                c1 = "ㅘ"
            elif r_first[1] == "ㅜ":
                c1 = "ㅝ"
            elif r_first_ == "어":
                c1 = "ㅝ"
            else:
                c1 = "ㅘ"
            r = compose("ㅇ", c1, r_first[2]) + ending[1:]
            surface = l + r
            candidates.add(surface)
            if debug:
                print("ㅂ 불규칙: {}".format(surface))
        elif r_first[0] == "ㅇ":
            surface = l + ending
            candidates.add(surface)
            if debug:
                print("ㅂ 불규칙: {}".format(surface))

    if r_first[1] == " " and (
        r_first[0] == "ㄴ" or r_first[0] == "ㄹ" or r_first[0] == "ㅁ" or r_first[0] == "ㅂ" or r_first[0] == "ㅆ"
    ):
        l = stem[:-1] + compose(l_last[0], l_last[1], r_first[0])
        r = ending[1:]
        surface = l + r
        candidates.add(surface)
        if r_first[1] != " ":
            candidates.add(stem + ending)
        if debug:
            print("어미의 첫 글자가 -ㄴ, -ㄹ, -ㅁ-, -ㅂ, -ㅆ 인 경우: {}".format(surface))

    if (l_last[2] == "ㅅ") and (r_first[0] == "ㅇ"):
        if stem[-1] == "벗":
            l = stem
        else:
            l = stem[:-1] + compose(l_last[0], l_last[1], " ")
        surface = l + ending
        candidates.add(surface)
        if debug:
            print("ㅅ 불규칙: {}".format(surface))

    if l_last[1] == "ㅜ" and l_last[2] == " " and r_first[0] == "ㅇ" and r_first[1] == "ㅓ":
        if l_last_ == "푸":
            l = stem[:-1] + "퍼"
        else:
            l = stem[:-1] + compose(l_last[0], "ㅝ", r_first[2])
        r = ending[1:]
        surface = l + r
        candidates.add(surface)
        if debug:
            print("우 불규칙: {}".format(surface))

    if l_last[1] == "ㅗ" and l_last[2] == " " and r_first[0] == "ㅇ" and r_first[1] == "ㅏ":
        l = stem[:-1] + compose(l_last[0], "ㅘ", r_first[2])
        r = ending[1:]
        surface = l + r
        candidates.add(surface)
        if debug:
            print("오 활용: {}".format(surface))

    if (l_last[1] == "ㅡ") and (l_last[2] == " ") and (r_first[0] == "ㅇ"):
        if l_last[0] == "ㅇ" and len(stem) > 1:
            surface = stem[:-1] + ending
        elif l_last[0] != "ㄹ":
            surface = stem[:-1] + compose(l_last[0], r_first[1], r_first[2]) + ending[1:]
        else:
            surface = None
        if surface is not None:
            candidates.add(surface)
        if debug and surface is not None:
            print("ㅡ 탈락 불규칙: {}".format(surface))

    if ending[:2] == "어라" or ending[:2] == "아라":
        if stem[-1] == "오":
            l = stem[:-1]
            r = "와" + ending[1:]
        elif stem[-1] == "우":
            l = stem[:-1]
            r = "워" + ending[1:]
        elif stem[-1] == "가":
            l = stem
            r = ending[1:]
        else:
            if l_last[1] in negative_moum:
                l = stem
                r = "어" + ending[1:]
            else:
                l = stem
                r = "아" + ending[1:]
        surface = l + r
        candidates.add(surface)
        if debug:
            print("거라/너라 불규칙: {}".format(surface))

    if (l_last_ == "르" and stem[-2:] != "구르") and (r_first[0] == "ㅇ" and r_first[1] == "ㅓ"):
        r = compose("ㄹ", r_first[1], r_first[2]) + ending[1:]
        surface = stem + r
        candidates.add(surface)
        if debug:
            print("러 불규칙: {}".format(surface))

    if l_last_ == "하" and r_first[0] == "ㅇ" and (r_first[1] == "ㅏ" or r_first[1] == "ㅓ"):
        r = compose(r_first[0], "ㅕ", r_first[2]) + ending[1:]
        surface0 = stem + r
        candidates.add(surface0)
        l = stem[:-1] + compose("ㅎ", "ㅐ", r_first[2])
        r = ending[1:]
        surface1 = l + r
        candidates.add(surface1)
        if debug:
            print("여 불규칙: {}, {}".format(surface0, surface1))

    if l_last[2] == "ㅎ" and r_first[1] != " ":
        if l_last_ == "좋" or l_last_ == "놓":
            l = stem
        else:
            l = stem[:-1] + compose(l_last[0], l_last[1], " ")
        r = ending
        surface = l + r
        candidates.add(surface)
        if debug:
            print("ㅎ 탈락 불규칙: {}".format(surface))

    if (l_last[2] == "ㅎ" and l_last_ != "좋") and (r_first[0] == "ㅇ" and r_first[1] == "ㅏ" or r_first[1] == "ㅓ"):
        l = stem[:-1] + compose(l_last[0], "ㅐ" if r_first[1] == "ㅏ" else "ㅔ", r_first[2])
        r = ending[1:]
        surface = l + r
        candidates.add(surface)
        if debug:
            print("ㅎ 축약 불규칙: {}".format(surface))

    if l_last[2] == "ㅎ" and r_first[0] == "ㄴ" and r_first[1] != " ":
        surface = stem + ending
        candidates.add(surface)
        if debug:
            print("ㅎ + 네 불규칙: {}".format(surface))

    if r_first_ == "어" and l_last[1] == "ㅣ" and l_last[2] == " ":
        surface = stem[:-1] + compose(l_last[0], "ㅕ", r_first[2]) + ending[1:]
        candidates.add(surface)
        surface = stem + ending
        candidates.add(surface)
        if debug:
            print("이 + 어 -> 여 규칙: {}".format(surface))

    if not candidates and r_first[1] != " ":
        if (l_last[2] == " ") and (r_first[0] == "ㅇ") and (r_first[1] == l_last[1]):
            l = stem[:-1] + compose(l_last[0], l_last[1], r_first[2])
            r = ending[1:]
            surface = l + r
            candidates.add(surface)
        else:
            surface = stem + ending
            candidates.add(surface)
        if debug:
            print("L + R 규칙 결합: {}".format(surface))

    return candidates


def _conjugate_stem(stem, debug=False):

    l_len = len(stem)
    l_last = decompose(stem[-1])
    l_last_ = stem[-1]

    candidates = {stem}

    if l_last[2] == "ㄷ":
        l = stem[:-1] + compose(l_last[0], l_last[1], "ㄹ")
        candidates.add(l)
        if debug:
            print("ㄷ 불규칙")

    if (l_last_ == "르") and l_len >= 2:
        c0, c1, c2 = decompose(stem[-2])
        l = stem[:-2] + compose(c0, c1, "ㄹ")
        candidates.add(l)
        if debug:
            print("르 불규칙")

    if l_last[2] == "ㅂ":
        l = stem[:-1] + compose(l_last[0], l_last[1], " ")
        candidates.add(l)
        if debug:
            print("ㅂ 불규칙")

    if l_last[2] == " ":
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], "ㄴ"))
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], "ㄹ"))
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], "ㅂ"))
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], "ㅆ"))
        if debug:
            print("어미의 첫 글자가 -ㄴ, -ㄹ, -ㅂ, -ㅆ 일 경우")

    if (l_last[2] == "ㅅ") and stem[-1] != "벗":
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], " "))
        if debug:
            print("ㅅ 불규칙")

    if l_last[1] == "ㅜ" and l_last[2] == " ":
        if l_last_ == "푸":
            l = "퍼"
        else:
            candidates.add(stem[:-1] + compose(l_last[0], "ㅝ", " "))
            candidates.add(stem[:-1] + compose(l_last[0], "ㅝ", "ㅆ"))
            if debug:
                print("우 불규칙")

    if l_last[1] == "ㅗ" and l_last[2] == " ":
        candidates.add(stem[:-1] + compose(l_last[0], "ㅘ", " "))
        candidates.add(stem[:-1] + compose(l_last[0], "ㅘ", "ㅆ"))
        if debug:
            print("오 + 았어 -> 왔어 규칙")

    if l_last[1] == "ㅡ" and l_last[2] == " ":
        candidates.add(stem[:-1] + compose(l_last[0], "ㅓ", " "))
        candidates.add(stem[:-1] + compose(l_last[0], "ㅓ", "ㅆ"))
        if l_last[0] == "ㅇ" and len(stem) > 1:
            candidates.add(stem[:-1])
        if debug:
            print("ㅡ 탈락 불규칙")

    if l_last_ == "하":
        candidates.add(stem[:-1] + "해")
        candidates.add(stem[:-1] + "했")
        if debug:
            print("하 -> 해, 했 활용")

    if l_last[2] == "ㅎ" and l_last_ != "좋":
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], " "))
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], "ㄴ"))
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], "ㄹ"))
        candidates.add(stem[:-1] + compose(l_last[0], l_last[1], "ㅆ"))
        if debug:
            print("ㅎ 탈락 불규칙")

    if l_last[2] == "ㅎ" and l_last_ != "좋":
        candidates.add(stem[:-1] + compose(l_last[0], "ㅐ", "ㅆ"))
        if debug:
            print("ㅎ 축약 불규칙")

    if l_last[1] == "ㅣ" and l_last[2] == " ":
        candidates.add(stem[:-1] + compose(l_last[0], "ㅕ", "ㅆ"))
        if debug:
            print("이었 -> 였 규칙")

    return candidates
