import random
from fakefarsi.data import *


def complete():
    f = fake_name()
    l = fake_last_name()
    n = f + " " + l
    a = fake_age()
    p = fake_phone()
    e = fake_email(f, l)
    s = f"نام و نام خانوادگی : {n} \nسن : {a} \nشماره تلفن : {p} \nایمیل : \n{e}"
    return s


def fake_name():
    return random.choice(names).strip()


def fake_last_name():
    return random.choice(last_names)


def fake_full_name():
    full_name = random.choice(names).strip() + " " + random.choice(last_names)
    return full_name


def fake_age():
    return random.randint(10, 99)


def fake_phone():
    phone_number = "09"
    index2 = ["1", "3", "9"]
    phone_number += random.choice(index2)
    for i in range(8):
        phone_number += str(random.randint(0, 9))
    return phone_number


def translator(text):
    mapping = {
        "آ": "a",
        "ا": "a",
        "ب": "b",
        "پ": "p",
        "ت": "t",
        "ث": "s",
        "ج": "j",
        "چ": "ch",
        "ه": "h",
        "خ": "kh",
        "د": "d",
        "ذ": "z",
        "ر": "r",
        "ز": "z",
        "س": "s",
        "ش": "sh",
        "ص": "s",
        "ظ": "z",
        "ط": "t",
        "ع": "a",
        "غ": "gh",
        "ق": "gh",
        "ف": "f",
        "ک": "k",
        "گ": "g",
        "ل": "l",
        "م": "m",
        "ن": "n",
        "و": "v",
        "ی": "i",
        "ء": "a",
        "ئ": "e",
        "ي": "i",
        "ح": "h",
        "ض": "z",
        "ك": "k",
    }

    for key, value in mapping.items():
        text = text.replace(key, value)

    return text


def fake_email(name, lastname):
    email = translator(name) + translator(lastname) + str(random.randint(1, 999)) + "@gmail.com"
    return email.replace(" ", "")


names = male_names + female_names

last_names = l_names
