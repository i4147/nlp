# farsi-textutils

A single, modern, type-annotated Persian (Farsi) text-processing library for
**Linux + Python 3.12+**, consolidated from the various Persian projects bundled
in this repository.

```python
>>> from farsi_textutils import normalize, word_tokenize, number_to_words
>>> normalize("مي روم به كتاب خانه ها")
'می‌روم به کتاب خانه‌ها'
>>> word_tokenize("سلام، دنیا!")
['سلام', '،', 'دنیا', '!']
>>> number_to_words(2030)
'دو هزار و سی'
```

The package has **no compiled extensions and no heavy ML dependencies**: it needs
`loguru` and `typer` only, and everything (tokenising, stemming, tagging,
chunking, spelling, similarity, keyword discovery, fake data, Iranian
identifier validation) is implemented with the standard library.

---

## What was dropped

| Reference material | Decision |
| --- | --- |
| Windows / macOS branches (`platform.system() == "Windows"`, `C:/Program Files/Java/...`) | **removed** – Linux only, `pathlib` paths everywhere |
| Python 2 support (`iteritems`, `basestring`, `try: from collections import Iterable`) | **removed** – Python 3.12+ syntax only (`match`-free, `X | Y` unions, `StrEnum`, `slots=True`) |
| `torch`, `onnx`, `sentencepiece` models | **dropped** – replaced by pure-Python n-gram models, dictionary/trie segmenters and TF-IDF/BM25 |
| `nltk` (Punkt sentence splitter) | **replaced** by a small abbreviation-aware state machine |
| `numpy`, `pandas` | **dropped** – `dict`/`Counter` based vectors and statistics |
| `logging` (stdlib) | **replaced** by `loguru` (see `logging_setup.py`) |
| `argparse` **and** `click` CLIs | **replaced** by a single `typer` application |
| Java/Wapiti `POSTagger`, `FindStems`, `SpellCheck` resource files | **reimplemented** heuristically (the reference `parsivar/tokenizer.py` was empty and no resource directory existed) |
| `os.path` traversal, `open()` without `encoding` | **replaced** by `pathlib` + explicit UTF-8 (`paths.py`) |
| Python 2-style `multiprocessing` boilerplate | **replaced** by `Pool.imap_unordered` with 4 workers (`parallel.py`) |
| Non-Persian sibling libraries (`janome`, `konlpy`, `ginza`, `snownlp`, `inltk`, `camel_tools`) | used as **algorithmic references only**; no code depends on them |

## Provenance

| Module | Ported / adapted from |
| --- | --- |
| `normalize.py` | `parsivar.normalizer` (orthography, half-space, spacing rules), `farsi_tools` (character cleanup) |
| `constants.py`, `data/orthography.py` | `parsivar.normalizer.sub_alphabets`, `farsi_tools.__init__` |
| `tokenize.py` | `parsivar.tokenizer` (API), `camel_tools.tokenizers.word`, `soynlp.tokenizer` (`LTokenizer`, `MaxScoreTokenizer`) |
| `digits.py` | `persian_tools.digits`, `persian_tools.ordinal_suffix`, `farsitools.PersianTools` |
| `dates.py` | `farsi_tools` date/number normalisation (`y1402m5d12` form) |
| `stem.py` | `parsivar.stemmer` (`Stemmer`, `AffixRules`) |
| `pos.py` | `parsivar` POS tag set + `camel_tools`-style lexicon tagger |
| `chunk.py` | `parsivar.chunker` grammar (regexp chunker) |
| `spell.py` | `soynlp`/`snownlp` n-gram spelling correction |
| `similarity.py` | `soynlp.vectorizer`, `snownlp` BM25/TF-IDF |
| `keywords.py` | `soynlp.word` word extraction (cohesion + branching entropy) |
| `stopwords.py` | `farsi_tools` stop-word list (plus a curated function-word preset) |
| `transliterate.py` | `farsi_tools` Pinglish helpers, `farsitools` slugify |
| `fake.py` | `fakefarsi` |
| `validators/*` | `persian_tools` (`national_id`, `legal_id`, `bank_card`, `sheba`, `bill`, `phone_number`, `plate`) |
| `corpus.py`, `paths.py`, `parallel.py`, `logging_setup.py`, `cli.py` | new glue written for this refactor |

`data/*.py` is generated from the reference packages by
[`arena/gen_data_modules.py`](../arena/gen_data_modules.py) (literal tables are
re-emitted as typed Python constants); `data/orthography.py` additionally comes
from [`arena/extract_orthography_map.py`](../arena/extract_orthography_map.py).
Two upstream data defects are repaired during generation: the ``ﻧ`` presentation
form mapped to ``ذ`` (fixed via NFKC), and the ``اصفحان`` province typo.

## Installation

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e .          # add [dev] for pytest/ruff/mypy, [docs] for mkdocs
```

## Usage

```python
import farsi_textutils as ft

# normalisation + tokenisation
ft.normalize("مي روم به كتاب خانه ها")  # 'می‌روم به کتاب خانه‌ها'
ft.word_tokenize("سلام، دنیا!")  # ['سلام', '،', 'دنیا', '!']
ft.sentence_tokenize("او آمد. سپس رفت.")  # ['او آمد.', 'سپس رفت.']

# numbers, digits and dates
ft.number_to_words(2030)  # 'دو هزار و سی'
ft.words_to_number("دو هزار و سی")  # 2030
ft.add_separator(1234567)  # '1,234,567'
ft.normalize_dates(["تاریخ", "12", "مرداد", "1402"])  # 'تاریخ y1402m5d12'

# morphology
ft.Stemmer().stem("کتاب‌ها")  # 'کتاب'
ft.tag_tokens(["کتاب", "خوب", "است"])
# [('کتاب', 'N_SING'), ('خوب', 'ADJ_SIM'), ('است', 'V_PRS')]
ft.RegexpChunker().chunk_text("کتاب خوب")

# text analysis
bm25 = ft.BM25().fit(documents)
bm25.rank("کتاب")  # [DocumentScore(...), ...]
ft.extract_keywords(documents, top_k=5)
model = ft.NgramModel.train(documents, workers=4)
ft.SpellChecker(model).correct_text("کتاب کتب است")

# Iranian identifiers
ft.validators.national_id.validate("0499370899")  # True
ft.validators.bank_card.bank_data("6037997512345670")
ft.validators.sheba.generate()
ft.validators.plate.render("12ب345-67")  # '12ب345ایران67'

# fake data
ft.PersianFaker(seed=42).profile()
```

### Parallel corpus processing

Every bulk operation goes through `parallel.py`
(`multiprocessing.Pool.imap_unordered`, four workers by default):

```python
from pathlib import Path
from farsi_textutils import CorpusPipeline

pipeline = CorpusPipeline(workers=4)
pipeline.normalise_directory("raw/", "clean/")
pipeline.tokenise_directory("clean/", "tokens/")
counts = pipeline.word_counts("clean/")
pipeline.build_language_model("clean/", output=Path("lm.json"))
```

### Logging

```python
from farsi_textutils import configure_logging, get_logger

configure_logging("INFO", logfile=Path("logs/farsi.log"))
get_logger("my-app").debug("hello {}", "world")
```

Importing the package keeps loguru silent; set `FARSI_TEXTUTILS_LOG_LEVEL=DEBUG`
to configure a sink at import time (handy inside worker processes).

### Command line

```console
$ farsi-textutils version
1.0.0
$ farsi-textutils normalize "مي روم به كتاب خانه ها"
می‌روم به کتاب خانه‌ها
$ farsi-textutils digits --to-words 2030
دو هزار و سی
$ farsi-textutils tokenize "سلام، دنیا!"
$ python -m farsi_textutils --help
```

Available commands: `version`, `normalize`, `standardise`, `tokenize`, `stem`,
`spell`, `digits`, `transliterate`, `keywords`, `similarity`, `corpus`, `fake`,
`validate`.

## Development

```bash
cd result
python -m pytest                            # 290+ tests, includes --doctest-modules
ruff check . && ruff format --check .
mypy farsi_textutils                        # strict
```

`pyproject.toml` configures pytest so that `pytest` alone runs both the unit
tests in `tests/` and the doctests inside every module.

## Layout

```
result/
├── README.md
├── pyproject.toml
├── _orthography.json            # intermediate character table (kept for provenance)
├── farsi_textutils/
│   ├── cli.py                   # typer application
│   ├── constants.py             # alphabets, suffixes, regexes
│   ├── chunk.py  corpus.py  dates.py  digits.py  fake.py  keywords.py
│   ├── logging_setup.py  mapping.py  normalize.py  parallel.py  paths.py
│   ├── pos.py  similarity.py  spell.py  stem.py  stopwords.py  tokenize.py
│   ├── transliterate.py
│   ├── data/                    # generated reference tables (banks, names, ...)
│   └── validators/              # national id, legal id, card, sheba, bill, phone, plate
└── tests/
```
