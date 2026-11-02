"""Tests for the I/O, parallel, corpus, date, transliteration and fake helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from farsi_textutils.corpus import CorpusPipeline, count_words_in_file
from farsi_textutils.dates import DateNormalizer, normalize_dates, normalize_numbers, words_to_number
from farsi_textutils.fake import PersianFaker, fake_profile
from farsi_textutils.logging_setup import configure_logging, get_logger, logged
from farsi_textutils.parallel import ParallelConfig, parallel_map, parallel_starmap
from farsi_textutils.paths import (
    ensure_directory,
    ensure_file,
    iter_files,
    read_lines,
    read_text,
    write_lines,
    write_text,
)
from farsi_textutils.transliterate import PinglishConverter, slugify, to_latin, to_persian
from farsi_textutils.validators import national_id, phone

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #


def test_read_and_write_text(tmp_path: Path) -> None:
    target = write_text(tmp_path / "a.txt", "سلام")
    assert target.read_text(encoding="utf-8") == "سلام"
    assert read_text(target) == "سلام"


def test_read_and_write_lines(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "b.txt", ["یک", "دو"])
    assert read_lines(path) == ["یک", "دو"]
    write_lines(tmp_path / "c.txt", ["یک", "", "دو"])
    assert read_lines(tmp_path / "c.txt") == ["یک", "دو"]
    assert read_lines(tmp_path / "c.txt", skip_blank=False) == ["یک", "", "دو"]


def test_ensure_directory_and_file(tmp_path: Path) -> None:
    directory = ensure_directory(tmp_path / "nested" / "dir")
    assert directory.is_dir()
    file = write_text(directory / "f.txt", "x")
    assert ensure_file(file) == file
    with pytest.raises(FileNotFoundError):
        ensure_file(directory / "missing.txt")


def test_iter_files(tmp_path: Path) -> None:
    write_text(tmp_path / "a.txt", "x")
    write_text(tmp_path / "b.md", "x")
    write_text(tmp_path / "sub" / "c.txt", "x")
    assert [path.name for path in iter_files(tmp_path)] == ["a.txt", "c.txt"]
    assert [path.name for path in iter_files(tmp_path, recursive=False)] == ["a.txt"]
    assert [path.name for path in iter_files(tmp_path, suffixes=(".md",))] == ["b.md"]


def test_read_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_text(tmp_path / "missing.txt")


# --------------------------------------------------------------------------- #
# parallel
# --------------------------------------------------------------------------- #


def test_parallel_map_and_starmap() -> None:
    assert sorted(parallel_map(abs, [-1, -2, -3], workers=2)) == [1, 2, 3]
    assert sorted(parallel_starmap(pow, [(2, 3), (3, 2)], workers=2)) == [8, 9]


def test_parallel_map_single_worker_matches_serial() -> None:
    items = ["الف", "ب", "پ"]
    assert sorted(parallel_map(str.upper, items, workers=1)) == sorted(str.upper(item) for item in items)


def test_parallel_config_validation() -> None:
    with pytest.raises(ValueError):
        ParallelConfig(workers=0)
    config = ParallelConfig(workers=8, chunksize=2)
    assert config.merged(workers=1).workers == 1
    assert config.merged() is config


# --------------------------------------------------------------------------- #
# corpus pipeline
# --------------------------------------------------------------------------- #


def test_corpus_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    write_text(source / "one.txt", "مي روم به كتاب خانه ها")
    write_text(source / "two.txt", "كتاب خوب است")
    write_text(source / "ignored.md", "كتاب")

    pipeline = CorpusPipeline(workers=2)
    assert len(pipeline.files(source)) == 2

    normalised = pipeline.normalise_directory(source, tmp_path / "clean")
    assert len(normalised) == 2
    assert "می\u200cروم" in read_text(tmp_path / "clean" / "one.txt")

    tokenised = pipeline.tokenise_directory(tmp_path / "clean", tmp_path / "tokens")
    assert len(tokenised) == 2
    # tokens are written one per line by default
    assert read_lines(tmp_path / "tokens" / "one.txt")[0] == "می\u200cروم"

    counts = pipeline.word_counts(tmp_path / "clean")
    assert counts["کتاب"] >= 1

    model = pipeline.build_language_model(tmp_path / "clean", output=tmp_path / "model.json", order=2)
    assert model.is_known("کتاب")
    assert (tmp_path / "model.json").exists()


def test_count_words_in_file(tmp_path: Path) -> None:
    path = write_text(tmp_path / "a.txt", "کتاب کتاب خوب")
    counts = count_words_in_file(path)
    assert counts["کتاب"] == 2


# --------------------------------------------------------------------------- #
# dates
# --------------------------------------------------------------------------- #


def test_normalize_numbers_and_dates() -> None:
    assert normalize_numbers(["او", "سه", "کتاب"]) == ["او", "3", "کتاب"]
    assert normalize_dates(["تاریخ", "12", "مرداد", "1402"]) == "تاریخ y1402m5d12"


def test_date_normalizer() -> None:
    normalizer = DateNormalizer()
    assert normalizer.normalize("تاریخ 12 مرداد 1402 بود") == "تاریخ y1402m5d12 بود"
    assert normalizer.find_numbers(["یک", "دو"]) == [0, 1]
    assert normalizer.find_date(["12", "مرداد", "1402"]) == ("y1402m5d12", 0, 2)
    assert normalizer.find_date(["کتاب"]) is None


def test_words_to_number_ignores_unknown_tokens() -> None:
    assert words_to_number(["دو", "هزار", "و", "سی"]) == 2030
    assert words_to_number(["سه", "و", "نیم"]) == 3
    assert words_to_number([]) == 0


# --------------------------------------------------------------------------- #
# transliteration
# --------------------------------------------------------------------------- #


def test_to_latin_and_slug() -> None:
    assert to_latin("سلام") == "slam"
    assert slugify("سلام دنیا") == "slam-dnia"
    assert slugify("سلام دنیا", separator="_") == "slam_dnia"
    assert len(slugify("سلام دنیا", max_length=4)) <= 4


def test_pinglish_converter() -> None:
    converter = PinglishConverter()
    assert converter.convert("ketab", lexicon={"کتاب"}) == "کتاب"
    assert converter.convert("salam", lexicon={"سلام"}) == "سلام"
    assert converter.candidates("ketab")
    assert to_persian("salam") == "سلم"


# --------------------------------------------------------------------------- #
# fake data
# --------------------------------------------------------------------------- #


def test_fake_values_are_valid() -> None:
    faker = PersianFaker(seed=42)
    assert phone.validate(faker.phone_number())
    assert national_id.validate(faker.national_id())
    assert faker.email().endswith("@example.com")
    assert faker.first_name(gender="male")
    assert 10 <= faker.age() <= 99


def test_fake_is_deterministic() -> None:
    assert PersianFaker(seed=5).full_name() == PersianFaker(seed=5).full_name()
    assert PersianFaker(seed=5).city() != ""
    assert len(fake_profile(seed=1)["full_name"].split()) >= 2


def test_fake_sentence_and_paragraph() -> None:
    faker = PersianFaker(seed=11)
    # *words* nouns plus the closing verb
    assert len(faker.sentence(words=4).split()) == 5
    # every default sentence is six nouns plus one verb
    assert len(faker.paragraph(sentences=2).split()) == 14
    assert faker.last_name()


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #


def test_logging_helpers(tmp_path: Path) -> None:
    logfile = tmp_path / "logs" / "run.log"
    configure_logging("DEBUG", logfile=logfile, colorize=False)
    get_logger("tests").debug("hello {}", "world")

    @logged("computing")
    def add(left: int, right: int) -> int:
        return left + right

    assert add(1, 2) == 3
    configured = configure_logging("INFO")
    assert configured is None
    assert logfile.exists()
