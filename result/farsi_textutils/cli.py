"""Command line interface, built with :mod:`typer`.

The reference projects used ``argparse`` (``ginza``, ``janome``'s tools) and
``click`` (``camel_tools.cli``, ``konlpy``).  Everything is consolidated here into
one ``typer`` application with sub-commands, typed options and rich help:

.. code-block:: console

    $ farsi-textutils normalize "مي روم به كتاب خانه ها"
    می‌روم به کتاب خانه‌ها
    $ farsi-textutils digits --to-words 2030
    دو هزار و سی
    $ farsi-textutils corpus normalize ./raw ./clean --workers 4

Run ``farsi-textutils --help`` for the full list.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from . import __version__
from .corpus import CorpusPipeline
from .digits import Language, add_separator, convert_digits, number_to_words, words_to_number
from .exceptions import FarsiTextUtilsError
from .fake import PersianFaker
from .keywords import WordExtractor, extract_keywords
from .logging_setup import configure_logging, get_logger
from .normalize import Normalizer, NormalizerConfig, standardize
from .paths import read_text, write_text
from .similarity import BM25
from .spell import NgramModel, SpellChecker
from .stem import Stemmer
from .stopwords import content_words
from .tokenize import Tokenizer, TokenizerConfig
from .transliterate import to_latin, to_persian
from .validators import bank_card, bill, legal_id, national_id, phone, plate, sheba

__all__ = ["app", "main"]

_log = get_logger("cli")

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Persian (Farsi) text utilities: normalise, tokenise, stem, spell-check, "
    "validate identifiers, generate fake data and process corpora.",
)
corpus_app = typer.Typer(no_args_is_help=True, help="Batch operations on a directory of text files.")
fake_app = typer.Typer(no_args_is_help=True, help="Generate fake Persian data.")
validate_app = typer.Typer(no_args_is_help=True, help="Validate Iranian identifiers.")

app.add_typer(corpus_app, name="corpus")
app.add_typer(fake_app, name="fake")
app.add_typer(validate_app, name="validate")


class OutputFormat(StrEnum):
    """Output format of the list-producing commands."""

    TEXT = "text"
    JSON = "json"


def _emit(lines: list[str], fmt: OutputFormat) -> None:
    """Print *lines* either raw or as a JSON array."""
    if fmt is OutputFormat.JSON:
        import json

        typer.echo(json.dumps(lines, ensure_ascii=False, indent=2))
        return
    for line in lines:
        typer.echo(line)


def _read_input(text: str | None, file: Path | None, use_stdin: bool) -> str:
    """Resolve the text to process from an argument, a file or stdin."""
    if file is not None:
        return read_text(file)
    if use_stdin:
        return sys.stdin.read()
    if text is None:
        raise typer.BadParameter("provide TEXT, --file or --stdin")
    return text


@app.callback()
def _configure(
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase log verbosity.")] = 0,
    log_file: Annotated[Path | None, typer.Option(help="Also write logs to this file.")] = None,
) -> None:
    """Configure logging for every sub-command."""
    level = "WARNING" if verbose == 0 else "INFO" if verbose == 1 else "DEBUG"
    configure_logging(level, logfile=log_file)  # type: ignore[arg-type]


@app.command()
def version() -> None:
    """Print the library version."""
    typer.echo(__version__)


@app.command()
def normalize(
    text: Annotated[str | None, typer.Argument(help="Text to normalise.")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Read the text from a file.")] = None,
    stdin: Annotated[bool, typer.Option("--stdin", help="Read the text from standard input.")] = False,
    fix_space: Annotated[bool, typer.Option("--fix-space/--no-fix-space", help="Repair half-spaces.")] = True,
    digits_ascii: Annotated[bool, typer.Option("--digits-ascii", help="Convert digits to ASCII.")] = False,
    dates: Annotated[bool, typer.Option("--dates", help="Canonicalise dates and number words.")] = False,
    pinglish: Annotated[bool, typer.Option("--pinglish", help="Convert Latin (pinglish) text.")] = False,
    eliminate_newlines: Annotated[bool, typer.Option("--eliminate-newlines", help="Join lines.")] = False,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write the result to a file.")] = None,
) -> None:
    """Normalise Persian text (orthography, half-spaces, digits, dates)."""
    payload = _read_input(text, file, stdin)
    config = NormalizerConfig(
        fix_space=fix_space,
        digits_to_ascii=digits_ascii,
        date_normalization=dates,
        pinglish_conversion=pinglish,
        new_line_elimination=eliminate_newlines,
    )
    result = Normalizer(config).normalize(payload)
    if out is not None:
        write_text(out, result)
        typer.echo(f"written to {out}")
    else:
        typer.echo(result)


@app.command()
def standardise(
    text: Annotated[str | None, typer.Argument(help="Text to standardise.")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f")] = None,
    persian_digits: Annotated[bool, typer.Option("--persian-digits/--ascii-digits")] = True,
    check: Annotated[bool, typer.Option("--check", help="Fail on non-Persian characters.")] = False,
) -> None:
    """Run the ``farsi_tools`` style standardisation pipeline."""
    payload = _read_input(text, file, False)
    try:
        typer.echo(standardize(payload, replace_ascii_digits=persian_digits, check_persian=check) or "")
    except FarsiTextUtilsError as error:
        typer.secho(str(error), err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from error


@app.command()
def tokenize(
    text: Annotated[str | None, typer.Argument(help="Text to tokenise.")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f")] = None,
    stdin: Annotated[bool, typer.Option("--stdin")] = False,
    mode: Annotated[str, typer.Option(help="'word', 'sentence' or 'paragraph'.")] = "word",
    split_half_space: Annotated[bool, typer.Option("--split-half-space")] = False,
    no_punctuation: Annotated[bool, typer.Option("--no-punctuation")] = False,
    content: Annotated[bool, typer.Option("--content-only", help="Drop stop words.")] = False,
    fmt: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = OutputFormat.TEXT,
) -> None:
    """Tokenise text into words, sentences or paragraphs."""
    payload = _read_input(text, file, stdin)
    tokenizer = Tokenizer(
        TokenizerConfig(split_half_space=split_half_space, keep_punctuation=not no_punctuation)
    )
    handlers = {
        "word": tokenizer.word,
        "sentence": tokenizer.sentence,
        "paragraph": tokenizer.paragraph,
    }
    if mode not in handlers:
        raise typer.BadParameter("mode must be one of word, sentence, paragraph")
    tokens = handlers[mode](payload)
    if content:
        tokens = content_words(" ".join(tokens))
    _emit(tokens, fmt)


@app.command()
def stem(
    words: Annotated[list[str], typer.Argument(help="Words to stem.")],
    pos: Annotated[str | None, typer.Option(help="Restrict to 'N' (noun) or 'V' (verb).")] = None,
) -> None:
    """Stem Persian words with the lexicon free stemmer."""
    stemmer = Stemmer()
    _emit([stemmer.stem(word, pos) for word in words], OutputFormat.TEXT)


@app.command()
def spell(
    text: Annotated[str | None, typer.Argument(help="Text to spell-check.")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f")] = None,
    model: Annotated[Path | None, typer.Option("--model", help="Use a trained model (JSON).")] = None,
    train_on: Annotated[
        Path | None, typer.Option("--train-on", help="Train from a directory of texts.")
    ] = None,
    workers: Annotated[int, typer.Option(help="Worker processes.")] = 4,
    save_model: Annotated[Path | None, typer.Option("--save-model")] = None,
) -> None:
    """Correct spelling using a pure-Python n-gram model."""
    if model is not None:
        checker = SpellChecker(NgramModel.load(model))
    elif train_on is not None:
        pipeline = CorpusPipeline(workers=workers)
        checker = SpellChecker(pipeline.build_language_model(train_on, output=save_model))
    else:
        raise typer.BadParameter("provide --model or --train-on")
    typer.echo(checker.correct_text(_read_input(text, file, False)))


@app.command()
def digits(
    number: Annotated[str | None, typer.Option("--number", "-n", help="Number or spelled number.")] = None,
    to_words: Annotated[int | None, typer.Option("--to-words", help="Spell the given integer.")] = None,
    to_number: Annotated[str | None, typer.Option("--to-number", help="Parse a spelled number.")] = None,
    convert_to: Annotated[
        Language | None, typer.Option("--convert", help="Convert to en/fa/ar digits.")
    ] = None,
    separator: Annotated[bool, typer.Option("--separator", help="Group thousands.")] = False,
) -> None:
    """Convert digits, spell numbers and parse spelled numbers."""
    if to_words is not None:
        typer.echo(number_to_words(to_words) or "")
    elif to_number is not None:
        typer.echo(str(words_to_number(to_number)))
    elif convert_to is not None and number is not None:
        value = convert_digits(number, convert_to)
        typer.echo(add_separator(value) if separator else value)
    else:
        raise typer.BadParameter("use --to-words, --to-number or --convert")


@app.command()
def transliterate(
    text: Annotated[str, typer.Argument(help="Text to transliterate.")],
    target: Annotated[str, typer.Option(help="'latin' (fa->en) or 'persian' (pinglish->fa).")] = "latin",
    lexicon: Annotated[Path | None, typer.Option(help="Word list used to pick the spelling.")] = None,
    slug: Annotated[bool, typer.Option("--slug", help="Produce a URL slug instead.")] = False,
) -> None:
    """Transliterate between Persian and Latin script."""
    if slug:
        from .transliterate import slugify

        typer.echo(slugify(text))
        return
    if target == "latin":
        typer.echo(to_latin(text))
    elif target == "persian":
        known = set(read_text(lexicon).split()) if lexicon is not None else None
        typer.echo(to_persian(text, lexicon=known))
    else:
        raise typer.BadParameter("target must be 'latin' or 'persian'")


@app.command()
def keywords(
    files: Annotated[list[Path], typer.Argument(help="Documents to analyse.")],
    top_k: Annotated[int, typer.Option("--top-k")] = 20,
    discover: Annotated[bool, typer.Option("--discover", help="Also discover unknown words.")] = False,
    workers: Annotated[int, typer.Option(help="Worker processes.")] = 4,
    fmt: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = OutputFormat.TEXT,
) -> None:
    """Extract TF-IDF keywords (and optionally discover words) from documents."""
    documents = [read_text(path) for path in files]
    ranked = extract_keywords(documents, top_k=top_k)
    lines = [f"{item.word}\t{item.frequency}\t{item.score:.3f}" for item in ranked]
    if discover:
        extractor = WordExtractor(workers=workers).fit(documents)
        lines.extend(f"{item.word}\tdiscovered\t{item.score:.3f}" for item in extractor.extract()[:top_k])
    _emit(lines, fmt)


@app.command()
def similarity(
    query: Annotated[str, typer.Argument(help="Search query.")],
    files: Annotated[list[Path], typer.Argument(help="Documents to rank.")],
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
) -> None:
    """Rank documents against a query with BM25."""
    documents = [read_text(path) for path in files]
    for score in BM25().fit(documents).rank(query, top_k=top_k):
        typer.echo(f"{score.score:.4f}\t{files[score.index]}")


# --------------------------------------------------------------------------
# validation sub-commands
# --------------------------------------------------------------------------
@validate_app.command("national-id")
def validate_national_id(values: Annotated[list[str], typer.Argument(help="IDs to validate.")]) -> None:
    """Validate national IDs and print their issuing province."""
    for value in values:
        valid = national_id.validate(value)
        place = national_id.find_place(value) if valid else None
        typer.echo(f"{value}\t{valid}\t{place['province'] if place else '-'}")


@validate_app.command("legal-id")
def validate_legal_id(values: Annotated[list[str], typer.Argument(help="IDs to validate.")]) -> None:
    """Validate legal-entity IDs."""
    for value in values:
        typer.echo(f"{value}\t{legal_id.validate(value)}")


@validate_app.command("card")
def validate_card(values: Annotated[list[str], typer.Argument(help="Card numbers.")]) -> None:
    """Validate bank card numbers and print the issuing bank."""
    for value in values:
        valid = bank_card.validate(value)
        bank = bank_card.bank_data(value) if valid else None
        typer.echo(f"{value}\t{valid}\t{bank['persian_name'] if bank else '-'}")


@validate_app.command("sheba")
def validate_sheba(values: Annotated[list[str], typer.Argument(help="SHEBA numbers.")]) -> None:
    """Validate SHEBA (IBAN) numbers."""
    for value in values:
        valid = sheba.validate(value)
        bank = sheba.bank_data(value) if valid else None
        typer.echo(f"{value}\t{valid}\t{bank['persian_name'] if bank else '-'}")


@validate_app.command("phone")
def validate_phone(values: Annotated[list[str], typer.Argument(help="Mobile numbers.")]) -> None:
    """Validate mobile numbers and print the operator."""
    for value in values:
        valid = phone.validate(value)
        info = phone.operator_data(value) if valid else None
        typer.echo(f"{value}\t{valid}\t{info['operator'] if info else '-'}")


@validate_app.command("plate")
def validate_plate(values: Annotated[list[str], typer.Argument(help="Licence plates.")]) -> None:
    """Validate licence plates and print their province."""
    for value in values:
        valid = plate.is_valid(value)
        info = plate.get_info(value) if valid else None
        typer.echo(f"{value}\t{valid}\t{info['province'] if info else '-'}")


@validate_app.command("bill")
def validate_bill(
    bill_id: Annotated[str | None, typer.Option("--bill-id")] = None,
    payment_id: Annotated[str | None, typer.Option("--payment-id")] = None,
    barcode: Annotated[str | None, typer.Option("--barcode")] = None,
) -> None:
    """Validate a utility bill and print its details."""
    detail = bill.get_detail(bill_id=bill_id, payment_id=payment_id, barcode=barcode)
    for key, value in detail.items():
        typer.echo(f"{key}\t{value}")


# --------------------------------------------------------------------------
# fake data sub-commands
# --------------------------------------------------------------------------
@fake_app.command("profile")
def fake_profile(
    count: Annotated[int, typer.Option("--count", "-n")] = 1,
    seed: Annotated[int | None, typer.Option(help="Seed for reproducible output.")] = None,
    fmt: Annotated[OutputFormat, typer.Option("--format", help="Output format.")] = OutputFormat.JSON,
) -> None:
    """Generate coherent fake person records."""
    faker = PersianFaker(seed)
    if fmt is OutputFormat.JSON:
        import json

        typer.echo(json.dumps([faker.profile() for _ in range(count)], ensure_ascii=False, indent=2))
        return
    for _ in range(count):
        typer.echo(" | ".join(f"{key}={value}" for key, value in faker.profile().items()))


@fake_app.command("text")
def fake_text(
    count: Annotated[int, typer.Option("--count", "-n")] = 1,
    sentences: Annotated[int, typer.Option("--sentences")] = 3,
    seed: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Generate fake Persian sentences."""
    faker = PersianFaker(seed)
    for _ in range(count):
        typer.echo(faker.paragraph(sentences=sentences))


# --------------------------------------------------------------------------
# corpus sub-commands
# --------------------------------------------------------------------------
@corpus_app.command("normalize")
def corpus_normalize(
    source: Annotated[Path, typer.Argument(help="Input directory.")],
    target: Annotated[Path, typer.Argument(help="Output directory.")],
    workers: Annotated[int, typer.Option(help="Worker processes.")] = 4,
    suffix: Annotated[str, typer.Option(help="File suffix to process.")] = ".txt",
) -> None:
    """Normalise every file of a directory tree."""
    pipeline = CorpusPipeline(workers=workers, suffixes=(suffix,))
    written = pipeline.normalise_directory(source, target)
    typer.echo(f"{len(written)} file(s) written to {target}")


@corpus_app.command("tokenize")
def corpus_tokenize(
    source: Annotated[Path, typer.Argument(help="Input directory.")],
    target: Annotated[Path, typer.Argument(help="Output directory.")],
    workers: Annotated[int, typer.Option(help="Worker processes.")] = 4,
    sentence_per_line: Annotated[bool, typer.Option("--sentence-per-line")] = False,
) -> None:
    """Tokenise every file of a directory tree."""
    pipeline = CorpusPipeline(workers=workers)
    written = pipeline.tokenise_directory(source, target, sentence_per_line=sentence_per_line)
    typer.echo(f"{len(written)} file(s) written to {target}")


@corpus_app.command("word-count")
def corpus_word_count(
    source: Annotated[Path, typer.Argument(help="Input directory.")],
    workers: Annotated[int, typer.Option(help="Worker processes.")] = 4,
    top_k: Annotated[int, typer.Option("--top-k")] = 20,
) -> None:
    """Count words across a corpus and print the most frequent ones."""
    pipeline = CorpusPipeline(workers=workers)
    for word, count in pipeline.word_counts(source).most_common(top_k):
        typer.echo(f"{count}\t{word}")


@corpus_app.command("train-lm")
def corpus_train_lm(
    source: Annotated[Path, typer.Argument(help="Input directory.")],
    output: Annotated[Path, typer.Argument(help="Destination JSON model.")],
    workers: Annotated[int, typer.Option(help="Worker processes.")] = 4,
    order: Annotated[int, typer.Option(help="Model order (1 or 2).")] = 2,
) -> None:
    """Train and save an n-gram language model for the spell checker."""
    pipeline = CorpusPipeline(workers=workers)
    model = pipeline.build_language_model(source, output=output, order=order)
    typer.echo(f"model with {model.vocabulary_size} words written to {output}")


def main() -> None:
    """Entry point used by the ``console_scripts`` hook and ``python -m``."""
    try:
        app()
    except FarsiTextUtilsError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=1) from error


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
