import re
import json
import gzip

from rdkit import RDLogger
from typing import Dict, Any, Generator
from pathlib import Path

RDLogger.DisableLog('rdApp.*')


def parse_sentences(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    """
    Yields dictionaries of the form:
      {'sent': <concatenated sentence>, 'hyps': [<dicts with score and pred>]}
    """
    sentence_pattern = re.compile(r'SENT \d+: (.+)')
    pre_hyp_signal = re.compile(r'^BEST HYP')
    hyp_pattern = re.compile(r'\[(-?\d+\.\d+)\] (\[.+\])')

    def parse_array(array_str):
        return json.loads(array_str.replace("'", '"'))

    def parse_hyp(fd):
        while not pre_hyp_signal.match(next(fd)):
            pass

        while hyp_match := hyp_pattern.match(next(fd)):
            yield {
                'score': float(hyp_match.group(1)),
                'pred': " ".join(parse_array(hyp_match.group(2))),
            }

    with file_path.open(mode='r', encoding="utf-8") as fd:
        for line in fd:
            if sent_match := sentence_pattern.match(line):
                yield {
                    'sent': " ".join(parse_array(sent_match.group(1))),
                    'hyps': list(parse_hyp(fd)),
                }


def main(input_path: Path):
    """Orchestrates reading, parsing, and saving."""

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # input_path.name is like tgt-test_n1000.txt__model_step_30000__n_best=10__beam_size=100.txt
    tgt_file = input_path.name.split("__")[0]
    src_file = tgt_file.replace("tgt", "src")

    tgt_file = input_path.parent.parent / "data" / tgt_file
    src_file = input_path.parent.parent / "data" / src_file

    # zip the files tgt and src
    tgt_lines = tgt_file.read_text().splitlines()
    src_lines = src_file.read_text().splitlines()

    dataset = dict(zip(src_lines, tgt_lines))

    parsed_predictions = list(parse_sentences(input_path))

    parsed_predictions = [
        {
            **entry,
            'ref': dataset.get(entry['sent']),
        }
        for entry in parsed_predictions
    ]

    # Count entries and hypotheses
    num_entries = len(parsed_predictions)
    num_hypotheses = sum(len(entry['hyps']) for entry in parsed_predictions)

    print(f"Entries: {num_entries}, Hypotheses: {num_hypotheses}")

    return parsed_predictions


def save_json(json_obj: Any, dest_file: Path):
    """
    Writes the JSON-like object as a JSON file.
      - If dest_file ends with '.json.gz', the file is compressed using gzip.
      - If dest_file ends with '.json', the file is written in plain text.
      - Otherwise, the function raises a ValueError.
    """
    json_str = json.dumps(json_obj, indent=4, ensure_ascii=False)

    if dest_file.name.endswith(".json.gz"):
        with gzip.open(dest_file, mode='wt', encoding='utf-8') as fd:
            fd.write(json_str)
    elif dest_file.name.endswith(".json"):
        dest_file.write_text(json_str, encoding="utf-8")
    else:
        raise ValueError("Destination file must have extension .json or .json.gz")


if __name__ == "__main__":
    translations: list[Path] = list(Path(__file__).parent.glob("work/**/translation/*.log"))

    for file in translations:
        # Assume main() returns a JSON-serializable object (e.g., a list or dict)
        parsed_json = main(file)
        archive_path = file.with_suffix(".json")
        save_json(parsed_json, archive_path)
