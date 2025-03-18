import csv
import gzip
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

from pathlib import Path

from rdkit import RDLogger
from rdkit.Chem import CanonSmiles, MolToSmiles, MolFromSmiles

RDLogger.DisableLog('rdApp.*')

MRN_EXE = "MestReNova"
[MRN_1H_SCRIPT] = Path(__file__).parent.glob("**/1H_predictor.qs")

SIM_SAMPLE = """
# SMILES: CC[C@@H]1NC(=O)N(c2ccc(Oc3ccc4c(c3)OCCO4)nc2)C12CC2
# Multiplets: 13
# Date: 2025-03-17T00:19:37.370Z
# MestReNova version: 15.1.0-38027
category,centroid,delta,j_values,nH,rangeMax,rangeMin
d,8.284794452735104,8.284833620670657,1.9130859375,1,8.304741596141563,8.26492564519975
dd,7.491911288167356,7.492124982371618,1.924072265625_7.783203125,1,7.5197343845290225,7.464334340808381
dd,6.985270959453218,6.986659579749393,2.3826904296875_8.7896728515625,1,7.01582611889596,6.957496375956522
# :::
dqd,1.4047848455553038,1.4042649285729503,3.480010986328125_7.3646240234375_13.34881591796875,1,1.4611164000570844,1.3474953128980203
td,0.8994097486273284,0.8989684900675606,1.5000152587890625_7.345367431640625,3,0.9331958788288838,0.8648431201310346
""".lstrip()

MAX_HYP = 3


def load_file(file: Path):
    if file.name.endswith(".json.gz"):
        return json.load(gzip.open(file, mode='rt'))

    if file.name.endswith(".json"):
        return json.load(file.open(mode='rt'))

    raise ValueError(f"Unsupported file type: {file.name}")


def sim_1hnmr_(smiles: str):
    # Note: MestReNova doesn't seem to understand explicit hydrogens

    if not MolFromSmiles(smiles):
        return

    smiles = CanonSmiles(smiles)

    # # draw a random representation of the same
    # smiles = MolToSmiles(MolFromSmiles(smiles), canonical=False, doRandom=True)

    with tempfile.NamedTemporaryFile(mode='wt', suffix=".csv") as peaks_file:
        subprocess.run([MRN_EXE, MRN_1H_SCRIPT, "-sf", f"predict_1H,{smiles},{peaks_file.name}"])
        peaks_file.flush()

        with open(peaks_file.name, mode='rt') as fd:
            return fd.read()


def sim_1hnmr(smiles: str):
    if not smiles:
        return

    if not MolFromSmiles(smiles):
        return

    smiles = CanonSmiles(smiles)

    with tempfile.NamedTemporaryFile(mode='wt', suffix=".csv") as peaks_file:
        display_num = os.getpid()  # Use unique display per process

        # Start virtual X server for this process
        xvfb_cmd = f"Xvfb :{display_num} -screen 0 1024x768x24 2> /dev/null &"
        os.system(xvfb_cmd)

        # Run MestReNova with isolated display
        env = os.environ.copy()
        env["DISPLAY"] = f":{display_num}"

        subprocess.run(
            args=[MRN_EXE, MRN_1H_SCRIPT, "-sf", f"predict_1H,{smiles},{peaks_file.name}"],
            env=env,
            stderr=subprocess.DEVNULL,
        )

        peaks_file.flush()

        # Stop Xvfb after running MestReNova
        os.system(f"pkill -f 'Xvfb :{display_num}'")

        with open(peaks_file.name, mode='rt') as fd:
            return fd.read()


def process_hnmr(multiplets) -> str:
    """
    Process 1H NMR multiplets into a "tokenized" string.

    Returns:
        A "tokenized" 1H NMR string.

    Source:
        https://github.com/numpde/fork-of-multimodal-spectroscopic-dataset/blob/d02fadb7a10f80317f699046fbd9e7ad4750dbf4/benchmark/generate_input.py
    """

    multiplet_str = "1HNMR | "

    for peak in multiplets:
        range_max = float(peak["rangeMax"])
        range_min = float(peak["rangeMin"])
        formatted_peak = "{:.2f} {:.2f} ".format(range_max, range_min)
        formatted_peak += "{} {}H ".format(peak["category"], peak["nH"])
        js = str(peak["j_values"])
        if js.casefold() != "None".casefold():
            split_js = list(filter(None, js.split("_")))
            processed_js = ["{:.2f}".format(float(j)) for j in split_js]
            formatted_peak += "J " + " ".join(processed_js)
        multiplet_str += formatted_peak.strip() + " | "

    return multiplet_str[:-2]


def process_sim(sim: str):
    if not sim:
        return None

    # Parse lines like: `# key: value`
    metadata = dict([
        line[2:].replace(": ", ":").split(":", 1)
        for line in filter(lambda x: x.startswith("#"), sim.splitlines())
    ])

    # Parse the CSV data
    reader = csv.DictReader([line for line in sim.splitlines() if not line.startswith("#")])

    return {
        **metadata,
        'multiplets': process_hnmr(list(reader)).strip(),
    }


def process_hypothesis(hyp):
    print(f"Re-simulating hypothesis: {hyp['pred']}")
    return process_sim(sim_1hnmr((hyp['pred'] or "").replace(" ", "")))


def resim_hypotheses_file(file: Path):
    data = load_file(file)

    for entry in data:
        resim_results = list(map(process_hypothesis, entry['hyps'][0:MAX_HYP]))

        # # Not working:
        # with ProcessPoolExecutor() as executor:
        #     resim_results = list(executor.map(process_hypothesis, entry['hyps']))

        # Update hypotheses with results
        for (hyp, resim) in zip(entry['hyps'], resim_results):
            hyp['resim'] = resim

        yield {
            **entry,
            'resim': process_sim(sim_1hnmr((entry['ref'] or "").replace(" ", ""))),
            'hyps': entry['hyps'],
        }


def main():
    translations: list[Path] = list(Path(__file__).parent.glob("work/**/translation/*.json"))

    translations = [
        file
        for file in translations
        if "_100000" in file.name
    ]

    print(f"Found {len(translations)} translation files: ", *translations, sep="\n")

    for file in translations:
        out_file = file.with_name(file.name + "-resim.jsonl")

        with out_file.open(mode='wt') as fd:
            for entry in resim_hypotheses_file(file):
                fd.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
