import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from parse import parse
from plox import Plox
from z_chembedding import ChemBERTaEmbedder, compute_dispersion
from u_utils import canon_or_none, mol_formula_or_none, is_match_while_not_none


def process_translation(
        translation_file: Path,
        use_chiral: bool = True,
        use_sum_formula: bool = False,
):
    print(f"Reading file {translation_file.relative_to(Path(__file__).parent)}")

    out_folder = Path(__file__).with_suffix('') / translation_file.relative_to(Path(__file__).parent)
    if not out_folder.exists():
        out_folder.mkdir(parents=True, exist_ok=True)

    pattern = "{tgt_val_file}__model_step_{model_step:d}__n_best={n_best:d}__beam_size={beam_size:d}.txt.json"
    parsed = parse(pattern, translation_file.name)
    print(f"Extracted parameters: {parsed.named}")

    with translation_file.open(mode='r') as fd:
        data = json.load(fd)

    df = pd.DataFrame(data=[
        {
            'ref': canon_or_none(sample['ref'], use_chiral=use_chiral),
            'sample_id': i,
            'score': hyp['score'],
            'is_match': is_match_while_not_none(
                canon_or_none(sample['ref'], use_chiral=use_chiral),
                canon_or_none(hyp['pred'], use_chiral=use_chiral),
            ),
            'sum_formula_match': is_match_while_not_none(
                mol_formula_or_none(sample['ref']),
                mol_formula_or_none(hyp['pred']),
            ),
            'pred': canon_or_none(hyp['pred'], use_chiral=use_chiral),
            'n': n + 1  # hypothesis rank (starting at 1)
        }
        for i, sample in enumerate(data)
        for n, hyp in enumerate(sample['hyps'])
    ])

    # Sanity check: cannot have 'is_match' if not 'sum_formula_match'
    assert not df[(~df['sum_formula_match']) & df['is_match']].any().any()

    if use_sum_formula:
        df['n'] = (
            df.groupby('sample_id')['sum_formula_match']
            .cumsum()  # cumulative sum within each sample_id
            .where(df['sum_formula_match'])  # keep only where sum_formula_match is True
        )
        df['n'] = df['n'].astype('Int64')
        print(f"Original number of samples: {len(data)}")
        print(f"Number of samples left after filtering by sum formula: {df.sample_id.nunique()}")

    assert df.n.min() == 1

    top_n_accuracy = {
        n: (
            df[df.n <= n]
            .groupby('sample_id')
            .is_match
            .any()
            .reindex(df['sample_id'].unique(), fill_value=False)
            .mean()
        )
        for n in sorted(df.n.dropna().unique())
    }
    print(top_n_accuracy)

    # Process each top-n setting
    for n in [6]:  # sorted(df.n.dropna().unique()):
        print(f"Processing top-{n} predictions")
        df_n = df[df.n <= n].copy()

        # Determine per-sample match status (True if at least one top-n prediction is a match)
        sample_match = df_n.groupby('sample_id').is_match.any()

        # Compute dispersion for each sample having at least two valid (non-None) predictions
        sample_dispersion = {}  # sample_id -> dispersion
        embedder = ChemBERTaEmbedder(use_cuda=False)
        for sample_id, group in df_n.groupby('sample_id'):
            smiles_list = group['pred'].dropna().unique().tolist()
            if len(smiles_list) < 2:
                continue  # cannot compute dispersion with fewer than 2 valid SMILES
            disp = compute_dispersion(embedder.embed(smiles_list))
            sample_dispersion[sample_id] = disp

        # Separate dispersion values by whether sample has a match or not
        dispersion_matched = [disp for sample_id, disp in sample_dispersion.items() if sample_match.loc[sample_id]]
        dispersion_unmatched = [disp for sample_id, disp in sample_dispersion.items() if not sample_match.loc[sample_id]]

        # Plot histograms using Plox
        with Plox({'figure.figsize': (10, 6)}) as px:
            # Define bins – dispersion is in [0, 1] since cosine distances are in [0,2]
            bins = np.linspace(0, 0.2, 2 ** 6 + 1)
            if dispersion_matched:
                px.a.hist(dispersion_matched, bins=bins, alpha=0.6,
                          label="Samples with top-n match", color="tab:blue")
            if dispersion_unmatched:
                px.a.hist(dispersion_unmatched, bins=bins, alpha=0.6,
                          label="Samples without top-n match", color="tab:red")

            px.a.set_xlabel("Dispersion (1/2 avg. cosine distance)")
            px.a.set_ylabel("Number of Samples")
            px.a.set_title(f"Dispersion Histogram (Top-{n} Predictions, Accuracy: {top_n_accuracy[n]:.2%})")
            px.a.legend()
            px.a.grid(True, lw=0.5, alpha=0.5)
            filename = f"dispersion_hist__use_chiral={use_chiral}__use_sum_formula={use_sum_formula}__top-{n}.png"
            px.f.savefig(out_folder / filename, dpi=300)
            print(f"Saved plot to {out_folder / filename}")


def main():
    translations: list[Path] = list(Path(__file__).parent.glob("work/**/translation/*.txt.json"))
    translations = [
        translation_file
        for translation_file in translations
        if (("_100000_" in translation_file.name) or ("_250000_" in translation_file.name))
        and ("_n10000." in translation_file.name)
    ]
    for translation_file in translations:
        for (use_chiral, use_sum_formula) in product([True, False], [True, False]):
            process_translation(translation_file, use_chiral=use_chiral, use_sum_formula=use_sum_formula)
            exit()


if __name__ == "__main__":
    main()
