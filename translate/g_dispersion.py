import json
from contextlib import contextmanager

import numpy as np
import pandas as pd

from pathlib import Path
from itertools import product
from parse import parse
from plox import Plox

from z_chembedding import ChemBERTaEmbedder, compute_dispersion
from u_utils import canon_or_none, mol_formula_or_none, is_match_while_not_none


def load_data(translation_file: Path, use_chiral: bool, use_sum_formula: bool):
    print(f"Reading file {translation_file.relative_to(Path(__file__).parent)}")
    out_folder = Path(__file__).with_suffix('') / translation_file.relative_to(Path(__file__).parent)
    out_folder.mkdir(parents=True, exist_ok=True)

    pattern = ("{tgt_val_file}__model_step_{model_step:d}__n_best={n_best:d}"
               "__beam_size={beam_size:d}.txt.json")
    parsed = parse(pattern, translation_file.name)
    print(f"Extracted parameters: {parsed.named}")

    with translation_file.open('r') as fd:
        data = json.load(fd)

    df = pd.DataFrame([
        {
            'ref': canon_or_none(sample['ref'], use_chiral=use_chiral),
            'sample_id': i,
            'score': hyp['score'],
            'is_match': is_match_while_not_none(
                canon_or_none(sample['ref'], use_chiral=use_chiral),
                canon_or_none(hyp['pred'], use_chiral=use_chiral)
            ),
            'sum_formula_match': is_match_while_not_none(
                mol_formula_or_none(sample['ref']),
                mol_formula_or_none(hyp['pred'])
            ),
            'pred': canon_or_none(hyp['pred'], use_chiral=use_chiral),
            'n': n,
        }
        for (i, sample) in enumerate(data)
        for (n, hyp) in enumerate(sample['hyps'], start=1)
    ])

    # Sanity check.
    assert not df[(~df['sum_formula_match']) & df['is_match']].any().any()

    if use_sum_formula:
        df['n'] = df.groupby('sample_id')['sum_formula_match'].cumsum().where(df['sum_formula_match'])
        df['n'] = df['n'].astype('Int64')
        print(f"Original number of samples: {len(data)}")
        print(f"Samples left after filtering by sum formula: {df.sample_id.nunique()}")

    assert df.n.min() == 1

    top_n_accuracy = {
        int(n): float(
            df[df.n <= n].groupby('sample_id').is_match.any()
            .reindex(df['sample_id'].unique(), fill_value=False).mean()
        )
        for n in sorted(df.n.dropna().unique())
    }
    print(top_n_accuracy)
    return df, top_n_accuracy, out_folder


def add_embeddings(df: pd.DataFrame, embedder: ChemBERTaEmbedder):
    df['embedding'] = df['pred'].apply(lambda s: embedder.embed([s])[0] if s is not None else None)


def compute_dispersion_group(group: pd.DataFrame):
    unique_smiles = group['pred'].dropna().unique()
    if len(unique_smiles) < 2:
        return np.nan
    unique_embeddings = []
    for s in unique_smiles:
        row = group[group['pred'] == s].iloc[0]
        if row['embedding'] is not None:
            unique_embeddings.append(row['embedding'])
    if len(unique_embeddings) < 2:
        return np.nan
    unique_embeddings = np.array(unique_embeddings)
    return compute_dispersion(unique_embeddings)


def compute_ref_cosine_distance_group(group: pd.DataFrame, embedder: ChemBERTaEmbedder):
    if not (group['n'] == 1).any():
        return np.nan
    first_row = group[group['n'] == 1].iloc[0]
    ref = group['ref'].iloc[0]
    if ref is None:
        return np.nan
    ref_emb = embedder.embed([ref])[0]
    pred_emb = first_row['embedding']
    if pred_emb is None:
        return np.nan
    from numpy.linalg import norm
    def cosine_distance(u, v):
        if norm(u) == 0 or norm(v) == 0:
            return np.nan
        return 1 - np.dot(u, v) / (norm(u) * norm(v))

    return cosine_distance(ref_emb, pred_emb)


@contextmanager
def plot_histogram(dispersion_matched, dispersion_unmatched, top_n_accuracy, n):
    with Plox({'figure.figsize': (10, 6)}) as px:
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

        yield px


@contextmanager
def plot_scatter(scatter_df, n, is_match):
    with Plox({'figure.figsize': (10, 6)}) as px:
        scatter_df = scatter_df[scatter_df['cosine_distance'] > 0]
        px.a.scatter(scatter_df['dispersion'], scatter_df['cosine_distance'], alpha=0.6)
        px.a.set_xlabel("Dispersion (1/2 avg. cosine distance)")
        px.a.set_ylabel("Cosine Distance (Reference vs First Prediction)")
        px.a.set_title(f"Dispersion vs. cosine distance (top-{n} predictions, {is_match = })")
        px.a.grid(True, lw=0.5, alpha=0.5)
        yield px


def process_top_n(
        n: int,
        df: pd.DataFrame,
        embedder: ChemBERTaEmbedder,
        top_n_accuracy: dict,
        out_folder: Path,
        use_chiral: bool,
        use_sum_formula: bool
):
    print(f"Processing top-{n} predictions")
    df_n = df[df.n <= n].copy()
    sample_match = df_n.groupby('sample_id').is_match.any()

    # Compute dispersion per sample (avoid deprecation warning with include_groups=False)
    sample_dispersion = df_n.groupby('sample_id').apply(
        lambda group: compute_dispersion_group(group.drop(columns='sample_id'))
    )

    dispersion_matched = [
        disp
        for (sample_id, disp) in sample_dispersion.items()
        if sample_match.loc[sample_id] and not pd.isna(disp)
    ]

    dispersion_unmatched = [
        disp
        for (sample_id, disp) in sample_dispersion.items()
        if (not sample_match.loc[sample_id]) and not pd.isna(disp)
    ]

    with plot_histogram(dispersion_matched, dispersion_unmatched, top_n_accuracy, n, ) as px:
        hist_filename = f"dispersion_hist__use_chiral={use_chiral}__use_sum_formula={use_sum_formula}__top-{n}.png"
        px.f.savefig(out_folder / hist_filename, dpi=300)
        print(f"Saved plot to {out_folder / hist_filename}")

    # Compute cosine distance between reference and first prediction per sample.
    sample_ref_cosine = df_n.groupby('sample_id').apply(
        lambda group: compute_ref_cosine_distance_group(group.drop(columns='sample_id'), embedder)
    )

    for is_match in [True, False]:
        ids = sample_match[sample_match == is_match].index

        scatter_df = pd.DataFrame({
            'dispersion': sample_dispersion.loc[ids],
            'cosine_distance': sample_ref_cosine.loc[ids]
        }).dropna()

        with plot_scatter(scatter_df, n, is_match=is_match) as px:
            scatter_filename = f"dispersion_vs_cosdist__is_match={is_match}__use_chiral={use_chiral}__use_sum_formula={use_sum_formula}__top-{n}.png"
            px.f.savefig(out_folder / scatter_filename, dpi=300)
            print(f"Saved plot to {out_folder / scatter_filename}")


def process_translation(translation_file: Path, use_chiral: bool = True, use_sum_formula: bool = False):
    df, top_n_accuracy, out_folder = load_data(translation_file, use_chiral, use_sum_formula)
    embedder = ChemBERTaEmbedder(use_cuda=False)
    add_embeddings(df, embedder)
    # Process desired top-n predictions (e.g., top-6)
    for n in [3, 6]:
        process_top_n(n, df, embedder, top_n_accuracy, out_folder, use_chiral, use_sum_formula)


def main():
    translations = list(Path(__file__).parent.glob("b_*/**/translation/*.txt.json"))

    translations = [
        t for t in translations
        if (("_100000_" in t.name) or ("_250000_" in t.name))
           and ("_n10000." in t.name)
    ]

    for translation_file in translations:
        for use_chiral, use_sum_formula in product([True, False], [True, False]):
            process_translation(translation_file, use_chiral=use_chiral, use_sum_formula=use_sum_formula)


if __name__ == "__main__":
    main()
