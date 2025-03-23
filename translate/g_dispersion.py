import functools
import json

import numpy as np
import pandas as pd

from contextlib import contextmanager
from pathlib import Path
from parse import parse
from plox import Plox
from sklearn.metrics.pairwise import cosine_distances

from u_utils import canon_or_none, mol_formula_or_none, is_match_while_not_none


def get_embedder():
    from z_chembedding import ChemBERTaEmbedder
    return ChemBERTaEmbedder(use_cuda=False)


def load_data(*, translation_file: Path, use_chiral: bool, use_sum_formula: bool):
    print(f"Reading file {translation_file.relative_to(Path(__file__).parent)}")
    out_folder = Path(__file__).with_suffix('') / translation_file.relative_to(Path(__file__).parent)
    out_folder.mkdir(parents=True, exist_ok=True)

    pattern = (
        "{tgt_val_file}"
        "__model_step_{model_step:d}"
        "__n_best={n_best:d}"
        "__beam_size={beam_size:d}"
        ".txt.json"
    )
    parsed = parse(pattern, translation_file.name)
    print(f"Extracted parameters: {parsed.named}")

    with translation_file.open('r') as fd:
        data = json.load(fd)

    # Subsample for debugging
    # data = data[:100]

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
        print(f"Samples left after filtering hypotheses by sum formula: {df.sample_id.nunique()}")

        # print(df.to_markdown())

    assert df.n.min() == 1

    top_n_accuracy = {
        int(n): float(
            df[df.n <= n].groupby('sample_id').is_match.any()
            .reindex(df['sample_id'].unique(), fill_value=False).mean()
        )
        for n in sorted(df.n.dropna().unique())
    }
    print(top_n_accuracy)

    return (df, top_n_accuracy, out_folder)


def compute_dispersion_for_group(group: pd.DataFrame):
    if len(group) <= 1:
        return np.nan

    return np.mean(cosine_distances(list(group['embedding'])))


def compute_ref_cosine_distance_group(group: pd.DataFrame):
    embedder = get_embedder()

    [ref] = group['ref'].unique()

    if ref is None:
        return np.nan

    if not (group.n == 1).any():
        return np.nan

    [ref_emb] = embedder.embed([ref])
    [pred_emd] = group[group.n == 1]['embedding']

    return cosine_distances([ref_emb, pred_emd])[0, 1]


@contextmanager
def plot_histogram(dispersions: pd.DataFrame, top_n_accuracy: dict, top_n: int):
    with Plox({'figure.figsize': (10, 6)}) as px:
        bins = np.linspace(0, 0.2, 2 ** 6 + 1)

        matched = dispersions['dispersion'][dispersions['is_topn_match']]
        unmatched = dispersions['dispersion'][~dispersions['is_topn_match']]

        px.a.hist(matched, bins=bins, alpha=0.6, label=f"Samples with top-{top_n} match", color="tab:blue")
        px.a.hist(unmatched, bins=bins, alpha=0.6, label=f"Samples without top-{top_n} match", color="tab:red")

        px.a.set_xlabel("Average mutual cosine distance of predictions")
        px.a.set_ylabel("Number of samples")
        px.a.set_title(f"Dispersion of top-{top_n} predictions (accuracy: {top_n_accuracy[top_n]:.2%})")
        px.a.legend()
        px.a.grid(True, lw=0.5, alpha=0.5)

        yield px


@contextmanager
def plot_scatter(scatter_df, top_n, is_topn_match, color):
    with Plox({'figure.figsize': (10, 6)}) as px:
        # scatter_df = scatter_df[scatter_df['cosine_distance'] > 0]
        (xx, yy) = (scatter_df['dispersion'], scatter_df['ref_to_first'])
        px.a.set_xlim(-0.02, 0.52)
        px.a.set_ylim(-0.02, 1.02)
        px.a.scatter(xx, yy, alpha=0.3, s=5, edgecolors='none', color=color)
        px.a.set_xlabel("Average mutual cosine distance of predictions")
        px.a.set_ylabel("Cosine distance of reference to first prediction")
        px.a.set_title(f"Dispersion vs. cosine distance (top-{top_n} predictions, {is_topn_match = })")
        px.a.grid(True, lw=0.5, alpha=0.5)
        yield px


def process_top_n(
        top_n: int,
        df: pd.DataFrame,
        top_n_accuracy: dict,
        out_folder: Path,
        use_chiral: bool,
        use_sum_formula: bool,
):
    print(f"Processing top-{top_n} predictions")

    datafile = out_folder / f"dispersions__use_chiral={use_chiral}__use_sum_formula={use_sum_formula}__top-{top_n}.tsv.gz"

    if datafile.exists():
        print(f"Not recomputing dispersion data, already exists: {datafile}")

        dispersions = pd.read_csv(
            datafile,
            compression='gzip',
            sep='\t',
            index_col=0,
        )
    else:
        print(f"Computing embeddings for {len(df)} predictions")

        embedder = get_embedder()

        if 'embedding' not in df.columns:
            df['embedding'] = df['pred'].apply(lambda s: embedder.embed([s])[0] if s is not None else None)

        # Note, this step filters predictions with the wrong sum formula if this was enabled in `load_data`
        df_n = df[df.n <= top_n].copy()

        dispersions = pd.DataFrame({
            'is_topn_match': df_n.groupby('sample_id').is_match.any(),
            'dispersion': df_n.groupby('sample_id').apply(
                compute_dispersion_for_group,
                include_groups=False,
            ),
            'ref_to_first': df_n.groupby('sample_id').apply(
                functools.partial(compute_ref_cosine_distance_group, embedder=embedder),
                include_groups=False,
            )
        })

        dispersions.to_csv(
            datafile,
            compression='gzip',
            sep='\t',
        )

    with plot_histogram(dispersions, top_n_accuracy=top_n_accuracy, top_n=top_n) as px:
        hist_file = datafile.with_name(f"{datafile.name}-disperion_hist.png")
        px.f.savefig(hist_file, dpi=300)
        print(f"Saved plot to {hist_file}")

    for is_topn_match in [True, False]:
        scatter_df = dispersions[dispersions['is_topn_match'] == is_topn_match]

        color = "tab:blue" if is_topn_match else "tab:red"

        with plot_scatter(scatter_df, top_n=top_n, is_topn_match=is_topn_match, color=color) as px:
            scat_file = datafile.with_name(f"{datafile.name}-disperion_hist__is_topn_match={is_topn_match}.png")
            px.f.savefig(scat_file, dpi=300)
            print(f"Saved plot to {scat_file}")


def process_translation(translation_file: Path, **params):
    print(f"Processing translation file {translation_file}")
    (df, top_n_accuracy, out_folder) = load_data(translation_file=translation_file, **params)

    # Process desired top-n predictions (e.g., top-6)
    for n in [3, 6]:
        process_top_n(n, df, top_n_accuracy, out_folder, **params)


def main():
    translations = list(Path(__file__).parent.glob("b_*/**/translation/*.txt.json"))

    translations = [
        t for t in translations
        if (("_100000_" in t.name) or ("_250000_" in t.name))
           and ("_n10000." in t.name)
    ]

    for translation_file in translations:
        use_chiral = True
        use_sum_formula = True
        process_translation(translation_file, use_chiral=use_chiral, use_sum_formula=use_sum_formula)


if __name__ == "__main__":
    main()
