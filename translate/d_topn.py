import json
from itertools import product

import numpy as np
import pandas as pd

from pathlib import Path
from parse import parse
from plox import Plox


from u_utils import canon_or_none, mol_formula_or_none, is_match_while_not_none


def process_translation(
        translation_file: Path,
        use_chiral: bool = True,
        use_sum_formula: bool = False,
):
    print(f"Reading file {translation_file.relative_to(Path(__file__).parent)}")

    out_folder = Path(__file__).with_suffix('') / translation_file.relative_to(Path(__file__).parent)
    assert out_folder.is_dir() or not out_folder.exists()
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
            'n': (n + 1)  # Capture index of hypothesis
        }
        for (i, sample) in enumerate(data)
        for (n, hyp) in enumerate(sample['hyps'])
    ])

    # Sanity check: cannot have 'is_match' if not 'sum_formula_match'
    assert not df[(~df['sum_formula_match']) & df['is_match']].any().any()

    if use_sum_formula:
        df['n'] = (
            df.groupby('sample_id')['sum_formula_match']
            .cumsum()  # Compute cumulative sum within each sample_id group
            .where(df['sum_formula_match'])  # Keep only where sum_formula_match is True, else NaN
        )

        df['n'] = df['n'].astype('Int64')  # Uses Pandas nullable integer type

        # How many samples are left? count different sample_ids:
        print(f"Original number of samples: {len(data)}")
        print(f"Number of samples left after filtering by sum formula: {df.sample_id.nunique()}")

    assert df.n.min() == 1

    top_n_accuracy = {
        n: (
            df[df.n <= n]  # Keep only top-n predictions
            .groupby('sample_id')
            .is_match
            .any()  # Check if at least one match exists
            .reindex(df['sample_id'].unique(), fill_value=False)  # Fill missing sample_ids with False
            .mean()  # Compute fraction of samples with a correct match
        )
        for n in sorted(df.n.dropna().unique())
    }
    print(top_n_accuracy)

    for n in sorted(df.n.dropna().unique()):
        print(f"Processing top-{n} predictions")

        df_n = df[df.n <= n].copy()

        filename = f"prediction_score_hist__use_chiral={use_chiral}__use_sum_formula={use_sum_formula}__top-{n}.png"

        (score_a, score_b) = (-0.5, 0)

        # Define different bins for correct and wrong predictions
        bins_correct = np.linspace(score_a, score_b, num=65)  # 65 bins for correct
        bins_wrong = np.linspace(score_a, score_b, num=49)  # 49 bins for incorrect

        # Apply binning separately
        df_n['bin'] = df_n.apply(
            lambda row: pd.cut([row['score']], bins=bins_correct if row['is_match'] else bins_wrong, labels=False)[0],
            axis=1,
        )

        # Convert MultiIndex groupby to a DataFrame with explicit columns
        grouped = df_n.groupby(['bin', 'n', 'is_match']).size().reset_index(name='count')

        with Plox({'figure.figsize': (10, 6)}) as px:
            for (is_match, color, bins) in zip([False, True], ['tab:red', 'tab:blue'], [bins_wrong, bins_correct]):
                subset = grouped[grouped['is_match'] == is_match]

                # Compute bin centers and bar width
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bar_width = np.diff(bins).mean()  # Set width as the average bin width

                bin_mapping = {i: bin_centers[i] for i in range(len(bin_centers))}  # Map bin indices to bin centers

                # Pivot so each bin gets a stacked bar
                pivoted = subset.pivot(index='bin', columns='n', values='count').fillna(0)
                pivoted.index = pivoted.index.map(bin_mapping)  # Map bin indices to actual score values

                bottom = np.zeros(len(pivoted))  # Track stacking levels
                for (c, n_col) in enumerate(pivoted.columns):
                    px.a.bar(
                        x=pivoted.index,  # Now using real bin centers
                        height=pivoted[n_col],
                        bottom=bottom,  # Stack bars
                        width=bar_width * 0.8,  # Adjust width slightly to avoid full overlap
                        label=f"{'correct' if is_match else 'wrong'} at n={n_col}",
                        alpha=0.5 + min(0.5, len(pivoted.columns) / 10) - (0.5 / len(pivoted.columns) * c),
                        color=color,
                        edgecolor='black',
                        lw=0.3,
                    )
                    bottom += pivoted[n_col].values  # Update stacking level

            px.a.set_xticks([x for x in np.linspace(-1, 0, 11) if (score_a <= x <= score_b)])
            px.a.set_xticklabels([f"{x:.1f}" for x in px.a.get_xticks()])
            px.a.set_xlim(score_a - (score_b - score_a) * 0.05, score_b + (score_b - score_a) * 0.05)

            px.a.set_ylim(0, 1.1 * df_n.groupby(['bin', 'is_match']).count().score.max(skipna=True))

            px.a.set_xlabel("Prediction score (log-likelihood according to the model)")
            px.a.set_yticks([])
            px.a.grid(True, lw=0.5, alpha=0.5, zorder=-100)

            px.a.set_title(f"Inferred SMILES (top-{n} accuracy: {top_n_accuracy[n]:.2%})")

            # Extract and process legend handles & labels
            (handles, labels) = px.a.get_legend_handles_labels()
            # Reverse order
            (handles, labels) = (handles[::-1], labels[::-1])
            # Swap first half with second half
            mid = len(handles) // 2
            handles = handles[mid:] + handles[:mid]
            labels = labels[mid:] + labels[:mid]
            # Apply the updated legend
            px.a.legend(handles, labels, loc='upper left', ncol=2, fontsize="small")

            params_to_show = {
                'use_chiral': use_chiral,
                'use_sum_formula': use_sum_formula,
                **parsed.named,
            }

            # Convert dictionary to formatted string
            named_text = "\n".join([f"{key}: {value}" for key, value in params_to_show.items()])

            # Get axis limits for precise positioning
            (x_min, x_max) = px.a.get_xlim()
            (y_min, y_max) = px.a.get_ylim()

            # Add text to the lower-left corner of the axis
            px.a.text(
                x_min + (x_max - x_min) * 0.02,  # Slightly inset from the left edge
                y_min + (y_max - y_min) * 0.02,  # Slightly inset from the bottom edge
                named_text,
                fontsize=8,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none')
            )

            px.f.savefig(out_folder / filename, dpi=300)

            print(f"Saved plot to {out_folder / filename}")


def main():
    translations: list[Path] = list(Path(__file__).parent.glob("work/**/translation/*.txt.json"))

    translations = [
        translation_file
        for translation_file in translations
        if ("_100000_" in translation_file.name) or ("_250000_" in translation_file.name)
    ]

    for translation_file in translations:
        for (use_chiral, use_sum_formula) in product([True, False], [True, False]):
            process_translation(translation_file, use_chiral=use_chiral, use_sum_formula=use_sum_formula)


if __name__ == "__main__":
    main()
