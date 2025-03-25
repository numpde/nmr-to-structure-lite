import json

import numpy as np
import pandas as pd

from contextlib import contextmanager
from pathlib import Path

from matplotlib import cm
from parse import parse
from plox import Plox
from scipy.stats import gaussian_kde


@contextmanager
def sfm_hist(df: pd.DataFrame, top_n: int = 10):
    sfm = df.groupby('sample_id').sum_formula_match.sum()

    with Plox() as px:
        px.a.hist(sfm, bins=range(0, top_n + 2), rwidth=0.8, align='left', color='tab:blue')

        px.a.set_xticks(range(0, top_n + 1))

        px.a.set_xlabel("Sum-formula matches")
        px.a.set_ylabel("Number of samples")

        px.a.set_title(f"Sum-formula matches among top-{top_n} predictions")

        px.a.grid(True, lw=0.5, alpha=0.5, zorder=-100)

        yield px


def process_translation(translation_file: Path):
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
            'ref': sample['ref'],
            'sample_id': i,
            'pred': hyp['pred'],
            'score': hyp['score'],
            'n': n  # Capture index of hypothesis
        }
        for (i, sample) in enumerate(data)
        for (n, hyp) in enumerate(sample['hyps'], start=1)
    ])

    print(df.head())

    if 'is_match' in df.columns:
        # Sanity check: cannot have 'is_match' if not 'sum_formula_match'
        assert not df[(~df['sum_formula_match']) & df['is_match']].any().any()

    assert df.n.min() == 1, "Check that we're 1-based"

    length_hist_filepath = out_folder / f"length_hist.png"
    print(f"Saving length histogram to {length_hist_filepath}")

    # Compute maximal length of the SMILES
    max_len = max(df['pred'].str.len().max(), df['ref'].str.len().max())

    # Round max_len to the next hundred
    max_len = int(np.ceil(max_len / 100) * 100)

    # Use this to limit the x-axis
    x_max = max_len - 100

    # Hypotheses enumerated in the dataset
    nn = df.n.dropna().unique()

    # Define colors using a gradient/colormap (from blue to red)
    colors = cm.coolwarm(np.linspace(0, 1, len(nn)))

    def plot_dist(all_smiles: pd.Series, **params):
        all_smiles = pd.Series(all_smiles)

        lengths = all_smiles.str.split(" ").apply(len)

        xx = np.linspace(0, max_len, 1 + (2 ** 10))

        kde = gaussian_kde(lengths, bw_method=0.05)
        yy = kde(xx)

        yy /= np.trapezoid(yy, xx)

        # use x_max
        yy = yy[xx <= x_max]
        xx = xx[xx <= x_max]

        px.a.plot(xx, yy, **params)

    with Plox({'figure.figsize': (10, 6)}) as px:
        plot_dist(df['ref'].unique(), label="reference", color='m', lw=1.5, alpha=0.8, zorder=10, ls='--')

        for (top_n, color) in zip(nn, colors):
            params = dict(
                label=f"hyp #{top_n}",
                alpha=(0.9 - 0.8 * (top_n / len(nn))),
                color=color,
                lw=2,
                zorder=(-top_n),
            )

            plot_dist(df[df.n == top_n].pred, **params)

        px.a.grid(True, lw=0.5, alpha=0.5, zorder=-100)

        px.a.set_xlabel("SMILES length")
        px.a.set_yticks([])

        # Set xticks/xlabels as multiples of 10
        # Use the current limits
        (x_min, x_max) = px.a.get_xlim()
        px.a.set_xticks([x for x in range(0, int(x_max) + 1, 10)])
        px.a.set_xticklabels([f"{x}" for x in px.a.get_xticks()])


        px.a.set_title("Length distribution of predicted SMILES")

        px.a.legend()

        params_to_show = {
            **parsed.named,
        }

        # Convert dictionary to formatted string
        named_text = "\n".join([f"{key}: {value}" for key, value in params_to_show.items()])

        # Get axis limits for precise positioning
        (x_min, x_max) = px.a.get_xlim()
        (y_min, y_max) = px.a.get_ylim()

        # Add text to the lower-left corner of the axis
        px.a.text(
            x_min + (x_max - x_min) * 0.02,
            y_max - (y_max - y_min) * 0.02,
            named_text,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor='white', alpha=0.4, edgecolor='none')
        )

        px.f.savefig(length_hist_filepath, dpi=300)
    #
    #
    # for n in sorted(df.n.dropna().unique()):
    #     print(f"Processing top-{n} predictions")
    #
    #     df_n = df[df.n <= n].copy()
    #
    #     filepath = out_folder / f"prediction_score_hist__use_chiral={use_chiral}__use_sum_formula={use_sum_formula}__top-{n}.png"
    #
    #     if filepath.exists():
    #         print(f"Skipping existing file {filepath}")
    #         continue
    #
    #     (score_a, score_b) = (-0.5, 0)
    #
    #     # Define different bins for correct and wrong predictions
    #     bins_correct = np.linspace(score_a, score_b, num=65)  # 65 bins for correct
    #     bins_wrong = np.linspace(score_a, score_b, num=49)  # 49 bins for incorrect
    #
    #     # Apply binning separately
    #     df_n['bin'] = df_n.apply(
    #         lambda row: pd.cut([row['score']], bins=bins_correct if row['is_match'] else bins_wrong, labels=False)[0],
    #         axis=1,
    #     )
    #
    #     # Convert MultiIndex groupby to a DataFrame with explicit columns
    #     grouped = df_n.groupby(['bin', 'n', 'is_match']).size().reset_index(name='count')
    #
    #     with Plox({'figure.figsize': (10, 6)}) as px:
    #         for (is_match, color, bins) in zip([False, True], ['tab:red', 'tab:blue'], [bins_wrong, bins_correct]):
    #             subset = grouped[grouped['is_match'] == is_match]
    #
    #             # Compute bin centers and bar width
    #             bin_centers = (bins[:-1] + bins[1:]) / 2
    #             bar_width = np.diff(bins).mean()  # Set width as the average bin width
    #
    #             bin_mapping = {i: bin_centers[i] for i in range(len(bin_centers))}  # Map bin indices to bin centers
    #
    #             # Pivot so each bin gets a stacked bar
    #             pivoted = subset.pivot(index='bin', columns='n', values='count').fillna(0)
    #             pivoted.index = pivoted.index.map(bin_mapping)  # Map bin indices to actual score values
    #
    #             bottom = np.zeros(len(pivoted))  # Track stacking levels
    #             for (c, n_col) in enumerate(pivoted.columns):
    #                 px.a.bar(
    #                     x=pivoted.index,  # Now using real bin centers
    #                     height=pivoted[n_col],
    #                     bottom=bottom,  # Stack bars
    #                     width=bar_width * 0.8,  # Adjust width slightly to avoid full overlap
    #                     label=f"{'correct' if is_match else 'wrong'} at n={n_col}",
    #                     alpha=0.5 + min(0.5, len(pivoted.columns) / 10) - (0.5 / len(pivoted.columns) * c),
    #                     color=color,
    #                     edgecolor='black',
    #                     lw=0.3,
    #                 )
    #                 bottom += pivoted[n_col].values  # Update stacking level
    #
    #         px.a.set_xticks([x for x in np.linspace(-1, 0, 11) if (score_a <= x <= score_b)])
    #         px.a.set_xticklabels([f"{x:.1f}" for x in px.a.get_xticks()])
    #         px.a.set_xlim(score_a - (score_b - score_a) * 0.05, score_b + (score_b - score_a) * 0.05)
    #
    #         px.a.set_ylim(0, 1.1 * df_n.groupby(['bin', 'is_match']).count().score.max(skipna=True))
    #
    #         px.a.set_xlabel("Prediction score (log-likelihood according to the model)")
    #         px.a.set_yticks([])
    #         px.a.grid(True, lw=0.5, alpha=0.5, zorder=-100)
    #
    #         px.a.set_title(f"Inferred SMILES (top-{n} accuracy: {top_n_accuracy[n]:.2%})")
    #
    #         # Extract and process legend handles & labels
    #         (handles, labels) = px.a.get_legend_handles_labels()
    #         # Reverse order
    #         (handles, labels) = (handles[::-1], labels[::-1])
    #         # Swap first half with second half
    #         mid = len(handles) // 2
    #         handles = handles[mid:] + handles[:mid]
    #         labels = labels[mid:] + labels[:mid]
    #         # Apply the updated legend
    #         px.a.legend(handles, labels, loc='upper left', ncol=2, fontsize="small")
    #
    #         params_to_show = {
    #             'use_chiral': use_chiral,
    #             'use_sum_formula': use_sum_formula,
    #             **parsed.named,
    #         }
    #
    #         # Convert dictionary to formatted string
    #         named_text = "\n".join([f"{key}: {value}" for key, value in params_to_show.items()])
    #
    #         # Get axis limits for precise positioning
    #         (x_min, x_max) = px.a.get_xlim()
    #         (y_min, y_max) = px.a.get_ylim()
    #
    #         # Add text to the lower-left corner of the axis
    #         px.a.text(
    #             x_min + (x_max - x_min) * 0.02,  # Slightly inset from the left edge
    #             y_min + (y_max - y_min) * 0.02,  # Slightly inset from the bottom edge
    #             named_text,
    #             fontsize=8,
    #             verticalalignment="bottom",
    #             horizontalalignment="left",
    #             bbox=dict(facecolor='white', alpha=0.4, edgecolor='none')
    #         )
    #
    #         px.f.savefig(filepath, dpi=300)
    #
    #         print(f"Saved plot to {filepath}")


def main():
    translations: list[Path] = list(Path(__file__).parent.glob("b_*/**/translation/*.txt.json"))

    translations = [
        translation_file
        for translation_file in translations
        if (("_100000_" in translation_file.name) or ("_250000_" in translation_file.name))
           # and ("_n10000." in translation_file.name)
    ]

    for translation_file in translations:
        process_translation(translation_file)


if __name__ == "__main__":
    main()
