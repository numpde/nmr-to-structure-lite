import json

import pandas as pd

from pathlib import Path

from plox import Plox
from rdkit import Chem
from sklearn.manifold import TSNE

from rdkit.Chem import MolFromSmiles, Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from translate.e_descriptors_aux import compute_nmr_difficulty_descriptors
from u_utils import canon_or_none, mol_formula_or_none, is_match_while_not_none

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def train_decision_tree(X, y, max_depth=4, test_size=0.3, random_state=42):
    """
    Train a Decision Tree Classifier, compute accuracy, and return results in a dictionary.

    Returns:
    - results (dict): Dictionary containing the model, accuracy, and confusion matrix.
    """


    # Split into train/test sets
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train Decision Tree
    class_weight = 'balanced'
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, class_weight=class_weight, criterion='gini')
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Return results as a dictionary
    return {
        'model': clf,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'confusions': {
            'true_negative': cm[0, 0],
            'false_positive': cm[0, 1],
            'false_negative': cm[1, 0],
            'true_positive': cm[1, 1],
        }
    }




def process_translation(translation_file: Path):
    print(f"Reading file {translation_file.relative_to(Path(__file__).parent)}")

    use_chiral = True

    out_folder = Path(__file__).with_suffix('') / translation_file.relative_to(Path(__file__).parent)
    assert out_folder.is_dir() or not out_folder.exists()
    out_folder.mkdir(parents=True, exist_ok=True)

    with translation_file.open(mode='r') as fd:
        data = json.load(fd)

    df = pd.DataFrame(data=[
        {
            'ref': canon_or_none(sample['ref']),
            'is_match': is_match_while_not_none(
                canon_or_none(sample['ref'], use_chiral=use_chiral),
                canon_or_none(hyp['pred'], use_chiral=use_chiral),
            ),
        }
        for (i, sample) in enumerate(data)
        for (n, hyp) in enumerate(sample['hyps'], start=1)
    ])

    df = df.groupby('ref').is_match.any().reset_index()

    df_descriptors = df['ref'].apply(compute_nmr_difficulty_descriptors).apply(pd.Series)
    df = df.join(df_descriptors)

    # Sanity check: no missing values
    assert not df.isnull().any().any()

    # Train the decision tree and compute accuracy
    dt_settings = {'max_depth': 4}
    dt_results = train_decision_tree(X=df_descriptors, y=df["is_match"].astype(int), **dt_settings)

    print(f"DT:", dt_results)

    # Plot the confusion matrix
    with Plox() as px:
        cm = dt_results['confusion_matrix']
        px.a.matshow(cm, cmap='Blues')

        for i in range(2):
            for j in range(2):
                px.a.text(j, i, f"{cm[i, j]:,}", ha='center', va='center', color='black')

        px.a.set_xlabel('Predicted label')
        px.a.set_ylabel('True label')

        px.f.savefig(Path(__file__).with_suffix('.png'), dpi=300)

    exit()

    # Display accuracy
    print(f"Decision Tree Accuracy: {dtc_accuracy:.4f}")


    with Plox() as px:
        plot_tree(dtc_model, filled=True, feature_names=df_descriptors.columns, class_names=['No Match', 'Match'], ax=px.a)

        px.f.savefig(Path(__file__).with_suffix('.png'), dpi=300)



    exit()

    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    df_tsne = pd.DataFrame(tsne.fit_transform(df_descriptors), columns=["TSNE-1", "TSNE-2"], index=df.index)

    # out_folder = Path(__file__).with_suffix('') / translation_file.relative_to(Path(__file__).parent)

    with Plox() as px:
        # colors = df_descriptors[df_descriptors.columns[0]]
        colors = df['is_match']
        scatter = px.a.scatter(df_tsne["TSNE-1"], df_tsne["TSNE-2"], c=colors, cmap="viridis", alpha=0.7)
        # px.a.set_title()

        px.a.axis("off")

        px.f.savefig(Path(__file__).with_suffix('.png'), dpi=300)


def main():
    translations: list[Path] = list(Path(__file__).parent.glob("work/**/translation/*.txt.json"))

    translations = [
        translation_file
        for translation_file in translations
        if (("_100000_" in translation_file.name) or ("_250000_" in translation_file.name))
        and ("_n10000." in translation_file.name)
    ]

    print(f"Found {len(translations)} translation files: ", *translations, sep="\n")

    for translation_file in translations:
        process_translation(translation_file)
        exit()


if __name__ == "__main__":
    main()
