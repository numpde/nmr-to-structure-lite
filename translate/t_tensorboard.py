from pathlib import Path
from datetime import timedelta

from sympy.physics.units import nanosecond
from tensorboard.backend.event_processing import event_accumulator
from plox import Plox

translations: list[Path] = list(Path(__file__).parent.parent.glob(
    "results/**/run/tensorboard/**/events.out.tfevents.*"
))

translations = sorted([
    file for file in translations
    if "000_tiny" not in str(file)
])

output_folder = Path(__file__).with_suffix('')
output_folder.mkdir(parents=True, exist_ok=True)

figure_filename = output_folder / "validation_accuracy.png"

with figure_filename.with_suffix(".sources.txt").open(mode='w') as fd:
    for file in translations:
        print(file.relative_to(Path(__file__).parent.parent), file=fd)

with Plox() as px:
    markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'h', 'H', 'X', 'd', '1', '2', '3', '4', '+', 'x']

    for path in translations:
        ea = event_accumulator.EventAccumulator(str(path))
        ea.Reload()

        print(ea.Tags().get('scalars', []))

        train_acc = ea.Scalars('train/accuracy')

        valid_acc = ea.Scalars('valid/accuracy')
        steps = [e.step for e in valid_acc]
        values = [e.value for e in valid_acc]

        (t0, t1) = (train_acc[0].wall_time, train_acc[-1].wall_time)
        hours = round((t1 - t0) / 3600)

        label = path.relative_to(Path(__file__).parent.parent).parts[1]
        label = f"{label} (~{hours}h)"

        px.a.plot(steps, values, label=label, marker=markers.pop(0))

    px.a.set_title("Validation accuracy during training")

    px.a.set_ylim(min(*px.a.get_ylim()), 100)

    xlim = px.a.get_xlim()
    xticks = px.a.get_xticks()
    px.a.set_xticks(xticks)  # Explicitly fix tick positions
    px.a.set_xticklabels([f"{int(x / 1000):,}k" for x in xticks])
    px.a.set_xlim(*xlim)

    px.a.set_xlabel("Training step (number of batches)")
    px.a.set_ylabel("Validation accuracy, %")

    (hh, ll) = px.a.get_legend_handles_labels()
    px.a.legend(reversed(hh), reversed(ll), loc='lower right', fontsize=10)

    px.a.grid(True, lw=0.5, alpha=0.5)

    px.f.savefig(figure_filename, dpi=300)

