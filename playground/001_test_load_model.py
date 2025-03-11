
import numpy as np
import torch

from pathlib import Path


model_files = sorted(Path('.').parent.parent.glob("results/001_big/**/*.pt"))
print(*model_files, sep="\n")

model_file = max(model_files, key=lambda p: p.stat().st_mtime).resolve()

import torch
from onmt.model_builder import build_base_model
from onmt.utils.parse import ArgumentParser
from onmt.translate.translator import Translator
from onmt.inputters.inputter import dict_to_vocabs
from ctranslate2.converters.opennmt_tf import _load_vocab
from onmt.transforms import make_transforms

# cf.
# https://github.com/OpenNMT/OpenNMT-py/blob/97111d97551c24857076a4102eabdb468b35cff4/onmt/encoders/transformer.py#L19

checkpoint_path = 'model_step_10000.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

model_opts = ArgumentParser.ckpt_model_opts(checkpoint['opt'])

# fields = get_fields('text', 0)

vocabs = checkpoint.get('vocab', None)

# Convert vocab lists to dictionaries
vocabs["src"] = {word: idx for idx, word in enumerate(vocabs["src"])}
vocabs["tgt"] = {word: idx for idx, word in enumerate(vocabs["tgt"])}


model = build_base_model(model_opts, vocabs=vocabs)
model.eval()  # Set the model to evaluation mode


translator = Translator(model=model, vocabs=dict_to_vocabs(vocabs), report_score=False)

#

from onmt.translate import GNMTGlobalScorer

global_scorer = GNMTGlobalScorer(alpha=0.7, beta=0.0, length_penalty='avg', coverage_penalty='none')
translator = Translator(model=model, vocabs=dict_to_vocabs(vocabs), global_scorer=global_scorer)

input_data = np.asarray([["Hello", "world", "!"], ["How", "are", "you", "?"]])
translated = translator.translate_batch(input_data, attn_debug=False)

#

from onmt.inputters.inputter import T
dataset = TextDataset(examples=examples, fields=translator.fields, filter_pred=lambda x: True)

#

from onmt.translate.translator import build_translator
from argparse import Namespace

opt = Namespace(
    models=[str(model_file)],
    alpha=0.7,
    beta=0.0,
    beam_size=5,
    # Add other necessary options here
)

translator = build_translator(opt, report_score=False)
