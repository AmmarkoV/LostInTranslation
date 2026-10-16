# LostInTranslation

Round-trip translator: takes text in a source language, translates it into an
intermediate language and back, and shows both steps. Translation is local
neural MT via [Argos Translate](https://github.com/argosopentech/argos-translate)
— no API keys, runs on CPU — behind a [Gradio](https://gradio.app) UI.

## Setup

```bash
./scripts/setup.sh
```

Creates `venv/`, installs Gradio and Argos Translate (with CPU-only PyTorch),
and pre-downloads the English↔German models so the first launch is instant.
Needs network access; expect roughly 1.7 GB for the venv plus the models.

## Run

```bash
venv/bin/python server.py                      # http://127.0.0.1:7860
venv/bin/python server.py --port 7861
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Interface to bind |
| `--port` | `7860` | Port to serve on |
| `--fetch-models` | off | Download the default models, then exit (used by `setup.sh`) |

## Using it

- **Source language** and **Intermediate language** are dropdowns covering the
  ~50 languages in the Argos package index.
- **1. Source text** → *Translate round trip* fills **2. Intermediate text**
  and **3. Back in the source language**. Pressing Enter in box 1 does the same.
- Box 2 is editable. Fix the intermediate by hand, then use
  *Re-translate intermediate → source* to redo only the second step.
- **Undo** restores the three text boxes to their contents before the last
  translation; **Clear** empties them. Neither touches the language dropdowns.
- **Remove line breaks from result** rewrites box 3 as one line, replacing the
  line breaks Argos keeps with spaces. It applies to the next translation and to
  whatever is already in box 3 when you tick it; unticking does not put the
  breaks back.
- **Copy result** copies box 3 to the clipboard — the copy icon in the corner of
  the box does the same.

## Notes

- Argos only ships `X → English` and `English → X` packages, so any other pair
  is pivoted through English automatically (`el → de` runs `el → en → de`).
  Pivoting costs quality compared with a direct pair, and drift on the return
  hop is normal for back-translation.
- Models live in `~/.local/share/argos-translate/` and are shared with other
  Argos installs on the machine. A language you have not used before is
  downloaded the first time you select it (tens to hundreds of MB), which
  blocks that one translation.

## SynthID-Text watermark scanner

A second, separate server: `synthid_server.py` scores pasted text with the Mean
and Weighted Mean detectors from
[synthid-text](https://github.com/google-deepmind/synthid-text).

**It does not tell you whether text is AI-generated.** Detection re-derives the
watermark's g-values from the text's *token ids*, so it only recognises a
watermark applied with the same tokenizer and the same keys. This server uses
GPT-2's tokenizer and the library's public reference `DEFAULT_WATERMARKING_CONFIG`
(30 keys, `ngram_len` 5). Gemini's production keys are not published, so text
from Gemini — or ChatGPT, or anything else — scores as unwatermarked. Upstream
also states the library is a reference implementation, "not intended for
production use".

### Setup

```bash
./scripts/setup-synthid.sh
```

Creates `venv-synthid/` — a separate venv on purpose, because `synthid-text`
pins `torch==2.4.0`, `numpy==1.26.0` and `jax[cuda]`, which conflict with the
translator's torch/numpy. The script installs CPU torch and CPU jax and then
`synthid-text` with `--no-deps`, which avoids the multi-GB CUDA wheels; expect
roughly 1.5 GB. Only the GPT-2 *tokenizer* (~2 MB) is downloaded — the model
weights are never needed, since detection reads token ids rather than logits.

### Run

```bash
venv-synthid/bin/python synthid_server.py            # http://127.0.0.1:7861
venv-synthid/bin/python synthid_server.py --port 7862
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Interface to bind |
| `--port` | `7861` | Port to serve on |
| `--fetch-tokenizer` | off | Download the tokenizer, then exit (used by `setup-synthid.sh`) |

### Reading the score

0.5 is the expectation for unwatermarked text; higher is more consistent with
the reference watermark. There is no calibrated threshold — upstream says to
pick one from the score distribution for your own use case and target false
positive rate.

The scanner reports how many n-grams it could actually score. N-grams are
dropped when they run past an end-of-sequence token or when their context has
already been seen, because a repeated context is not watermarked and carries no
signal. Short or repetitive text therefore leaves very few usable n-grams, and
the score swings far from 0.5 on noise alone — a repetitive 16-token string
scores 0.58 off 2 n-grams. Below 50 scorable n-grams the UI flags the score as
noise.

For reference, over 300 tokens the detectors separate cleanly: watermarked text
scores ~0.68 (Mean) against ~0.51 for unwatermarked.
