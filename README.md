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
