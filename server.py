#!/usr/bin/env python3
"""Source -> pivot -> source round-trip translator.

Argos Translate (local neural MT, CPU, no API keys) behind a Gradio UI.
Both languages are chosen from dropdowns; the intermediate text is an editable
textbox, so you can fix it by hand and re-run only the second step.
"""

import argparse

import argostranslate.package
import argostranslate.translate
import gradio as gr

# Argos only ships X -> English and English -> X packages and pivots every
# other pair through English, so English is the hub.
HUB = "en"
DEFAULT_SOURCE = "en"
DEFAULT_PIVOT = "de"


def language_names() -> dict[str, str]:
    """Language code -> name, for every language the Argos index offers."""
    packages = argostranslate.package.get_available_packages()
    names = {p.from_code: p.from_name for p in packages}
    names.update({p.to_code: p.to_name for p in packages})
    return names


def ensure_package(from_code: str, to_code: str) -> None:
    """Install the Argos model for a language pair if it is not installed yet."""
    installed = argostranslate.translate.get_installed_languages()
    source = next((lang for lang in installed if lang.code == from_code), None)
    if source is not None and any(
        t.to_lang.code == to_code for t in source.translations_from
    ):
        return

    print(f"Downloading Argos model {from_code} -> {to_code} ...", flush=True)
    argostranslate.package.update_package_index()
    packages = argostranslate.package.get_available_packages()
    package = next(
        (p for p in packages if p.from_code == from_code and p.to_code == to_code),
        None,
    )
    if package is None:
        raise RuntimeError(f"No Argos package available for {from_code} -> {to_code}")
    argostranslate.package.install_from_path(package.download())


def ensure_language(code: str) -> None:
    """Make sure a language can reach the hub and be reached back from it."""
    if code == HUB:
        return
    ensure_package(code, HUB)
    ensure_package(HUB, code)


def translate(text: str, from_code: str, to_code: str) -> str:
    """Translate one step. Argos keeps the input's line breaks."""
    return argostranslate.translate.translate(text, from_code, to_code)


def build_ui() -> gr.Blocks:
    names = language_names()
    choices = [
        (f"{name} ({code})", code)
        for code, name in sorted(names.items(), key=lambda item: item[1])
    ]

    with gr.Blocks(title="Round-trip translator") as demo:
        gr.Markdown(
            "## Round-trip translator\n"
            "Translates the source into the intermediate language and back, then "
            "shows both steps. Edit the intermediate text and re-run the second "
            "step to steer the result."
        )

        # Text of the three boxes before the last action, so Undo can restore it.
        undo_state = gr.State(None)

        with gr.Row():
            source_lang = gr.Dropdown(
                choices=choices, value=DEFAULT_SOURCE, label="Source language"
            )
            pivot_lang = gr.Dropdown(
                choices=choices, value=DEFAULT_PIVOT, label="Intermediate language"
            )

        source = gr.Textbox(
            label="1. Source text", lines=8, placeholder="Paste text here…"
        )
        with gr.Row():
            round_trip_btn = gr.Button("Translate round trip", variant="primary")
            from_pivot_btn = gr.Button("Re-translate intermediate → source")
            undo_btn = gr.Button("Undo")
            clear_btn = gr.Button("Clear")

        intermediate = gr.Textbox(
            label="2. Intermediate text (editable — used by the button above)",
            lines=8,
        )
        result = gr.Textbox(
            label="3. Back in the source language", lines=8, buttons=["copy"]
        )
        status = gr.Markdown()

        def round_trip(source_text, intermediate_text, result_text, prev_snapshot,
                       source_code, pivot_code):
            if source_code == pivot_code:
                return (
                    intermediate_text,
                    result_text,
                    prev_snapshot,
                    "Pick two different languages for the round trip.",
                )
            if not source_text.strip():
                return (
                    intermediate_text,
                    result_text,
                    prev_snapshot,
                    "Nothing to translate — the source text is empty.",
                )
            snapshot = [source_text, intermediate_text, result_text]
            ensure_language(source_code)
            ensure_language(pivot_code)
            middle = translate(source_text, source_code, pivot_code)
            back = translate(middle, pivot_code, source_code)
            return (
                middle,
                back,
                snapshot,
                f"Translated {names[source_code]} → {names[pivot_code]} "
                f"→ {names[source_code]}.",
            )

        def from_pivot(source_text, intermediate_text, result_text, prev_snapshot,
                       source_code, pivot_code):
            if source_code == pivot_code:
                return (
                    result_text,
                    prev_snapshot,
                    "Pick two different languages for the round trip.",
                )
            if not intermediate_text.strip():
                return (
                    result_text,
                    prev_snapshot,
                    "Nothing to translate — the intermediate text is empty.",
                )
            snapshot = [source_text, intermediate_text, result_text]
            ensure_language(source_code)
            ensure_language(pivot_code)
            return (
                translate(intermediate_text, pivot_code, source_code),
                snapshot,
                f"Translated {names[pivot_code]} → {names[source_code]}.",
            )

        def undo(undo_state_value):
            if undo_state_value is None:
                return gr.update(), gr.update(), gr.update(), None, "Nothing to undo."
            return (*undo_state_value, None, "Restored the previous text.")

        def clear():
            return "", "", "", None, ""

        full_inputs = [source, intermediate, result, undo_state, source_lang, pivot_lang]
        round_trip_outputs = [intermediate, result, undo_state, status]
        round_trip_btn.click(round_trip, full_inputs, round_trip_outputs)
        source.submit(round_trip, full_inputs, round_trip_outputs)
        from_pivot_btn.click(from_pivot, full_inputs, [result, undo_state, status])
        undo_btn.click(
            undo, [undo_state], [source, intermediate, result, undo_state, status]
        )
        clear_btn.click(
            clear, None, [source, intermediate, result, undo_state, status]
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    parser.add_argument("--port", type=int, default=7860, help="port to serve on")
    parser.add_argument(
        "--fetch-models",
        action="store_true",
        help="download the default models if missing, then exit (used by scripts/setup.sh)",
    )
    args = parser.parse_args()

    ensure_language(DEFAULT_SOURCE)
    ensure_language(DEFAULT_PIVOT)

    if args.fetch_models:
        print(f"Argos models for {DEFAULT_SOURCE} <-> {DEFAULT_PIVOT} are installed.")
        raise SystemExit(0)

    build_ui().launch(server_name=args.host, server_port=args.port)
