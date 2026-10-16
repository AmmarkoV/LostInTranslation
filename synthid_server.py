#!/usr/bin/env python3
"""SynthID-Text watermark scanner.

Scores pasted text with the Mean and Weighted Mean detectors from
google-deepmind/synthid-text, behind a Gradio UI.

Detection re-derives the watermark's g-values from the text's token ids, so it
only recognises text watermarked with the same tokenizer and the same keys.
This uses GPT-2's tokenizer and the reference DEFAULT_WATERMARKING_CONFIG that
ships with the library -- it cannot detect Gemini's production watermark, whose
keys are not public.
"""

import argparse

import gradio as gr
import torch
import transformers
from synthid_text import detector_mean, logits_processing, synthid_mixin

TOKENIZER_NAME = "gpt2"

# Only used when generating; the constructor requires them but detection never
# reads them.
UNUSED_TEMPERATURE = 1.0
UNUSED_TOP_K = 40


def build_processor() -> logits_processing.SynthIDLogitsProcessor:
    """The reference watermarking config, pinned to CPU."""
    config = dict(synthid_mixin.DEFAULT_WATERMARKING_CONFIG)
    config["device"] = torch.device("cpu")
    return logits_processing.SynthIDLogitsProcessor(
        **config, temperature=UNUSED_TEMPERATURE, top_k=UNUSED_TOP_K
    )


def scan(text: str, tokenizer, processor) -> tuple[str, str]:
    """Score one piece of text. Returns (verdict markdown, detail markdown)."""
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    num_tokens = input_ids.shape[1]
    if num_tokens < processor.ngram_len:
        return (
            f"Need at least {processor.ngram_len} tokens to score; "
            f"this is {num_tokens}.",
            "",
        )

    # The detection recipe from the library's notebook: drop n-grams that run
    # past an end-of-sequence token, and n-grams whose context was already seen
    # (a repeated context is not watermarked, so its g-values carry no signal).
    eos_mask = processor.compute_eos_token_mask(
        input_ids=input_ids, eos_token_id=tokenizer.eos_token_id
    )[:, processor.ngram_len - 1 :]
    repetition_mask = processor.compute_context_repetition_mask(input_ids=input_ids)
    mask = (repetition_mask * eos_mask).numpy()
    g_values = processor.compute_g_values(input_ids=input_ids).numpy()

    scorable = int(mask.sum())
    if scorable == 0:
        return (
            "No scorable n-grams — every context in this text is a repeat.",
            f"{num_tokens} tokens, {mask.shape[1]} n-grams, 0 usable.",
        )

    # weighted_mean_score multiplies g_values in place, so it gets a copy.
    mean = float(detector_mean.mean_score(g_values, mask)[0])
    weighted = float(
        detector_mean.weighted_mean_score(g_values.copy(), mask)[0]
    )

    verdict = (
        f"### Mean {mean:.4f} · Weighted Mean {weighted:.4f}\n"
        "0.5 is the expectation for unwatermarked text. Higher is more "
        "consistent with the reference watermark."
    )
    if scorable < 50:
        verdict += (
            f"\n\n**Only {scorable} scorable n-grams — treat this score as "
            "noise.** A short or repetitive passage swings well away from 0.5 "
            "on its own."
        )
    detail = (
        f"{num_tokens} tokens · {mask.shape[1]} n-grams · {scorable} scorable "
        f"(after dropping {mask.shape[1] - scorable} repeated-context or "
        f"post-EOS) · depth {g_values.shape[-1]}"
    )
    return verdict, detail


def build_ui(tokenizer, processor) -> gr.Blocks:
    with gr.Blocks(title="SynthID-Text scanner") as demo:
        gr.Markdown(
            "## SynthID-Text watermark scanner\n"
            "Scores text with the Mean and Weighted Mean detectors from "
            "[synthid-text](https://github.com/google-deepmind/synthid-text).\n\n"
            "**This does not tell you whether text is AI-generated.** It only "
            f"recognises a watermark applied with the `{TOKENIZER_NAME}` "
            "tokenizer and the library's public reference keys. Text from "
            "Gemini, ChatGPT or any other product will score as unwatermarked, "
            "because their keys are not public. The library itself is a "
            "reference implementation, not for production use."
        )

        text = gr.Textbox(label="Text to scan", lines=12, placeholder="Paste text here…")
        scan_btn = gr.Button("Scan", variant="primary")
        verdict = gr.Markdown()
        detail = gr.Markdown()

        scan_btn.click(lambda t: scan(t, tokenizer, processor), text, [verdict, detail])
        text.submit(lambda t: scan(t, tokenizer, processor), text, [verdict, detail])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    parser.add_argument("--port", type=int, default=7861, help="port to serve on")
    parser.add_argument(
        "--fetch-tokenizer",
        action="store_true",
        help="download the tokenizer if missing, then exit (used by scripts/setup-synthid.sh)",
    )
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    if args.fetch_tokenizer:
        print(f"Tokenizer {TOKENIZER_NAME} is installed.")
        raise SystemExit(0)

    build_ui(tokenizer, build_processor()).launch(
        server_name=args.host, server_port=args.port
    )
