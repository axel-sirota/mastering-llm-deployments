"""
Lab 6: Post-Training Quantization + Benchmarking
=================================================

GOAL
----
Apply three post-training quantization (PTQ) techniques to t5-small and measure
how each affects:
  1. Inference latency  — wall-clock seconds to summarise one dialogue
  2. Summarisation quality — real BLEU and ROUGE-L scores vs. reference summaries

WHY POST-TRAINING QUANTIZATION?
--------------------------------
By default, PyTorch models store every weight as a 32-bit float (fp32).
That gives you ~4 bytes per parameter. PTQ reduces that precision *after*
training — no retraining required — trading a small accuracy loss for large
gains in speed and memory:

  fp32      — baseline; 4 bytes/param, highest precision
  bfloat16  — 2 bytes/param; same exponent range as fp32 (so no overflow),
               halved mantissa precision; negligible accuracy loss; native
               acceleration on A100, H100, Apple Silicon (MPS)
  int8      — 1 byte/param; ~4× smaller than fp32; 2–3× faster on CPU via
               dynamic quantization (weights stored as int8, dequantised to
               fp32 per-operation at inference time)

This script runs all three modes and prints a comparison table so you can
see the real-world latency/accuracy tradeoff.

USAGE
-----
    python starter_code.py
    (no arguments needed — all three modes run automatically)
"""

import time

# ---------------------------------------------------------------------------
# torch
# The core PyTorch library.  We use:
#   torch.no_grad()            — context manager that disables gradient
#                                tracking during inference (saves memory, ~20%
#                                faster because autograd bookkeeping is skipped)
#   torch.ao.quantization      — the "Architecture Optimization" sub-package;
#                                contains quantize_dynamic() for dynamic PTQ
#   torch.nn.Linear            — the layer class we target for int8 replacement
#   torch.qint8                — the int8 quantised dtype
#   torch.bfloat16             — the 16-bit brain-float dtype
# ---------------------------------------------------------------------------
import torch

# ---------------------------------------------------------------------------
# transformers (Hugging Face)
# Provides pre-trained models and tokenizers.
#
# AutoTokenizer
#   Loads the correct tokenizer for a given checkpoint automatically.
#   A tokenizer converts raw text → integer token IDs that the model
#   understands, and converts model output token IDs back to text.
#   Key methods used here:
#     __call__(text, ...)  → dict of tensors {input_ids, attention_mask, ...}
#     decode(ids, ...)     → string
#
# AutoModelForSeq2SeqLM
#   Loads an encoder-decoder model (like T5) that maps one sequence
#   (the dialogue) to another (the summary).
#   Key method used here:
#     generate(**inputs)   → tensor of output token IDs (beam search by default)
# ---------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# evaluate (Hugging Face)
# A library for computing NLP evaluation metrics in a consistent way.
# We load two metrics:
#
# BLEU  (Bilingual Evaluation Understudy)
#   Measures n-gram precision: what fraction of 1–4 word sequences in the
#   prediction also appear in the reference.
#   Score range: 0.0 (no overlap) → 1.0 (exact match).
#   API: evaluate.load("bleu").compute(predictions=[...], references=[[...]])
#   NOTE: references must be a list-of-lists — each prediction can have
#         multiple references (we have one, so wrap: [[ref1], [ref2], ...]).
#
# ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
#   Measures recall of n-grams and subsequences.
#   ROUGE-L specifically measures the Longest Common Subsequence (LCS) between
#   prediction and reference — it captures sentence-level structure better
#   than pure n-gram matching.
#   Score range: 0.0 → 1.0.
#   API: evaluate.load("rouge").compute(predictions=[...], references=[...])
#   NOTE: references here is a plain list (one string per prediction).
# ---------------------------------------------------------------------------
import evaluate


# ---------------------------------------------------------------------------
# Sample dialogue/reference pairs for accuracy evaluation.
# Using five hardcoded examples so the lab runs without downloading a dataset.
# Format mirrors DialogSum (used in Lab 3):
#   "dialogue"  — the raw conversation, prefixed with "summarize: " so T5
#                 knows which task to perform (T5 is a multi-task model that
#                 uses text prefixes to select its behaviour)
#   "reference" — a human-written summary we compare model output against
# ---------------------------------------------------------------------------
SAMPLE_DATA = [
    {
        "dialogue": (
            "summarize: #Person1#: Hey, are we still meeting for lunch tomorrow? "
            "#Person2#: Yes, definitely! How about noon at the Italian place on Main? "
            "#Person1#: Sounds perfect. I'll see you then."
        ),
        "reference": (
            "#Person1# and #Person2# confirm their lunch plans for noon tomorrow "
            "at the Italian restaurant on Main Street."
        ),
    },
    {
        "dialogue": (
            "summarize: #Person1#: I can't believe how much traffic there was this morning. "
            "#Person2#: I know, it took me an hour just to get to the office. "
            "#Person1#: Maybe we should start carpooling. #Person2#: That's a great idea."
        ),
        "reference": (
            "#Person1# and #Person2# complain about heavy morning traffic "
            "and agree to start carpooling."
        ),
    },
    {
        "dialogue": (
            "summarize: #Person1#: Did you finish the quarterly report? "
            "#Person2#: Almost, I just need to add the final charts. "
            "#Person1#: The deadline is tomorrow morning. "
            "#Person2#: I'll have it ready tonight."
        ),
        "reference": (
            "#Person1# checks on the quarterly report. "
            "#Person2# says they will finish it tonight before the morning deadline."
        ),
    },
    {
        "dialogue": (
            "summarize: #Person1#: My laptop keeps freezing. Can IT help? "
            "#Person2#: Sure, have you tried restarting it first? "
            "#Person1#: Yes, multiple times. "
            "#Person2#: I'll send a ticket to IT for you."
        ),
        "reference": (
            "#Person1# reports laptop freezing issues. "
            "#Person2# will submit an IT support ticket after basic troubleshooting."
        ),
    },
    {
        "dialogue": (
            "summarize: #Person1#: What do you think about the new office layout? "
            "#Person2#: I like the open space, but it's a bit noisy. "
            "#Person1#: We could request some quiet zones. "
            "#Person2#: Good idea, let's email the facilities team."
        ),
        "reference": (
            "#Person1# and #Person2# discuss the new office layout and "
            "agree to request quiet zones from the facilities team."
        ),
    },
]


# ===========================================================================
# EXERCISE 1 of 3
# ===========================================================================
def quantize_model(model, mode: str):
    """Apply a post-training quantization technique to a loaded model.

    The model arrives already loaded from disk in fp32 (the default).
    Your job is to transform it according to `mode` and return it.

    Modes
    -----
    "fp32"
        No change — this is the baseline for comparison.

    "bfloat16"
        Cast every weight tensor from 32-bit float to 16-bit "brain float".
        Use:  model = model.to(torch.bfloat16)
        Why:  Halves memory footprint with negligible accuracy loss because
              bfloat16 keeps the same 8-bit exponent range as fp32 (avoids
              overflow), sacrificing only mantissa precision.
        Note: On CPU, bfloat16 arithmetic may not be faster than fp32 because
              most x86 CPUs lack native bfloat16 SIMD instructions.  The
              speedup is most pronounced on GPU (A100+) or Apple Silicon (MPS).

    "int8"
        Dynamic post-training quantization via torch.ao.quantization.
        Use:  model = torch.ao.quantization.quantize_dynamic(
                          model, {torch.nn.Linear}, dtype=torch.qint8)
        Why:  Converts nn.Linear weight matrices to int8 at model-load time.
              At inference, each layer dequantises weights back to fp32 on the
              fly, performs the matmul, then discards the fp32 copy.  Net
              effect: ~4× smaller model on disk/RAM, 2–3× faster matmuls on
              CPU due to int8 SIMD throughput.
        IMPORTANT: quantize_dynamic() returns a *new* model object.
                   You must reassign: model = torch.ao.quantization...()

    Args:
        model : Loaded AutoModelForSeq2SeqLM (fp32 weights, eval mode).
        mode  : One of "fp32", "bfloat16", or "int8".

    Returns:
        The (possibly quantized) model.
    """
    if mode == "fp32":
        print("  [fp32] Baseline — no quantization applied.")

    elif mode == "bfloat16":
        # ------------------------------------------------------------------
        # YOUR CODE
        # Cast the model to bfloat16.
        # Hint: model = model.to(torch.bfloat16)
        # Then print a confirmation message.
        # ------------------------------------------------------------------
        pass

    elif mode == "int8":
        # ------------------------------------------------------------------
        # YOUR CODE
        # Apply dynamic quantization to all nn.Linear layers.
        # Hint:
        #   model = torch.ao.quantization.quantize_dynamic(
        #               model, {torch.nn.Linear}, dtype=torch.qint8)
        # Then print a confirmation message.
        # ------------------------------------------------------------------
        pass

    else:
        print(f"  [?] Unknown mode '{mode}' — returning model unchanged.")

    return model


# ===========================================================================
# EXERCISE 2 of 3
# ===========================================================================
def generate_summary(model, tokenizer, text: str) -> str:
    """Run one forward pass to produce a text summary.

    Step-by-step walkthrough
    ------------------------
    1. Tokenization
       tokenizer(text, return_tensors="pt", truncation=True,
                 padding="max_length", max_length=256)
       → returns a dict: {"input_ids": Tensor, "attention_mask": Tensor}
         input_ids      — integer IDs, one per token; shape [1, 256]
         attention_mask — 1 for real tokens, 0 for padding; same shape
         return_tensors="pt" means "give me PyTorch tensors, not lists"

    2. torch.no_grad()
       Wrapping inference in this context manager tells PyTorch *not* to
       record operations for gradient computation.  During training this
       recording is needed for backprop; during inference it is pure overhead.
       Disabling it saves memory and speeds up the forward pass.

    3. model.generate(**inputs)
       The ** unpacks the tokenizer dict as keyword arguments, so PyTorch
       receives input_ids= and attention_mask= separately.
       generate() runs beam search (by default: 4 beams) and returns a 2-D
       tensor of output token IDs, shape [batch_size, seq_len].

    4. tokenizer.decode(outputs[0], skip_special_tokens=True)
       outputs[0] is the first (and only) sequence in the batch.
       decode() converts token IDs → a human-readable string.
       skip_special_tokens=True removes <pad>, </s>, etc. from the output.

    Args:
        model     : Any loaded (or quantized) Seq2Seq model in eval mode.
        tokenizer : The matching tokenizer (must come from the same checkpoint).
        text      : Raw input string, already prefixed with "summarize: ".

    Returns:
        The decoded summary string.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    with torch.no_grad():
        # ------------------------------------------------------------------
        # YOUR CODE
        # Run model.generate() with the tokenized inputs unpacked as kwargs.
        # Hint: outputs = model.generate(**inputs)
        # ------------------------------------------------------------------
        outputs = None  # replace this line

    # ----------------------------------------------------------------------
    # YOUR CODE
    # Decode the first output sequence back to a string.
    # Hint: summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # ----------------------------------------------------------------------
    summary = None  # replace this line

    return summary


# ===========================================================================
# EXERCISE 3 of 3
# ===========================================================================
def compute_metrics(predictions: list, references: list) -> dict:
    """Compute BLEU and ROUGE-L for a batch of predictions.

    BLEU — Bilingual Evaluation Understudy
    ---------------------------------------
    Computes the geometric mean of 1-gram through 4-gram precision scores,
    with a brevity penalty for overly short predictions.
    Score range: 0.0 (no n-gram overlap) to 1.0 (perfect match).

    evaluate.load("bleu").compute() signature:
        predictions : list[str]  — one generated string per example
        references  : list[list[str]]  — one or more reference strings PER
                      example, so each element must itself be a list.
                      With one reference: [[ref0], [ref1], [ref2], ...]

    ROUGE — Recall-Oriented Understudy for Gisting Evaluation
    ----------------------------------------------------------
    Measures recall of shared content between prediction and reference.
    ROUGE-L uses the Longest Common Subsequence (LCS), which is more
    forgiving than exact n-gram matching and captures sentence-level order.
    Score range: 0.0 to 1.0.

    evaluate.load("rouge").compute() signature:
        predictions : list[str]
        references  : list[str]  — plain list (not list-of-lists, unlike BLEU)

    The result dict contains keys: "rouge1", "rouge2", "rougeL", "rougeLsum".
    We use "rougeL" (capital L) for sentence-level LCS.

    Args:
        predictions : Model-generated summaries, one per SAMPLE_DATA entry.
        references  : Human-written reference summaries, one per entry.

    Returns:
        dict with keys "bleu" (float) and "rouge_l" (float), both 0–1.
    """
    # ----------------------------------------------------------------------
    # YOUR CODE
    # Load the BLEU metric.
    # Hint: bleu_metric = evaluate.load("bleu")
    # ----------------------------------------------------------------------
    bleu_metric = None  # replace this line

    # ----------------------------------------------------------------------
    # YOUR CODE
    # Load the ROUGE metric.
    # Hint: rouge_metric = evaluate.load("rouge")
    # ----------------------------------------------------------------------
    rouge_metric = None  # replace this line

    # ----------------------------------------------------------------------
    # YOUR CODE
    # Compute BLEU.  Remember: references must be a list-of-lists.
    # Hint: bleu_result = bleu_metric.compute(
    #           predictions=predictions,
    #           references=[[r] for r in references])
    # ----------------------------------------------------------------------
    bleu_result = None  # replace this line

    # ----------------------------------------------------------------------
    # YOUR CODE
    # Compute ROUGE.  References is a plain list here.
    # Hint: rouge_result = rouge_metric.compute(
    #           predictions=predictions,
    #           references=references)
    # ----------------------------------------------------------------------
    rouge_result = None  # replace this line

    return {
        "bleu": round(bleu_result["bleu"], 4),
        "rouge_l": round(rouge_result["rougeL"], 4),
    }


# ===========================================================================
# SCAFFOLDING — already implemented; no changes needed below this line.
# ===========================================================================

def benchmark_variant(model, tokenizer, mode_label: str, num_runs: int = 5) -> dict:
    """Measure latency and accuracy for one quantization variant.

    Latency methodology
    -------------------
    We run the *same* dialogue `num_runs` times and average the wall-clock
    time.  A single warmup pass is run first so that any one-time JIT
    compilation or cache warming doesn't inflate the first measurement.

    Accuracy methodology
    --------------------
    We generate summaries for all five SAMPLE_DATA dialogues and compare
    them to the reference strings using BLEU and ROUGE-L.

    Args:
        model      : The (quantized) model to benchmark.
        tokenizer  : Matching tokenizer.
        mode_label : Human-readable label printed in the table.
        num_runs   : Number of latency repetitions to average.

    Returns:
        dict with keys: mode, avg_latency_s, bleu, rouge_l.
    """
    print(f"\n--- Benchmarking: {mode_label} ---")

    # Warmup: one pass so JIT and OS caches are warm before timing starts
    warmup_text = SAMPLE_DATA[0]["dialogue"]
    generate_summary(model, tokenizer, warmup_text)

    # Latency: average over num_runs passes on the first sample
    start = time.time()
    for _ in range(num_runs):
        generate_summary(model, tokenizer, warmup_text)
    avg_latency = (time.time() - start) / num_runs
    print(f"  Avg latency ({num_runs} runs): {avg_latency:.4f}s")

    # Accuracy: generate summaries for all five samples
    predictions = [
        generate_summary(model, tokenizer, d["dialogue"]) for d in SAMPLE_DATA
    ]
    references = [d["reference"] for d in SAMPLE_DATA]
    metrics = compute_metrics(predictions, references)
    print(f"  BLEU:    {metrics['bleu']:.4f}")
    print(f"  ROUGE-L: {metrics['rouge_l']:.4f}")

    return {
        "mode": mode_label,
        "avg_latency_s": round(avg_latency, 4),
        "bleu": metrics["bleu"],
        "rouge_l": metrics["rouge_l"],
    }


def main():
    model_name = "local_models/t5-small"
    print(f"Loading tokenizer from '{model_name}' …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    results = []

    # Run each quantization mode in sequence.
    # The model is reloaded fresh each iteration so that bfloat16 or int8
    # state from one run doesn't carry over into the next.
    for mode in ("fp32", "bfloat16", "int8"):
        print(f"\nLoading base model for mode: {mode}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
        # eval() disables dropout and other training-only behaviour
        model.eval()
        model = quantize_model(model, mode)
        result = benchmark_variant(model, tokenizer, mode_label=mode)
        results.append(result)

    # Print a side-by-side comparison table
    print("\n" + "=" * 60)
    print(f"{'Mode':<12} {'Latency (s)':<14} {'BLEU':<10} {'ROUGE-L'}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['mode']:<12} {r['avg_latency_s']:<14} "
            f"{r['bleu']:<10} {r['rouge_l']}"
        )
    print("=" * 60)
    print("\nDone! Compare the rows to see the latency/accuracy trade-off.")
    print("Expected pattern: int8 fastest; bfloat16 ~= fp32 on CPU; "
          "accuracy drops slightly with more aggressive quantization.")


if __name__ == "__main__":
    main()
