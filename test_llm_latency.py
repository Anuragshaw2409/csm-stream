"""Measure streaming latency of the Gemma model this project talks to.

Uses the same LLMInterface / OpenRouter path as main.py, so the numbers
reflect what speak_streaming() actually waits on.

Usage:
    python test_llm_latency.py --openrouter-api-key sk-or-v1-...
    python test_llm_latency.py --openrouter-api-key sk-or-v1-... --prompt "..." --runs 5
"""
import argparse
import statistics
import time

from llm_interface import LLMInterface

SYSTEM_PROMPT = "You are a warm, friendly voice companion having a spoken conversation. Keep replies conversational and concise."


def run_once(llm, system_prompt, prompt):
    """Stream one response, returning (ttft_ms, gaps_ms, total_ms, text)."""
    chunk_times = []
    text = ""

    start = time.perf_counter()
    for delta in llm.generate_response_stream(system_prompt, prompt, ""):
        chunk_times.append(time.perf_counter() - start)
        text += delta
    total = time.perf_counter() - start

    if not chunk_times:
        return None

    ttft = chunk_times[0] * 1000
    gaps = [(chunk_times[i] - chunk_times[i - 1]) * 1000 for i in range(1, len(chunk_times))]
    return ttft, gaps, total * 1000, text


def main():
    parser = argparse.ArgumentParser(description="Test Gemma/OpenRouter streaming latency")
    parser.add_argument("--openrouter-api-key", required=True)
    parser.add_argument("--openrouter-model", default="google/gemma-4-31b-it:exacto")
    parser.add_argument("--prompt", default="What should I make for breakfast? Give me a few ideas.")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    llm = LLMInterface(api_key=args.openrouter_api_key, model=args.openrouter_model)

    print(f"model  : {args.openrouter_model}")
    print(f"prompt : {args.prompt!r}")

    ttfts, totals, all_gaps = [], [], []

    for i in range(1, args.runs + 1):
        result = run_once(llm, SYSTEM_PROMPT, args.prompt)
        if result is None:
            print(f"\nrun {i}: no response received")
            continue

        ttft, gaps, total, text = result
        ttfts.append(ttft)
        totals.append(total)
        all_gaps.extend(gaps)

        print(f"\n--- run {i} ---")
        print(f"  time to first chunk : {ttft:8.0f} ms")
        print(f"  total response time : {total:8.0f} ms")
        print(f"  chunks received     : {len(gaps) + 1}  ({len(text)} chars)")
        if gaps:
            print(f"  inter-chunk delay   : min {min(gaps):.0f} ms | "
                  f"median {statistics.median(gaps):.0f} ms | max {max(gaps):.0f} ms")
        print(f"  response: {text.strip()}")

    if not ttfts:
        return

    print(f"\n=== summary over {len(ttfts)} run(s) ===")
    print(f"  time to first chunk : min {min(ttfts):.0f} ms | "
          f"mean {statistics.mean(ttfts):.0f} ms | max {max(ttfts):.0f} ms")
    print(f"  total response time : min {min(totals):.0f} ms | "
          f"mean {statistics.mean(totals):.0f} ms | max {max(totals):.0f} ms")
    if all_gaps:
        print(f"  inter-chunk delay   : min {min(all_gaps):.0f} ms | "
              f"median {statistics.median(all_gaps):.0f} ms | max {max(all_gaps):.0f} ms")


if __name__ == "__main__":
    main()
