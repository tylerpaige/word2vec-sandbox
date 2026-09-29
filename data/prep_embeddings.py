"""
One-time preprocessing: download the pretrained GloVe 6B / 100d word
vectors (via gensim-data's release asset) and emit a compact, int8
quantized subset for the browser word2vec-algebra web app.

Source: GloVe 6B (Pennington, Socher, Manning, EMNLP 2014), Public Domain
(PDDL). https://nlp.stanford.edu/projects/glove/

Output (all row-index-aligned, index 0 = most frequent word):
  words.txt         - newline-delimited vocabulary, top VOCAB_SIZE words
  vectors_int8.bin  - flat Int8Array, row-major, VOCAB_SIZE x DIMS
  scales_f32.bin    - flat Float32Array, one per-vector dequant scale

Dequantize: value = vectors_int8[i*DIMS + j] * scales[i] / 127
"""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

SOURCE_URL = (
    "https://github.com/piskvorky/gensim-data/releases/download/"
    "glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"
)
VOCAB_SIZE = 100_000
DIMS = 100

HERE = Path(__file__).parent
RAW_GZ = HERE / "glove-wiki-gigaword-100.gz"
OUT_WORDS = HERE / "words.txt"
OUT_VECTORS = HERE / "vectors_int8.bin"
OUT_SCALES = HERE / "scales_f32.bin"


def download():
    if RAW_GZ.exists():
        print(f"Already downloaded: {RAW_GZ} ({RAW_GZ.stat().st_size:,} bytes)")
        return
    print(f"Downloading {SOURCE_URL} ...")
    urllib.request.urlretrieve(SOURCE_URL, RAW_GZ)
    print(f"Downloaded {RAW_GZ.stat().st_size:,} bytes")


def parse_and_quantize():
    words = []
    vectors = np.empty((VOCAB_SIZE, DIMS), dtype=np.float32)

    with gzip.open(RAW_GZ, "rt", encoding="utf-8") as f:
        first_line = f.readline()
        # gensim KeyedVectors text format starts with a "<count> <dims>" header
        parts = first_line.split()
        if len(parts) != 2:
            # no header; treat first_line as data
            f.seek(0)

        for line in f:
            if len(words) >= VOCAB_SIZE:
                break
            parts = line.rstrip("\n").split(" ")
            word = parts[0]
            vals = parts[1:]
            if len(vals) != DIMS:
                continue
            words.append(word)
            vectors[len(words) - 1] = np.array(vals, dtype=np.float32)

    assert len(words) == VOCAB_SIZE, f"only got {len(words)} words"

    # per-vector int8 quantization
    scales = np.abs(vectors).max(axis=1) / 127.0
    scales[scales == 0] = 1.0  # guard against an all-zero row
    quantized = np.round(vectors / scales[:, None]).astype(np.int8)

    return words, quantized, scales.astype(np.float32)


def write_outputs(words, quantized, scales):
    OUT_WORDS.write_text("\n".join(words) + "\n", encoding="utf-8")
    quantized.tofile(OUT_VECTORS)
    scales.tofile(OUT_SCALES)
    print(f"{OUT_WORDS.name}: {OUT_WORDS.stat().st_size:,} bytes")
    print(f"{OUT_VECTORS.name}: {OUT_VECTORS.stat().st_size:,} bytes")
    print(f"{OUT_SCALES.name}: {OUT_SCALES.stat().st_size:,} bytes")


def sanity_check(words, quantized, scales):
    index = {w: i for i, w in enumerate(words)}

    def unit_vec(word):
        i = index[word]
        v = quantized[i].astype(np.float32) * scales[i]
        return v / np.linalg.norm(v)

    query = unit_vec("king") - unit_vec("man") + unit_vec("woman")
    query = query / np.linalg.norm(query)

    all_vecs = quantized.astype(np.float32) * scales[:, None]
    norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_all = all_vecs / norms

    scores = unit_all @ query
    exclude = {index["king"], index["man"], index["woman"]}
    ranked = np.argsort(-scores)
    top5 = [(words[i], float(scores[i])) for i in ranked if i not in exclude][:5]

    print("king - man + woman =>", top5)
    assert top5[0][0] == "queen", f"expected 'queen' on top, got {top5}"
    print("Sanity check passed.")


if __name__ == "__main__":
    download()
    words, quantized, scales = parse_and_quantize()
    write_outputs(words, quantized, scales)
    sanity_check(words, quantized, scales)
