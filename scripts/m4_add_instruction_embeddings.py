"""M4-A: precompute CLIP text embeddings for every episode and copy the dataset.

Inputs : an existing phase-aware dataset directory (created by M3.4A)
         expected layout:
           <in_dir>/episodes/ep_NNNNNN.npz   (obs, actions, ... — phase-aware NPZ format)
           <in_dir>/episodes.jsonl           (per-episode metadata incl. "instruction")
           <in_dir>/dataset_schema.json
           <in_dir>/splits.json
           <in_dir>/summary.json
           [<in_dir>/action_bounds.json]

Outputs: same files copied to <out_dir>, plus
           - per-episode NPZ gains a `lang_emb` array of shape (T, lang_dim)
             so it can be concatenated directly with `obs` row-by-row at train time
           - dataset_schema.json gains `lang_emb_dim` and `text_encoder` fields
           - summary.json gains `text_encoder` and embedding stats
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer


def encode_instructions(
    instructions: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name).to(device)
    model.eval()

    all_emb: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(instructions), batch_size):
            batch = instructions[start : start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = model(**tokens)
            emb = out.pooler_output.cpu().numpy().astype(np.float32)
            all_emb.append(emb)

    return np.concatenate(all_emb, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--text-encoder", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(exist_ok=True)

    episodes_jsonl_in = in_dir / "episodes.jsonl"
    with episodes_jsonl_in.open("r", encoding="utf-8") as f:
        episode_records = [json.loads(line) for line in f if line.strip()]

    instructions = [rec["instruction"] for rec in episode_records]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[m4-A] encoding {len(instructions)} instructions with {args.text_encoder}")
    embeddings = encode_instructions(
        instructions=instructions,
        model_name=args.text_encoder,
        device=device,
        batch_size=args.batch_size,
    )
    lang_dim = int(embeddings.shape[1])
    print(f"[m4-A] embeddings shape={embeddings.shape}, dtype={embeddings.dtype}")

    assert len(episode_records) == embeddings.shape[0]
    out_records: list[dict] = []
    for rec, emb in zip(episode_records, embeddings):
        ep_path_in = in_dir / "episodes" / Path(rec["npz_path"]).name
        ep_path_out = out_dir / "episodes" / ep_path_in.name

        with np.load(ep_path_in, allow_pickle=True) as data:
            arrays = {key: data[key] for key in data.files}

        T = int(arrays["obs"].shape[0])
        arrays["lang_emb"] = np.broadcast_to(emb[None, :], (T, lang_dim)).astype(np.float32).copy()

        np.savez_compressed(ep_path_out, **arrays)

        new_rec = dict(rec)
        new_rec["npz_path"] = str(ep_path_out)
        new_rec["lang_emb_dim"] = lang_dim
        out_records.append(new_rec)

    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    schema_path = in_dir / "dataset_schema.json"
    if schema_path.exists():
        schema = json.load(schema_path.open("r", encoding="utf-8"))
    else:
        schema = {}
    schema["lang_emb_dim"] = lang_dim
    schema["text_encoder"] = args.text_encoder
    schema["source_dataset_dir"] = str(in_dir)
    with (out_dir / "dataset_schema.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    for fname in ["splits.json", "action_bounds.json"]:
        src = in_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)

    summary_in_path = in_dir / "summary.json"
    summary = json.load(summary_in_path.open("r", encoding="utf-8")) if summary_in_path.exists() else {}
    summary["milestone"] = "M4A"
    summary["description"] = "Phase-aware dataset augmented with CLIP text embeddings of the instruction."
    summary["source_dataset_dir"] = str(in_dir)
    summary["text_encoder"] = args.text_encoder
    summary["lang_emb_dim"] = lang_dim
    summary["lang_emb_stats"] = {
        "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
        "min_norm": float(np.min(np.linalg.norm(embeddings, axis=1))),
        "max_norm": float(np.max(np.linalg.norm(embeddings, axis=1))),
        "num_unique_instructions": len(set(instructions)),
        "num_episodes": len(instructions),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[m4-A] done. out_dir={out_dir}")
    print(json.dumps(summary["lang_emb_stats"], indent=2))


if __name__ == "__main__":
    main()
