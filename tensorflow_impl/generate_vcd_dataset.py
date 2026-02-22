#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import numpy as np


DEFAULT_GLIF_DATA_DIR = "/home/wanghz/EC_V1/data/GLIF_V1_network"
DEFAULT_LGN_TABLE = os.path.join(DEFAULT_GLIF_DATA_DIR, "lgn_full_col_cells_3.csv")


def str2bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def make_shared_sequences(
    *,
    batch_size: int,
    examples_in_epoch: int,
    n_images: int,
    p_reappear: float,
    seed: int,
):
    rd = np.random.RandomState(seed)
    n_pairs = int(2 * examples_in_epoch)
    seqs = np.zeros((n_pairs, batch_size), dtype=np.int32)
    changes = np.zeros((n_pairs, batch_size), dtype=np.int32)

    current = rd.randint(0, n_images, size=(batch_size,), dtype=np.int32)
    for i in range(n_pairs):
        change = rd.uniform(size=(batch_size,)) > p_reappear
        cand = rd.randint(0, max(n_images - 1, 1), size=(batch_size,), dtype=np.int32)
        cand = np.where(cand >= current, cand + 1, cand)
        cand = np.mod(cand, n_images)
        current = np.where(change, cand, current)

        seqs[i] = current
        changes[i] = change.astype(np.int32)

    return seqs, changes


def collect_dataset(ds, num_samples: int):
    xs, ys, ils, ws = [], [], [], []
    total = 0

    for batch in ds:
        x, y, image_label, w = [t.numpy() for t in batch]
        take = min(x.shape[0], num_samples - total)
        xs.append(x[:take])
        ys.append(y[:take])
        ils.append(image_label[:take])
        ws.append(w[:take])
        total += int(take)
        if total >= num_samples:
            break

    if total < num_samples:
        raise RuntimeError(f"Dataset ended early: requested {num_samples}, got {total}")

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    image_label = np.concatenate(ils, axis=0)
    w = np.concatenate(ws, axis=0)
    return x, y, image_label, w


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate VCD dataset via tensorflow_impl/stim_dataset.py and export NPZ"
    )
    parser.add_argument("--mode", type=str, default="vcd_ni", choices=["vcd_ni", "continuing"])
    parser.add_argument("--output_npz", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3000)

    parser.add_argument("--current_input", type=str2bool, default=True)
    parser.add_argument("--p_reappear", type=float, default=0.5)
    parser.add_argument("--im_slice", type=int, default=100)

    # vcd_ni mode
    parser.add_argument("--image_h5", type=str, default=None, help="H5 file containing key 'data'")
    parser.add_argument("--from_lgn", type=str2bool, default=True)
    parser.add_argument("--lgn_data_path", type=str, default=DEFAULT_LGN_TABLE)
    parser.add_argument("--intensity", type=float, default=2.0)
    parser.add_argument("--pre_delay", type=int, default=50)
    parser.add_argument("--post_delay", type=int, default=150)
    parser.add_argument("--pre_chunks", type=int, default=4)
    parser.add_argument("--resp_chunks", type=int, default=1)
    parser.add_argument("--post_chunks", type=int, default=1)
    parser.add_argument("--pairs_in_epoch", type=int, default=781)
    parser.add_argument("--vcd_batch_size", type=int, default=2)

    # continuing mode
    parser.add_argument(
        "--rates_path",
        type=str,
        default=os.path.join(DEFAULT_GLIF_DATA_DIR, "many_small_stimuli.pkl"),
    )
    parser.add_argument("--seq_len", type=int, default=600)
    parser.add_argument("--delay", type=int, default=200)
    parser.add_argument("--examples_in_epoch", type=int, default=50)
    parser.add_argument("--n_images", type=int, default=40)
    parser.add_argument("--cont_batch_size", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure local imports come from tensorflow_impl/
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    import tensorflow as tf
    import lgn_model
    import stim_dataset

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # For from_lgn path in stim_dataset, keep source file unchanged and inject path at runtime.
    if args.from_lgn:
        lgn_model.LGN = functools.partial(lgn_model.LGN, lgn_data_path=args.lgn_data_path)
        stim_dataset.lgn_model.LGN = lgn_model.LGN

    if args.mode == "vcd_ni":
        if args.image_h5 is None:
            raise ValueError("--image_h5 is required when --mode vcd_ni")

        ds = stim_dataset.generate_VCD_NI_from_path(
            path=args.image_h5,
            intensity=args.intensity,
            im_slice=args.im_slice,
            pre_delay=args.pre_delay,
            post_delay=args.post_delay,
            p_reappear=args.p_reappear,
            pre_chunks=args.pre_chunks,
            resp_chunks=args.resp_chunks,
            post_chunks=args.post_chunks,
            current_input=args.current_input,
            batch_size=args.vcd_batch_size,
            pairs_in_epoch=args.pairs_in_epoch,
            from_lgn=args.from_lgn,
        )
    else:
        shared_seqs, shared_changes = make_shared_sequences(
            batch_size=args.cont_batch_size,
            examples_in_epoch=args.examples_in_epoch,
            n_images=args.n_images,
            p_reappear=args.p_reappear,
            seed=args.seed,
        )

        ds = stim_dataset.generate_data_set_continuing(
            path=args.rates_path,
            batch_size=args.cont_batch_size,
            seq_len=args.seq_len,
            examples_in_epoch=args.examples_in_epoch,
            p_reappear=args.p_reappear,
            im_slice=args.im_slice,
            delay=args.delay,
            n_images=args.n_images,
            current_input=args.current_input,
            pre_chunks=args.pre_chunks,
            resp_chunks=args.resp_chunks,
            shared_seqs=shared_seqs,
            shared_changes=shared_changes,
        )

    ds = ds.batch(args.batch_size).prefetch(1)
    x, y, image_label, w = collect_dataset(ds, num_samples=args.num_samples)

    out = Path(args.output_npz).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "mode": args.mode,
        "num_samples": int(args.num_samples),
        "x_shape": tuple(int(v) for v in x.shape),
        "y_shape": tuple(int(v) for v in y.shape),
        "image_label_shape": tuple(int(v) for v in image_label.shape),
        "w_shape": tuple(int(v) for v in w.shape),
        "from_lgn": bool(args.from_lgn),
        "lgn_data_path": args.lgn_data_path,
    }

    np.savez_compressed(
        out,
        x=x.astype(np.float32, copy=False),
        y=y.astype(np.int32, copy=False),
        image_label=image_label.astype(np.int32, copy=False),
        w=w.astype(np.float32, copy=False),
        meta=json.dumps(meta, ensure_ascii=True),
    )

    print(f"Saved dataset to: {out}")
    print(json.dumps(meta, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
