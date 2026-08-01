"""Download OpenMHC LSM-2 and extract a few daily representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/embeddings/smoke"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints/lsm2-daily"))
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--repo-id", default="MyHeartCounts/openmhc-lsm2-daily")
    parser.add_argument("--checkpoint-name", default="loss=0.2706.ckpt")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-days", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    import openmhc
    from data.normalization import load_global_normalization_stats
    from openmhc.models.lsm2.modules import LSM2Module
    from openmhc.models.lsm2.utils import create_inherited_mask

    if args.checkpoint_path is not None:
        checkpoint_path = str(args.checkpoint_path.resolve())
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(checkpoint_path)
    else:
        checkpoint_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.checkpoint_name,
            local_dir=args.checkpoint_dir,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSM2Module.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    ).model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    stats_path = args.data_dir / "processed" / "normalization_stats.json"
    stats = load_global_normalization_stats(stats_path)
    data, _ = next(
        iter(
            openmhc.iter_split_data(
                args.split,
                version="xs",
                data_dir=args.data_dir,
                batch_size=args.max_days,
            )
        )
    )
    metadata = openmhc.load_sample_metadata(
        args.split,
        version="xs",
        data_dir=args.data_dir,
    )[: len(data)]

    normalized = np.stack([stats.normalize_numpy(day) for day in data])
    tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    inherited_mask = create_inherited_mask(tensor, patch_size=model.patch_size)
    with torch.inference_mode():
        latent, applied_mask = model.forward_encoder_dense(tensor, inherited_mask)
        rows = []
        for index in range(latent.shape[0]):
            observed = applied_mask[index] == 0
            if not bool(observed.any()):
                raise RuntimeError(f"sample {index} has no observed patch")
            rows.append(latent[index, observed].mean(dim=0).cpu().numpy())
    embeddings = np.stack(rows).astype(np.float32)

    output_path = args.output_dir / "lsm2_daily_embeddings.npz"
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        user_id=np.asarray([row["user_id"] for row in metadata]),
        date=np.asarray([row["date"] for row in metadata]),
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "input_shape": list(data.shape),
        "embedding_shape": list(embeddings.shape),
        "finite": bool(np.isfinite(embeddings).all()),
        "output": str(output_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
