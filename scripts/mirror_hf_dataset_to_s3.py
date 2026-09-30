"""Mirror HF Hub dataset sources to S3 for `load_dataset_mix` via `data.local_data_dir`.

Row budgets use parquet footer metadata (HTTP range reads). Non-parquet sources are
streamed and written as local parquet. Downloads are resumable; S3 upload uses `aws s3 sync`.
"""

import math
import tempfile
from pathlib import Path
from typing import List, Optional

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
from omegaconf import OmegaConf

from autoindexer.utils.aws import aws_s3_sync

_ROWS_PER_PARQUET_SHARD = 50_000


def _select_shards_for_row_budget(
    dataset_name: str, data_dir: Optional[str], min_rows: int, revision: Optional[str] = None
) -> Optional[List[str]]:
    """Parquet shard paths covering at least `min_rows`, or None if the source has no parquet."""
    fs = HfFileSystem()
    prefix = f"datasets/{dataset_name}" + (f"@{revision}" if revision else "")
    if data_dir:
        prefix = f"{prefix}/{data_dir}"
    shard_paths = sorted(fs.glob(f"{prefix}/**/*.parquet"))
    if not shard_paths:
        return None

    repo_prefix = f"datasets/{dataset_name}" + (f"@{revision}" if revision else "") + "/"
    selected = []
    cumulative_rows = 0
    for path in shard_paths:
        if cumulative_rows >= min_rows:
            break
        cumulative_rows += pq.ParquetFile(path, filesystem=fs).metadata.num_rows
        selected.append(path.removeprefix(repo_prefix))

    label = f"{dataset_name}/{data_dir}" if data_dir else dataset_name
    print(f"{label}: {len(selected)}/{len(shard_paths)} shard(s) cover {cumulative_rows:,} rows (budget: {min_rows:,})")
    return selected


def _stream_rows_to_parquet(
    dataset_name: str,
    data_dir: Optional[str],
    text_column: str,
    min_rows: int,
    revision: Optional[str],
    destination: Path,
) -> int:
    """Stream up to `min_rows` into parquet under `destination` (for sources without parquet footers)."""
    stream = datasets.load_dataset(
        dataset_name, data_dir=data_dir, split="train", streaming=True, revision=revision
    ).select_columns([text_column])
    schema = pa.schema([(text_column, pa.string())])
    destination.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    shard_index = 0
    batch: List[str] = []

    def flush() -> None:
        nonlocal shard_index, batch
        if not batch:
            return
        table = pa.table({text_column: batch}, schema=schema)
        pq.write_table(table, destination / f"shard_{shard_index:05d}.parquet", compression="zstd")
        shard_index += 1
        batch = []

    for row in stream:
        batch.append(row[text_column])
        rows_written += 1
        if len(batch) >= _ROWS_PER_PARQUET_SHARD:
            flush()
            print(f"  {dataset_name}: {rows_written:,}/{min_rows:,} rows -> {shard_index} shard(s)")
        if rows_written >= min_rows:
            break
    flush()
    return rows_written


def mirror_source(
    dataset_name: str,
    local_dir: str,
    s3_uri: str,
    data_dir: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    min_rows: Optional[int] = None,
    text_column: str = "text",
    mirror_full: bool = False,
) -> None:
    """Download one source (optionally row-capped) and sync to ``<s3_uri>/<local_dir>/``."""
    with tempfile.TemporaryDirectory(dir=cache_dir) as snapshot_dir:
        if min_rows is not None:
            shard_filenames = _select_shards_for_row_budget(dataset_name, data_dir, min_rows, revision)
            if shard_filenames is not None:
                print(f"Downloading {len(shard_filenames)} shard(s) of {dataset_name}...")
                for filename in shard_filenames:
                    hf_hub_download(repo_id=dataset_name, repo_type="dataset", revision=revision, filename=filename, local_dir=snapshot_dir)
                source_dir = Path(snapshot_dir) / data_dir if data_dir else Path(snapshot_dir)
            else:
                print(f"{dataset_name} ships no parquet shards -- streaming {min_rows:,} row(s) and converting to parquet...")
                rows_written = _stream_rows_to_parquet(
                    dataset_name=dataset_name,
                    data_dir=data_dir,
                    text_column=text_column,
                    min_rows=min_rows,
                    revision=revision,
                    destination=Path(snapshot_dir),
                )
                if rows_written < min_rows:
                    print(f"{dataset_name} exhausted after {rows_written:,} row(s), short of the {min_rows:,} budgeted.")
                else:
                    print(f"{dataset_name}: wrote {rows_written:,} row(s) as parquet.")
                source_dir = Path(snapshot_dir)
        else:
            if not mirror_full:
                raise SystemExit(
                    f"Refusing to mirror {dataset_name} in full: these sources run to TBs and a full mirror is "
                    "almost never what a training run needs. Pass --total-samples/--min-rows to size the mirror "
                    "to a run's budget, or --mirror-full to say you really do want the whole thing."
                )
            allow_patterns = [f"{data_dir}/**"] if data_dir else None
            print(f"Downloading {dataset_name}" + (f" (data_dir={data_dir})" if data_dir else "") + " in full...")
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                revision=revision,
                allow_patterns=allow_patterns,
                local_dir=snapshot_dir,
            )
            source_dir = Path(snapshot_dir) / data_dir if data_dir else Path(snapshot_dir)

        destination = f"{s3_uri.rstrip('/')}/{local_dir}"
        aws_s3_sync(str(source_dir), destination)


def mirror_all(
    datamix_config: str,
    s3_uri: str,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    total_samples: Optional[int] = None,
    margin: float = 1.2,
    mirror_full: bool = False,
) -> None:
    """Mirror every ``local_dir`` source in a ``_datamix.yaml`` config, with optional weighted row caps."""
    config = OmegaConf.load(datamix_config)
    sources = OmegaConf.to_container(config.data.sources, resolve=True)
    num_val_samples_per_source = config.data.get("num_val_samples_per_source", 200)
    skipped = [source["name"] for source in sources if not source.get("local_dir")]
    if skipped:
        print(f"Skipping sources with no local_dir (still Hub-streamed): {skipped}")

    total_weight = sum(float(source.get("weight", 1.0)) for source in sources)

    for source in sources:
        if not source.get("local_dir"):
            continue
        min_rows = None
        if total_samples:
            share = float(source.get("weight", 1.0)) / total_weight
            min_rows = math.ceil(share * total_samples * margin) + num_val_samples_per_source
        mirror_source(
            dataset_name=source["dataset_name"],
            local_dir=source["local_dir"],
            s3_uri=s3_uri,
            data_dir=source.get("data_dir"),
            revision=revision,
            cache_dir=cache_dir,
            min_rows=min_rows,
            text_column=source.get("text_column", "text"),
            mirror_full=mirror_full,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mirror HF Hub dataset source(s) to S3 for local (non-streaming) loading")
    parser.add_argument("--all", action="store_true", help="Mirror every local_dir-tagged source in --datamix-config")
    parser.add_argument("--datamix-config", help="Path to a _datamix.yaml-shaped Hydra config, required with --all")
    parser.add_argument("--total-samples", type=int, help="With --all: cap each source to its weighted share of this many training samples")
    parser.add_argument("--margin", type=float, default=1.2, help="With --total-samples: safety multiplier on each source's row budget (default: 1.2)")
    parser.add_argument("--dataset-name", help="HF Hub dataset id, required without --all")
    parser.add_argument("--data-dir", help="Subdirectory to mirror, e.g. a StarCoderData language")
    parser.add_argument("--local-dir", help="Destination path under --s3-uri, required without --all")
    parser.add_argument("--min-rows", type=int, help="Without --all: cap this source to just enough shards to cover this many rows")
    parser.add_argument("--text-column", default="text", help="Without --all: source's text column, kept when converting a raw-text source to parquet (default: text)")
    parser.add_argument("--mirror-full", action="store_true", help="Allow mirroring a source in full (TB-scale) when no row budget is given")
    parser.add_argument("--s3-uri", required=True, help="S3 prefix to mirror into, e.g. s3://<bucket>/cpt_datamix")
    parser.add_argument("--revision", help="HF Hub dataset revision/commit to pin to")
    parser.add_argument("--cache-dir", help="Local directory for the temporary download (default: system tmp)")

    args = parser.parse_args()

    if args.all:
        if not args.datamix_config:
            parser.error("--all requires --datamix-config")
        mirror_all(
            datamix_config=args.datamix_config,
            s3_uri=args.s3_uri,
            revision=args.revision,
            cache_dir=args.cache_dir,
            total_samples=args.total_samples,
            margin=args.margin,
            mirror_full=args.mirror_full,
        )
    else:
        if not args.dataset_name or not args.local_dir:
            parser.error("--dataset-name and --local-dir are required without --all")
        mirror_source(
            dataset_name=args.dataset_name,
            local_dir=args.local_dir,
            s3_uri=args.s3_uri,
            data_dir=args.data_dir,
            revision=args.revision,
            cache_dir=args.cache_dir,
            min_rows=args.min_rows,
            text_column=args.text_column,
            mirror_full=args.mirror_full,
        )
