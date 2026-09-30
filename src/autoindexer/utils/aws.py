import os
import subprocess

import fsspec


def aws_s3_sync(source: str, destination: str) -> None:
    """Incremental sync between local paths and ``s3://`` URIs via ``aws s3 sync``."""
    result = subprocess.run(
        ["aws", "s3", "sync", source, destination],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"`aws s3 sync` failed (exit {result.returncode}): {detail}")


def verify_s3_upload(local_dir: str, s3_uri: str) -> None:
    """Raise if ``s3_uri`` does not contain every file under ``local_dir``."""
    local_files = {
        os.path.relpath(os.path.join(root, f), local_dir) for root, _, files in os.walk(local_dir) for f in files
    }
    fs, root = fsspec.core.url_to_fs(s3_uri)
    remote_files = {os.path.relpath(path, root) for path in fs.find(s3_uri)}
    missing = local_files - remote_files
    if missing:
        raise RuntimeError(
            f"{len(missing)} file(s) missing from {s3_uri} after `aws s3 sync` (e.g. {next(iter(missing))!r})"
        )
