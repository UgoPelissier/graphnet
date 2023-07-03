import os
import logging
from lightning.fabric.utilities.cloud_io import get_filesystem


def get_next_version(logs: str) -> int:
    """Get the next version number for the logger."""
    log = logging.getLogger(__name__)
    fs = get_filesystem(logs)

    try:
        listdir_info = fs.listdir(logs)
    except OSError:
        log.warning("Missing logger folder: %s", logs)
        return 0

    existing_versions = []
    for listing in listdir_info:
        d = listing["name"]
        bn = os.path.basename(d)
        if fs.isdir(d) and bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace("/", "")
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1