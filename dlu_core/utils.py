import re
from pathlib import Path


def find_files_recursive(directories, pattern):
    regex = re.compile(pattern)
    matching_files = []

    for directory in directories:
        base_path = Path(directory)
        all_files = base_path.rglob("*")
        matching_files.extend(
            [
                Path(str(file))
                for file in all_files
                if file.is_file() and regex.search(str(file.name))
            ]
        )

    if len(matching_files) > 1:
        logging.warning(
            f"find few matching files for {annotation_path} with same annotation file name, choose first"
        )
        raise Exception
    elif len(matching_files) == 0:
        logging.error(f"Don't find any images for {annotation_path}")
        raise Exception()
    return matching_files
