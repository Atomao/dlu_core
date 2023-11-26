import re
from pathlib import Path


def handle_source(self, sources):
    if isinstance(sources, str):
        sources = [sources]
    elif isinstance(sources, list):
        if all(isinstance(item, str) for item in sources):
            sources = sources
        else:
            raise ValueError("All items in the list must be strings")
    else:
        raise TypeError("sources must be a string or a list of strings")
    return [Path(source) for source in sources]


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
