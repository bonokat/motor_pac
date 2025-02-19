import os

def check_paths(*paths: str) -> None:
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

