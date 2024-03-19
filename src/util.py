"""Utility functions for the project."""

import hashlib
import os

import requests


def compute_md5(file_path):
    """Computes the md5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


FILE_MD5_HASH = "401b3c4cac9e28a586ed15e81256680d"


def get_names_list() -> list[str]:
    """
    Returns a list of the names from the repo.

    If there is no local copy the names.txt gets downloaded and save locally,
    otherwise we load the names from the local file.
    """
    folder_path = os.path.join("data")
    file_name = "names.txt"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path) and compute_md5(file_path) == FILE_MD5_HASH:
        with open(file_path, "r", encoding="utf-8") as file:
            names_list: list[str] = [
                name.strip() for name in file.readlines() if name.strip() != ""
            ]
        return names_list

    url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise RuntimeError("Failed to fetch names list")

    names_list = [name for name in response.text.split("\n") if name != ""]

    if not os.path.exists(folder_path):
        print("{folder_path} does not exist, creating it...")
        os.makedirs(folder_path)
    with open(file_path, "w", encoding="utf-8") as file:
        print("Saving names.txt locally...")
        file.write(response.text)

    return names_list
