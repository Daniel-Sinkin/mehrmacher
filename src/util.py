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


NAME_FILE_MD5_HASH = "401b3c4cac9e28a586ed15e81256680d"


def get_names_list(do_print=False) -> list[str]:
    """
    Returns a list of the names from the repo.

    If there is no local copy the names.txt gets downloaded and save locally,
    otherwise we load the names from the local file.
    """
    folder_path = os.path.join("data")
    file_name = "names.txt"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        if do_print:
            print("names.txt exists, checking MD5 Hash...")
        if compute_md5(file_path) == NAME_FILE_MD5_HASH:
            if do_print:
                print("...the MD5 Hash matches, reading from disc.")
            with open(file_path, "r", encoding="utf-8") as file:
                names_list: list[str] = [
                    name.strip() for name in file.readlines() if name.strip() != ""
                ]
            return names_list
        else:
            if do_print:
                print("...the MD5 Hash does not match, downloading a new file.")

    url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise RuntimeError("Failed to fetch names list")
    if do_print:
        print("Managed to download the file...")

    names_list = [name for name in response.text.splitlines()]

    if not os.path.exists(folder_path):
        if do_print:
            print("...{folder_path} does not exist, creating it...")
        os.makedirs(folder_path)
    with open(file_path, "w", encoding="utf-8") as file:
        if do_print:
            print("...saving names.txt locally...")
        file.write(response.text)

    if do_print:
        print("...get_names_list() finished.")

    return names_list


def get_chars() -> str:
    """Returns 'abcdefghijklmnopqrstuvwxyz'."""
    return "".join([chr(i) for i in range(ord("a"), ord("z") + 1)])


def get_itos_stoi() -> tuple[dict[int, str], dict[str, int]]:
    """Returns integer to character and character to integer mappings."""
    chars = get_chars()
    itos: dict[int, str] = {0: "."}
    itos.update({i + 1: s for i, s in enumerate(chars)})
    stoi: dict[str, int] = {s: i for i, s in itos.items()}
    return itos, stoi


if __name__ == "__main__":
    get_names_list()
