import os
from functools import reduce

header = """
# Table of contents

* [What is notes](README.md)

"""

def replace_summary():
    kb_contents = make_contents("./knowledge/")
    kb_texts = contents_to_text(kb_contents["knowledge"], texts=f"## Knowledge Base\n\n")

    code_contents = make_contents("./code/")
    code_texts = contents_to_text(code_contents, texts=f"## Code\n\n")

    note_contents = make_contents("./notes/")
    note_texts = contents_to_text(note_contents["notes"], texts=f"## Notes\n\n")

    summary_texts = ""
    summary_texts += header
    summary_texts += kb_texts
    summary_texts += code_texts
    summary_texts += note_texts

    summary_path = "./SUMMARY.md"
    with open(summary_path, "w") as f:
        f.write(summary_texts)

def make_contents(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        files = [f"{os.path.dirname(path)}/{f}" for f in files if f.endswith(".md")]
        if files:
            subdir = files
        else:
            subdir = {}
        parent = reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir

def contents_to_text(contents, texts="", indentation=1):
    tab_str = "    "
    for k, v in contents.items():
        padding = "".join([tab_str for _ in range(indentation - 1)])
        texts += padding + f"* [{capitalize_base_words(k)}]()\n"

        if type(v) == dict:
            texts += contents_to_text(v, texts="", indentation=indentation+1)
        elif type(v) == list:
            padding = "".join([tab_str for _ in range(indentation)])
            for item in v:
                texts += padding + f"* [{capitalize_base_words(item, remove_ext=True)}]({item})\n"
        else:
            raise ValueError("")
    return texts + "\n"

def capitalize_base_words(file_path, sep="_", remove_ext=False):
    basenamse = os.path.basename(file_path)
    words = basenamse.split(sep)
    c_words = []
    for word in words:
        word = word.capitalize()
        if remove_ext:
            word = word.split(".")[0]
        c_words.append(word)
    return " ".join(c_words)
