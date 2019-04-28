#-- coding: utf-8 -*-

import argparse
import os
from os import path
import re
import requests
from tqdm import tqdm


paper_url = "https://raw.githubusercontent.com/DongjunLee/notes/master/papers.md"

sync_path = "sync_path"
base_dname = "hb-research-notes"
base_path = ""

category = "Etc"
title = "paper_title"

arXiv_pattern = "\[arXiv.*\]\(.+\)"
date_pattern = "\(\d\d\d\d"


def sync(s_path):
    global sync_path
    sync_path = s_path

    global base_path
    base_path = path.join(sync_path, base_dname)

    notes = read_notes()

    r = requests.get(paper_url)
    parsed_data = parse(r.text)

    for category, items in parsed_data.items():

        for item in items:
            if item["title"] in notes:
                pass
            else:
                download(category, item["title"], item["pdf"])


def read_notes():
    notes = []

    for _, _, files in os.walk(base_path):
        for f in files:

            if f.endswith(".pdf"):
                notes.append(f)
    return notes


def parse(raw_text):

    parsed_data = {}

    for line in raw_text.splitlines():

        # parse category
        if line.startswith("###"):
            category = line[4:].strip()
            category = category.replace(" ", "_")

        # parse arXiv title
        if line.startswith("-") and line.endswith(")"):
            date = re.search("\(\d\d\d\d", line)
            if date is None:
                date_index = line.index("(")
            else:
                date_index = line.index(date.group())

            title = line[2:date_index]
            title = title.replace("**", "")
            title = title.strip()

        # parse arXiv url
        if "arXiv" in line:
            for arXiv in re.findall(arXiv_pattern, line):
                url = arXiv[arXiv.index("http"):-1]
                if "," in url:
                    url = url[:url.index(",")-1]

                pdf = url.replace("abs", "pdf") + ".pdf"
                published_date = url[url.index("abs") + 4:]

                if category not in parsed_data:
                    parsed_data[category] = []

                parsed_data[category].append({
                    "title": f"({published_date}) {title}.pdf",
                    "pdf": pdf,
                    "published_date": published_date
                })

    return parsed_data


def download(d_name, f_name, url):
    d_path = os.path.join(base_path, d_name)
    if not os.path.exists(d_path):
        os.makedirs(d_path)

    r = requests.get(url, stream=True)
    if r.status_code == 200:
        total_length = r.headers.get('content-length')
        f_path = os.path.join(d_path, f_name)

        with open(f_path, 'wb') as f:
            r.raw.decode_content = True
            copyfileobj(r.raw, f, int(total_length))


def copyfileobj(fsrc, fdst, total_length, length=16*1024):
    print(f"start download arXiv ... {fdst.name}")
    pbar = tqdm(total=total_length)
    while True:
        buf = fsrc.read(length)
        pbar.update(length)
        if not buf:
            pbar.close()
            break
        fdst.write(buf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sync_path', type=str, default='sync_path',
                        help='enter your sync_path')
    args = parser.parse_args()

    sync(args.sync_path)
