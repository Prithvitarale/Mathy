"""
usage: python fetch_pdf.py --url_file [filename] --out_dir [papers_raw/mathy]
assign ids
"""
from urllib.request import urlretrieve
import os
import argparse


def load_data(url_file, out_dir):
    filename = url_file
    f = open(filename, encoding="utf8")
    links = f.readlines()
    # os.mkdir(out_dir)
    for i, link in enumerate(links):
        print(link.strip())
        path = os.path.join(out_dir, f"{i}.pdf")
        urlretrieve(link.strip(), path)

# print(sys.argv)
# os.mkdir(".\\papers_raw")

parser = argparse.ArgumentParser()
parser.add_argument("--url_file")
parser.add_argument("--out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir)
# print(args.url_file)
load_data(args.url_file, args.out_dir)

# error handling pending
# why are explanations important
# add xinsu0918@gmail.com to meetings

# conda env create -f environment.yml
# conda create -n control
# conda activate control
# conda env update --prefix ./env --file environment.yml  --prune
