"""
usage: python fetch_pdf.py --url_file [filename] --out_dir [papers_raw/mathy]
assign ids
"""
from urllib.request import urlretrieve
import sys
import os
def load_data(url_file, out_dir):
    filename = url_file
    f = open(filename, encoding="utf8")
    links = f.readlines()
    os.mkdir(out_dir)
    for i, link in enumerate(links):
        print(i)
        path = os.path.join(out_dir, f"{i}.pdf")
        urlretrieve(link.strip(), path)

print(sys.argv)
os.mkdir(".\\papers_raw")
load_data(sys.argv[2], sys.argv[4])
# conda env create -f environment.yml
# conda create -n control
# conda activate control