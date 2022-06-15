"""
usage: python fetch_pdf.py --url_file [filename] --out_dir [papers_raw/mathy_pos]
assign ids
"""
import urllib.request
import os
import argparse
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
# opener.addheaders = [('User-agent', 'Mozilla/5.0'), ('Content-Length', max_size)]
urllib.request.install_opener(opener)


def load_data(url_file, out_dir, concept):
    filename = url_file
    invalid_links_filename = os.path.join("urls", concept+"_invalid_links.txt")
    invalid_links_file = open(invalid_links_filename, "w+")
    f = open(filename, encoding="utf8")
    links = f.readlines()
    f.close()
    # os.mkdir(out_dir)
    for i, link in enumerate(links):
        print(link.strip())
        path = os.path.join(out_dir, f"{i}.pdf")
        try:
            urllib.request.urlretrieve(link.strip(), path)
            # urlrequest, browser(done), this is part of urlib.request
            # content_length - size of pdf
        except:
            invalid_links_file.write(link.strip())
            print(f"Invalid link: {link.strip()}")
    invalid_links_file.close()

# print(sys.argv)
# os.mkdir(".\\papers_raw")

parser = argparse.ArgumentParser()
parser.add_argument("--url_file")
parser.add_argument("--out_dir")
parser.add_argument("--concept")
args = parser.parse_args()
os.makedirs(args.out_dir)
# print(args.url_file)
load_data(args.url_file, args.out_dir, args.concept)

# error handling pending
# why are explanations important
# add xinsu0918@gmail.com to meetings

# conda env create -f environment.yml
# conda create -n control
# conda activate control
# conda env update --prefix ./env --file environment.yml  --prune
