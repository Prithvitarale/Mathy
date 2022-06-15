"""
usage: python process_pdf.py --input_dir [.\papers_raw\mathy\*] --output_dir [.\papers_processed\mathy]

"""
import pdftotext
import glob
import os
import argparse
from sklearn.model_selection import train_test_split
import requests

from bs4 import BeautifulSoup
import sys
# sys.path.append("C:\\Users\\prith\\miniconda3\\Lib\\site-packages\\pdfminer")
# from pdfminer.high_level import extract_pages
# from pdfminer.layout import LTTextContainer, LTChar

def process_pdf(input_dir, output_dir):
    files = []
    for i, file in enumerate(glob.glob(input_dir)):
        try:
            # if only_abstract_and_title:
            #     txt = get_abstract(file)
            #     # print(len(txt))
            #     files.append(txt)
            # else:
            with open(file, "rb") as pdf:
                text = pdftotext.PDF(pdf, raw=True)
                pdfintext = ""
                for page in text:
                    pdfintext += page
                files.append(pdfintext)
        except Exception as e:
            print(f"Invalid link: {file.strip()}")
            print(e)

    x_train, x_test = train_test_split(files, test_size=0.20)
    write_to_disk("train", output_dir, x_train)
    write_to_disk("test.txt", output_dir, x_test)

def process_links(input_dir, output_dir):
    files = []
    # try:
    f = open(input_dir, encoding="utf8")
    links = f.readlines()
    for i, link in enumerate(links):
        # print(link)
        try:
            files.append(get_abstract(link))
        except Exception as e:
            print(f"Invalid link: {link.strip()}")
            print(e)

    x_train, x_test = train_test_split(files, test_size=0.20)
    write_to_disk("train", output_dir, x_train)
    write_to_disk("test.txt", output_dir, x_test)

# def get_abstract(pdf_file):
#     for page_layout in extract_pages(pdf_file):
#         found_abstract = False
#         for element in page_layout:
#             if isinstance(element, LTTextContainer):
#                 txt = element.get_text()
#                 if txt.lower() == "abstract\n" or txt.lower() == "abstract":
#                     found_abstract = True
#                     continue
#                 elif "abstract\n".lower() in txt.lower():
#                     return txt
#                 if found_abstract:
#                     txt = element.get_text()
#                     # print(len(txt))
#                     return txt
#             else:
#                 continue

def get_abstract(link):
    page = requests.get(link.strip(), headers={'User-agent':'Mozilla/5.0'})
    print("got link")
    soup = BeautifulSoup(page.content, 'html.parser')
    print("got soup")
    abs = soup.find("meta", {"name": "description"})
    # print(abs["content"])
    return abs["content"]

def write_to_disk(dir_name, output_dir, data):
    output_filename = os.path.join(output_dir, dir_name)
    os.mkdir(output_filename)
    for i, x in enumerate(data):
        txt_file = os.path.join(output_filename, f"{i}.txt")
        f = open(txt_file, "w+", encoding="utf8")
        f.write(x)
        f.close()


# print(pdftotext.PDF.__dict__)
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir")
parser.add_argument("--output_dir")
parser.add_argument("--only_abstract_and_title", required=False, action="store_true")
args = parser.parse_args()
# print(args.url_file)
os.makedirs(args.output_dir) #TODO: uncomment later
if args.only_abstract_and_title:
    process_links(args.input_dir, args.output_dir)
else:
    process_pdf(args.input_dir, args.output_dir)

# may have to rewrite this without pdftotext
# library to divide pdf into parts
# look at that s2 abstract library