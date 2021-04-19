"""
usage: python process_pdf.py --input_dir [papers_raw/mathy] --output_dir [papers_processed/mathy]

"""
import pdftotext
import glob
import os
import sys
import argparse
def process_pdf(input_dir, output_dir):
    print(glob.glob(input_dir))
    for i, pdf_file in enumerate(glob.glob(input_dir)):
        pdf = open(pdf_file, "rb")
        # print(i)
        # print(pdf)
        # print(pdf)
        text = pdftotext.PDF(pdf)
        pdf.close()
        pdfintext = ""
        for page in text:
            pdfintext += page + "\n"
        output_filename = os.path.join(output_dir, f"{i}.txt")
        f = open(output_filename, "w+", encoding="utf8")
        f.write(pdfintext)
        f.close()
os.mkdir(".\\papers_processed")
os.mkdir(".\\papers_processed\\mathy")

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir")
parser.add_argument("--output_dir")
args = parser.parse_args()
# print(args.url_file)
process_pdf(args.input_dir, args.output_dir)
