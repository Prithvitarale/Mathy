"""
usage: python process_pdf.py --input_dir [papers_raw/mathy] --output_dir [papers_processed/mathy]

"""
import pdftotext
import glob
import os
import argparse
from sklearn.model_selection import train_test_split

def process_pdf(input_dir, output_dir):
    files = []
    for i, pdf_file in enumerate(glob.glob(input_dir)):
        try:
            print(str(pdf_file))
            pdf = open(pdf_file, "rb")
            text = pdftotext.PDF(pdf, raw=True)
            pdf.close()
            pdfintext = ""
            for page in text:
                pdfintext += page
            files.append(pdfintext)
        except:
            print(f"Invalid link: {pdf_file.strip()}")
    x_train, x_test = train_test_split(files, test_size=0.20)
    write_to_disk("train", output_dir, x_train)
    write_to_disk("test", output_dir, x_test)


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
args = parser.parse_args()
# print(args.url_file)
os.makedirs(args.output_dir)

process_pdf(args.input_dir, args.output_dir)
