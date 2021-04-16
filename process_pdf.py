"""
usage: python process_pdf.py --input_dir [papers_raw/mathy] --output_dir [papers_processed/mathy]

"""
import pdftotext
import glob
import os
import sys
def process_pdf(input_dir, output_dir):
    for i, pdf_file in enumerate(glob.glob(input_dir)):
        pdf = open(pdf_file, "rb")
        # print(i)
        text = pdftotext.PDF(pdf)
        pdfintext = ""
        for page in text:
            pdfintext += page + "\n"
        output_filename = os.path.join(output_dir, f"{i}.txt")
        f = open(output_filename, "w+", encoding="utf8")
        f.write(pdfintext)
        f.close()
os.mkdir(".\\papers_processed")
os.mkdir(".\\papers_processed\\mathy")
process_pdf(sys.argv[2], sys.argv[4])
