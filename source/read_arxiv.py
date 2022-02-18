import jsonlines

arxiv_papers_file = open("../urls/arxiv_papers.txt", "w")

with jsonlines.open('../arxiv-metadata-oai-snapshot.json') as f:
    for line in f.iter():
        url = "https://arxiv.org/pdf/" + line['id'] + ".pdf"
        arxiv_papers_file.write(url)
        arxiv_papers_file.write("\n")

arxiv_papers_file.close()
