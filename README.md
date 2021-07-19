# Mathy
**Installation**

**Learning a concept**
1) Add URLS of positive and negative examples of the concept in two separate files
    - make sure both files have the name of the concept in them
2) Fetch the PDFs from the web:

    a) run: ```python fetch_pdf.py --url_file [filename] --out_dir [papers_raw/concept_name]```
    
    b) here you will have to run this twice: once for the positive examples (e.g., mathy)
       and the second time for the negative examples (e.g., non_mathy)
3) Process the PDFs (convert them into .txts)

    a) run: ```python process_pdf.py --input_dir [papers_raw/concept_name] --output_dir [papers_processed/concept_name]```
    
    b) here you will have to run this twice: once for the positive examples (e.g., mathy)
       and the second time for the negative examples (e.g., non_mathy)
4) Make embeddings for the PDFs (now txt)

    a) run: ```python classify.py --data-dir [papers_processed] --concept concept_name --process_data```
    
    b) you don't have to do this twice; the code will take care of that
5) Plot the learning curve for the concept

    a) run: ```python classify.py --data-dir [papers_processed] --concept concept_name --learning_curve --learning_curve_output [output_file.png] --model [lr/svm/perceptron]```
    
    b) here lr = Logistic Regression, svm = SVM, perceptron = Multi-layer Perceptron

**Frequent errors**
