# ner_pt
Repository of paper "Combining Word Embeddings for Portuguese Named Entity Recognition"

### Example Usage

To run the scripts for this work, follow the steps below. All datasets will be available after cloning the project (git clone https://github.com/messias077/ner_en.git). Don't forget to create a python virtual environment and install the dependencies from the requirements.txt file!

```
# Preparing the corpus
python le_ner_corpus_clean.py
python split_clean_paramopama.py

# Choosing a corpus name
At the beginning of the script (run_crf_experiments.py and/or run_flair_experiments.py), 
uncomment the name of the corpus you want to use

# Running the experiments
python run_crf_experiments.py
python run_flair_experiments.py
```

### Citing our work

Please cite [the following paper](https://link.springer.com/chapter/10.1007/978-3-030-98305-5_19) when using this work:

```
@inproceedings{silva2022propor,
  author={da Silva, Messias Gomes and de Oliveira, Hil{\'a}rio Tomaz Alves},
  title={Combining Word Embeddings for Portuguese Named Entity Recognition},
  booktitle = {Computational Processing of the Portuguese Language},
  year      = {2022},
  volume    = {13208},
  pages     = {198--208},
  publisher = {Springer International Publishing},
  isbn      = {978-3-030-98305-5}
}
```
