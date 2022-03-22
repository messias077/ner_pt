# ner_pt
Repository of paper "Combining Word Embeddings for Portuguese Named Entity Recognition"

### Example Usage

To run the scripts for this work, follow the steps below. All datasets will be available after cloning the project (git clone https://github.com/messias077/ner_en.git). Don't forget to create a python virtual environment and install the dependencies from the requirements.txt file!

```python
# Preparing the corpus
python le_ner_corpus_clean.py
python split_clean_paramopama.py
```
```
# Choosing a corpus name
At the beginning of the script (python run_crf_experiments.py and/or python run_flair_experiments.py), uncomment the name of the corpus you want to use
```

```python
# Running the experiments
python run_crf_experiments.py
python run_flair_experiments.py

```

### Citing our work

Please cite [the following paper](https://link.springer.com/chapter/10.1007/978-3-030-98305-5_19) when using this work:

```
@inproceedings{silva2022propor,
  title={Combining Word Embeddings for Portuguese Named Entity Recognition},
  author={Silva, M.G. and Oliveira, H.T.A.},
  booktitle = {{PROPOR} 2022, 14th International Conference on the Computational Processing of Portuguese},
  volume    = {13208},
  pages     = {198--208},
  year      = {2022},
  publisher = {Springer}
}
```
