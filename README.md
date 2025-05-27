# Mutagenic: An Embedding-Based Approach to Protein Masking for Functional Redesign

![Mutagenic Poster](mutagenic_poster.png)

# Introduction

This repository contains the code associated with the research paper [Mutagenic: An Embedding-Based Approach to Protein Masking for Functional Redesign](https://openreview.net/pdf?id=0wy0QiPkKU). 

In MUTAGENIC, we investigate the following question: How can we efficiently identify residues to
edit in the engineering of proteins with specific target functions? More formally, let $F$ denote a set of possible protein functions. Given a protein sequence $s = s_1s_2\dots s_N$ composed of amino acids $s_i$, a function f $\in F$ and a target function f' $\in F$, our goal is to return a new protein sequence s' with functionality f'.



We propose to tackle this problem by leveraging the breakthrough capabilities of protein foundation models, specifically the recently released [ESM-3](https://github.com/evolutionaryscale/esm) model. Inspired by interpretability approaches in the natural language processing literature, we leverage the representational
learning capabilities of ESM-3 to create a novel pipeline that involves (1) masking the resides of the input protein and then (2) using ESM-3 to fill in these masks to achieve the target functionality. In particular, we propose to identify these masking sites via embedding-based similarity scores from the ESM-3 model. Please see our associated tiny paper for more details.  


# Setup & Installation

### Compute Requirements

We ran all of our scripts on a machine equipped with a single Nvidia A100 GPU. Due to the size of the ESM-3 foundation model, we note that our scripts may not run on machines with less GPU memory. 

### Getting Started

To reproduce our experimental results, please first clone this repository on your machine

```
git clone https://github.com/vihan-lakshman/mutagenic-experiments.git
```

Next, we __strongly__ recommend using a python virtual environment to manage all software dependencies. For example, you can create a virtual environment called `venv` as follows:

```
python3 -m venv venv
```

And then activate the environment with the command:

```
source venv/bin/activate
```

### Installing Dependencies

Once you have the virtualenv set up and `cd` into the repo, you can install all of the necessary software dependencies via the following command:

```
pip3 install -r requirements.txt
```

### Huggingface Hub Login

Finally, you will need to authenticate to Huggingface Hub to be able to use the ESM-3 model. After setting up a Huggingface Hub access token (if necessary), please run the following command and enter in your token when prompted:

```
huggingface-cli login
```

# Running the Code

After completing all of the steps in the previous section, running the code to reproduce our results is fairly straightforward and involves running two python scripts. The first, called `functional_embeddings.py` will generate UMAP plots of Gene Ontology (GO) function terms, demonstrating that ESM3 functional
embeddings can successfully distinguish between and relate different biological functions -- validating a core assumption behind the design of our pipeline. 

```
python3 functional_embeddings.py
```
The second script, called `masking_pipeline.py` contains our core pipeline logic and generates the plots featured in our paper. 

```
python3 masking_pipeline.py
```

And that's it! Please file an issue or reach out to us via email if you encounter any issues. 

## Citation
```
@inproceedings{pan2025mutagenic,
  title={Mutagenic: An Embedding-Based Approach to Protein Masking for Functional Redesign},
  author={Pan, Robin and Zhu, Richard Yuxuan and Lakshman, Vihan and Qu, Fiona},
  booktitle={Learning Meaningful Representations of Life (LMRL) Workshop at ICLR 2025},
  year={2025}
}
```
