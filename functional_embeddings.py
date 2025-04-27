import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm
from umap import UMAP

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.tokenization import InterProQuantizedTokenizer
from esm.utils.types import FunctionAnnotation
from huggingface_hub import login

from utils import get_label_embedding, get_keywords_from_interpro

GO_TERM_TO_INTERPRO_IDS = {'GO:0009055': [], 
                           'GO:0008270': [], 
                           'GO:0005179': [], 
                           'GO:0003677': []}

GO_TERM_TO_NAMES = {'GO:0009055': 'electron transfer activity',
                     'GO:0008270': 'zinc ion binding',
                     'GO:0005179': 'hormone activity',
                     'GO:0003677': 'DNA binding'
                    }

def read_interpro_data():
    df = pd.read_csv('data/InterProDescriptions.tsv', sep='\t')
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        go_terms = str(row['GO Terms']).split(',')
        for go_term in go_terms:
            if go_term in GO_TERM_TO_INTERPRO_IDS:
                GO_TERM_TO_INTERPRO_IDS[go_term].append(row['Accession'])

    interpro_function_annotations = [FunctionAnnotation(label="IPR011992", start=1, end=1),]
    return interpro_function_annotations


def embed_go_terms(model, sequences):
    # Dictionary to store embeddings for all sequences by GO term
    go_term_to_embeddings = {}

    # Process GO terms for each sequence
    for go_term, interpro_ids in GO_TERM_TO_INTERPRO_IDS.items():
        embeddings_by_sequence = {seq: [] for seq in sequences}
        for idx, interpro_id in tqdm(enumerate(interpro_ids)):
            for seq in sequences:
                embedding = get_label_embedding(model, interpro_id, sequence=seq)
                if embedding is not None and not np.allclose(embedding, 0):
                    embeddings_by_sequence[seq].append(embedding)
            
        go_term_to_embeddings[go_term] = embeddings_by_sequence
    
    return go_term_to_embeddings


def plot_results(sequences, go_term_to_embeddings):
    for seq in sequences:
        all_embeddings = []
        go_terms_for_embeddings = []

        # Collect embeddings for the current sequence
        for go_term, embeddings_by_sequence in go_term_to_embeddings.items():
            for embedding in embeddings_by_sequence[seq]:
                all_embeddings.append(embedding)
                go_terms_for_embeddings.append(go_term)

        if all_embeddings:
            reducer = UMAP(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(np.array(all_embeddings))

            plt.figure(figsize=(10, 8))
            for go_term in set(go_terms_for_embeddings):
                indices = [i for i, term in enumerate(go_terms_for_embeddings) if term == go_term]
                plt.scatter(
                   reduced_embeddings[indices, 0],
                   reduced_embeddings[indices, 1],
                   label=GO_TERM_TO_NAMES.get(go_term, go_term),
                   alpha=0.4,
                   s=8
                )

            plt.legend()
            plt.title(f"UMAP of ESM3 Embeddings with Sequence '{seq}' Colored by GO Term")
            plt.savefig(f"umap_{seq}.png")


def main():
    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda") # or "cpu"
    interpro_function_annotations = read_interpro_data()
    
    # List of sequences to generate embeddings for
    sequences = ["A", "AAAAAAAAAA", "G", "GGGGGGGGGG"]
    
    go_term_to_embeddings = embed_go_terms(model, sequences)
    plot_results(sequences, go_term_to_embeddings)


if __name__ == "__main__":
    main()
