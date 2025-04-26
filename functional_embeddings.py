import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from umap import UMAP

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.tokenization import InterProQuantizedTokenizer
from esm.utils.types import FunctionAnnotation
from huggingface_hub import login

GO_TERM_TO_INTERPRO_IDS = {'GO:0009055': [], 
                           'GO:0008270': [], 
                           'GO:0005179': [], 
                           'GO:0003677': []}

GO_TERM_TO_NAMES = {'GO:0009055': 'electron transfer activity',
                     'GO:0008270': 'zinc ion binding',
                     'GO:0005179': 'hormone activity',
                     'GO:0003677': 'DNA binding'
                    }


def get_keywords_from_interpro(
    interpro_annotations,
    interpro2keywords=InterProQuantizedTokenizer().interpro2keywords,
):
    keyword_annotations_list = []
    for interpro_annotation in interpro_annotations:
        keywords = interpro2keywords.get(interpro_annotation.label, [])
        keyword_annotations_list.extend([
            FunctionAnnotation(
                label=keyword,
                start=interpro_annotation.start,
                end=interpro_annotation.end,
            )
            for keyword in keywords
        ])
    
    return keyword_annotations_list


def get_label_embedding(model, interpro_label, sequence):
    hostProtein = ESMProtein(sequence=sequence)
    embedding_function = model.encoder.function_embed
    hostProtein.function_annotations = get_keywords_from_interpro(
        [FunctionAnnotation(label=interpro_label, start=1, end=len(sequence))])
    hostProtein_tensor = model.encode(hostProtein)
    device = hostProtein_tensor.function.device  # Get the device of protein2_tensor.function
    embedding_function = embedding_function.to(device)  # Move embedding_function to the device

    function_embed = torch.cat(
        [
          embed_fn(funcs.to(device)) # Ensure funcs is on the same device
          for embed_fn, funcs in zip(
              embedding_function, hostProtein_tensor.function.unbind(-1)
          )
      ],
      -1,)

    if function_embed.shape[0] >= 3:
        row_sum = function_embed.sum(dim=0)  # Sum all rows
        row_avg = row_sum / (function_embed.shape[0] - 2)  # Divide by (number of rows - 2)
        row_avg_np = row_avg.cpu().detach().type(torch.float32).numpy()
        return row_avg_np
    else:
        return None


def main():
    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda") # or "cpu"
    
    df = pd.read_csv('data/InterProDescriptions.tsv', sep='\t')
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        go_terms = str(row['GO Terms']).split(',')
        for go_term in go_terms:
            if go_term in GO_TERM_TO_INTERPRO_IDS:
                GO_TERM_TO_INTERPRO_IDS[go_term].append(row['Accession'])

    interpro_function_annotations = [FunctionAnnotation(label="IPR011992", start=1, end=1),]

    protein2 = ESMProtein(function_annotations=get_keywords_from_interpro(interpro_function_annotations))
    
    # List of standard amino acid single-letter codes
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    # Function to generate a random amino acid sequence of a given length

    def generate_random_sequence(length=10):
        return ''.join(random.choices(amino_acids, k=length))

    # List of sequences to generate embeddings for
    sequences = ["A", "AAAAAAAAAA", "G", "GGGGGGGGGG","random"]

    # Dictionary to store embeddings for all sequences by GO term
    go_term_to_embeddings = {}
    i = 0

    # Process GO terms for each sequence
    for go_term, interpro_ids in GO_TERM_TO_INTERPRO_IDS.items():
        embeddings_by_sequence = {seq: [] for seq in sequences}
        for interpro_id in interpro_ids:
            for seq in sequences:
                if seq=="random":
                    actual_seq = generate_random_sequence()
                else:
                    actual_seq = seq

                embedding = get_label_embedding(model, interpro_id, sequence=actual_seq)
                if embedding is not None:
                    embeddings_by_sequence[seq].append(embedding)
            
            if i % 100 == 0:
                print(f"Processed {i} GO terms")
            i += 1
        
    go_term_to_embeddings[go_term] = embeddings_by_sequence
    
    # Create a copy of the original dictionary
    original_go_term_to_embeddings = go_term_to_embeddings.copy()

    # Iterate through the dictionary and remove all-zero arrays
    new_go_term_to_embeddings = {}
    for go_term, embeddings_by_sequence in go_term_to_embeddings.items():
        new_embeddings_by_sequence = {}
        for sequence, embeddings in embeddings_by_sequence.items():
            new_embeddings = []
            for embedding in embeddings:
                if embedding is not None and not np.allclose(embedding, 0):
                    new_embeddings.append(embedding)
            new_embeddings_by_sequence[sequence] = new_embeddings
    new_go_term_to_embeddings[go_term] = new_embeddings_by_sequence

    go_term_to_embeddings = new_go_term_to_embeddings
    
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

if __name__ == "__main__":
    main()
