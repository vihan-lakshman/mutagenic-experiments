import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

import blosum as bl
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.tokenization import InterProQuantizedTokenizer
from esm.utils.types import FunctionAnnotation
from huggingface_hub import login

from utils import get_keywords_from_interpro

blosum_matrix = bl.BLOSUM(80)

def mask_sequence(model, df, df2, embeddings_dict, selected_proteins=None):
    allnuminterpro = []
    allpercentmasks = df['percent_deleted'].tolist()
    allpercentidentities = []
    allindexes = []
    allmasked = []
    sequence_similarity = []
    masked_sequence = []
    generated_sequence_list = []
    protein_embedding_list = []
    
    df_to_iterate = selected_proteins if selected_proteins is not None else df

    for index, row in df_to_iterate.iterrows():
        if row["Entry"] not in embeddings_dict:
            continue
        if index not in df2['Index'].tolist():
            continue
        else:
            modified_prompt = df2[df2['Index']==index]['Masked Sequences'].tolist()[0]
      
        protein_prompt = ESMProtein(sequence=modified_prompt)

        # make the function annotations
        interpro_ids = embeddings_dict[df.iloc[index]['Entry']]
        functionlist = []
        for interpro_id in interpro_ids['InterPro_ids']:
            functionlist.append(FunctionAnnotation(label=interpro_id, start=1, end=len(modified_prompt)))

        # generate w/function annotations
        protein_prompt.function_annotations = get_keywords_from_interpro(functionlist)
        torch.cuda.empty_cache()
        sequence_generation = model.generate(
              protein_prompt,
              GenerationConfig(
              track="sequence",
              num_steps=protein_prompt.sequence.count("_") // 2,
              temperature=0.5,
          ),
        )

        generated_sequence = sequence_generation.sequence
        generated_sequence_list.append(generated_sequence)
        # Ensure sequences are of equal length
        if len(generated_sequence) != len(row['sequence']):
            print("Sequences must be of the same length to calculate Hamming distance.")
            sequence_similarity.append(None)
        else:
            blosum_score = 0
            for gen_residue, target_residue in zip(generated_sequence, row['sequence']):
                blosum_val =  blosum_matrix[gen_residue][target_residue]
                blosum_score += blosum_val
            blosum_score = blosum_score / len(generated_sequence)
            sequence_similarity.append(blosum_score)
        torch.cuda.empty_cache()

    return generated_sequence_list, sequence_similarity


def run_masking_pipeline(model, df, df2, df3, embeddings_dict):
    generated_sequence_list, sequence_similarity = mask_sequence(model, df, df2, embeddings_dict)
    df2['Generated Sequences'] = generated_sequence_list
    df2['Sequence Similarity'] = sequence_similarity
    selected_proteins = df[df.index.isin(df2['Index'])] 
    generated_sequence_list, sequence_similarity = mask_sequence(model, df, df3, embeddings_dict, selected_proteins=selected_proteins)
    
    df3['Generated Sequences'] = generated_sequence_list
    df3['Sequence Similarity'] = sequence_similarity
    return df2, df3


def plot_correct_versus_deleted(df2, df3):
    # Fit regression lines
    slope2, intercept2 = np.polyfit(df2['Percentage Deleted'], df2['Percent Correct'], 1)
    slope3, intercept3 = np.polyfit(df3['Percentage Deleted'], df3['Percent Correct'], 1)

    # Generate x values for regression lines
    x_vals2 = np.linspace(df2['Percentage Deleted'].min(), df2['Percentage Deleted'].max(), 100)
    y_vals2 = slope2 * x_vals2 + intercept2

    x_vals3 = np.linspace(df3['Percentage Deleted'].min(), df3['Percentage Deleted'].max(), 100)
    y_vals3 = slope3 * x_vals3 + intercept3

    # Plot scatter and regression lines
    plt.figure(figsize=(10, 6))
    plt.scatter(df2['Percentage Deleted'], df2['Percent Correct'], label='Embedding Masking Model', marker='o')
    plt.scatter(df3['Percentage Deleted'], df3['Percent Correct'], label='Random Masking Model', marker='x')

    plt.plot(x_vals2, y_vals2, color='blue', linestyle='--', label='Fit: Embedding Masking')
    plt.plot(x_vals3, y_vals3, color='orange', linestyle='--', label='Fit: Random Masking')

    plt.xlabel('Percentage Deleted')
    plt.ylabel('Percent Correct')
    plt.title('Percent Correct vs. Percentage Deleted')
    plt.legend()
    plt.grid(True)
    plt.savefig("correct_v_deleted.png")


def plot_correct_versus_similarity(df2, df3):
    # Assuming df2 and df3 are pandas DataFrames with columns 'Percentage Deleted' and 'Sequence Similarity'
    df2 = df2.dropna(subset=['Percentage Deleted', 'Sequence Similarity'])
    df3 = df3.dropna(subset=['Percentage Deleted', 'Sequence Similarity'])

    # Drop rows with -infinity values
    df2 = df2[(df2['Percentage Deleted'] != -np.inf) & (df2['Sequence Similarity'] != -np.inf)]
    df3 = df3[(df3['Percentage Deleted'] != -np.inf) & (df3['Sequence Similarity'] != -np.inf)]

    # Fit regression lines for Sequence Similarity
    slope2, intercept2 = np.polyfit(df2['Percentage Deleted'], df2['Sequence Similarity'], 1)
    slope3, intercept3 = np.polyfit(df3['Percentage Deleted'], df3['Sequence Similarity'], 1)

    # Generate x values for regression lines
    x_vals2 = np.linspace(df2['Percentage Deleted'].min(), df2['Percentage Deleted'].max(), 100)
    y_vals2 = slope2 * x_vals2 + intercept2

    x_vals3 = np.linspace(df3['Percentage Deleted'].min(), df3['Percentage Deleted'].max(), 100)
    y_vals3 = slope3 * x_vals3 + intercept3

    # Plot scatter and regression lines
    plt.figure(figsize=(10, 6))
    plt.scatter(df2['Percentage Deleted'], df2['Sequence Similarity'], label='Embedding Masking Model', marker='o')
    plt.scatter(df3['Percentage Deleted'], df3['Sequence Similarity'], label='Random Masking Model', marker='x')

    plt.plot(x_vals2, y_vals2, color='blue', linestyle='--', label='Fit: Embedding Masking')
    plt.plot(x_vals3, y_vals3, color='orange', linestyle='--', label='Fit: Random Masking')

    plt.xlabel('Percentage Deleted')
    plt.ylabel('Sequence Similarity')
    plt.title('Sequence Similarity vs. Percentage Deleted (Sum of Squares Distance)')
    plt.legend()
    plt.grid(True)
    plt.savefig("similarity_v_deleted.png")


def main():
    df = pd.read_csv('data/with_seq_similarity_and_mutant_seq_input_df_sumSquare.csv')
    df2 = pd.read_csv('data/with_seq_similarity_embedding_output_full_sumSquare.csv')
    df3 = pd.read_csv('data/with_seq_similarity_random_output_full_sumSquare.csv')

    model = ESM3.from_pretrained("esm3_sm_open_v1").to('cuda')
    embeddings_dict = np.load('data/embeddings_dict.npy',allow_pickle=True)
    embeddings_dict = dict(embeddings_dict.item())
    
    df2, df3 = run_masking_pipeline(model, df, df2, df3, embeddings_dict)

    plot_correct_versus_deleted(df2, df3)
    plot_correct_versus_similarity(df2, df3)


if __name__=='__main__':
    main()
