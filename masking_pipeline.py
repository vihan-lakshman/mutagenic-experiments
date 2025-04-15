import pandas as pd
df = pd.read_csv('data/with_seq_similarity_and_mutant_seq_input_df_sumSquare.csv')
df2 = pd.read_csv('data/with_seq_similarity_embedding_output_full_sumSquare.csv')
df3 = pd.read_csv('data/with_seq_similarity_random_output_full_sumSquare.csv')
print(df.columns)
print(df2.columns)
print(df3.columns)
print(df2['Masked Sequences'].head())
df2_full_exists = True
df3_full_exists = True

import torch
from sklearn.metrics.pairwise import cosine_similarity

import blosum as bl
matrix = bl.BLOSUM(80)

# Dictionary to hold the results
embeddings_dict = {}

# Iterate through each row in the DataFrame
for _, row in df.iterrows():
    entry = row['Entry']
    interpro = row['InterPro']

    # Skip rows where 'Interpro' is None
    if pd.isna(interpro) or not interpro.strip():
        continue

    # Split the InterPro IDs by semicolons
    interpro_ids = interpro.split(';')
    interpro_ids = interpro_ids[:-1]

    # Initialize entry in the dictionary if not present
    if entry not in embeddings_dict:
        embeddings_dict[entry] = {
            'InterPro_ids': interpro_ids
        }

from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

login()

model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to('cuda')

import torch.nn as nn
from esm.tokenization import InterProQuantizedTokenizer
from esm.utils.types import FunctionAnnotation
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

def get_label_embedding(interpro_label,sequence):
  hostProtein = ESMProtein(sequence=sequence)
  embedding_function = model.encoder.function_embed
  hostProtein.function_annotations = get_keywords_from_interpro([FunctionAnnotation(label=interpro_label, start=1, end=len(sequence))])
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
      -1,
  )
  if function_embed.shape[0] >= 3:
      row_sum = function_embed.sum(dim=0)  # Sum all rows
      row_avg = row_sum / (function_embed.shape[0] - 2)  # Divide by (number of rows - 2)
      row_avg_np = row_avg.cpu().detach().type(torch.float32).numpy()
      return row_avg_np
  else:
      return None

import numpy as np
embeddings_dict = np.load('data/embeddings_dict.npy',allow_pickle=True)
embeddings_dict = dict(embeddings_dict.item())

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def calculatesquareddist(proteinembed, embedding_np):
    # Ensure inputs are numpy arrays for compatibility with cosine_similarity
    proteinembed = np.array(proteinembed)
    embedding_np = np.array(embedding_np)

    squared_similarities = []

    # Iterate over each row in proteinembed
    for protein_row in proteinembed:
        # Compute cosine similarity of the current row with all rows of embedding_np
        cos_sim = cosine_similarity(embedding_np, protein_row.reshape(1, -1)) ** 2
        # Sum the squared cosine similarities
        squared_sum = np.sum(cos_sim)
        squared_similarities.append(squared_sum)

    # Convert results to a tensor for consistency if needed
    return squared_similarities

def embedding_masking_model(
    prompt,
    model,
    df,
    embeddings_dict,
    percentage=10,
):
    """
    Helper function to process a protein sequence, calculate similarities,
    and return indices for masking.

    Args:
        prompt (str): The protein sequence to be processed.
        model: The model used for protein generation and embeddings.
        df (pd.DataFrame): DataFrame containing protein data.
        embeddings_dict (dict): Dictionary storing embeddings and other details.

    Returns:
        List[int]: Indices used for masking in the sequence.
    """
    # Create an ESMProtein object
    protein = ESMProtein(sequence=prompt)

    # Configure the model for generation
    generation_config = GenerationConfig(track="function", num_steps=8)

    # Generate the protein
    generated_protein = model.generate(protein, generation_config)

    # Check if function annotations are available
    entry = df.loc[df['substituted_seq'] == prompt, 'Entry'].iloc[0]
    if generated_protein.function_annotations is None:
        embeddings_dict[entry]['hamming_distance'] = None
        return [],[]

    # Getting embedding for the protein
    protein_tensor = model.encode(generated_protein)
    embedding_function = model.encoder.function_embed
    device = protein_tensor.function.device  # Get the device of protein_tensor.function
    embedding_function = embedding_function.to(device)  # Move embedding_function to the device

    function_embed = torch.cat(
        [
            embed_fn(funcs.to(device))  # Ensure funcs is on the same device
            for embed_fn, funcs in zip(
                embedding_function, protein_tensor.function.unbind(-1)
            )
        ],
        -1,
    )

    # Exclude start and end tokens
    function_embed = function_embed[1:-1, :]

    # Convert the protein_tensor.function to a NumPy array
    protein_np = function_embed.cpu().detach().type(torch.float32).numpy()

    # Retrieve target sequence and embedding
    embedding = embeddings_dict[entry]['embedding']

    # Calculate cosine similarity
    #embedding = embedding.cpu().detach().type(torch.float32).numpy()
    similarities = calculatesquareddist(protein_np, embedding)

    num_indices = int(len(prompt) * percentage / 100)

    # Ensure we select at least 1 index
    num_indices = max(1, num_indices)

    # Find the top 10 indices with the lowest cosine similarity
    indices = np.argsort(similarities)[:num_indices]

    # Store the indices in the embeddings_dict
    embeddings_dict[entry]['indices'] = indices.tolist()

    return indices.tolist(), protein_np

import math
row = df.iloc[6]
maskedindeces,embedding = embedding_masking_model(row['substituted_seq'], model, df, embeddings_dict,percentage=math.ceil(row['percent_deleted']))
print(embedding.shape)

def get_random_indices(prompt, percentage):
    """
    Randomly select indices to mask based on the percentage of the prompt's length.
    """
    num_indices = int(len(prompt) * percentage / 100)
    # Ensure we select at least one index
    num_indices = max(1, num_indices)

    # Randomly select unique indices to mask
    return random.sample(range(len(prompt)), num_indices)

import math
allnuminterpro = []
allpercentmasks = df['percent_deleted'].tolist()
allpercentidentities = []
allindexes = []
allmasked = []
sequence_similarity = []
masked_sequence = []
generated_sequence_list = []
protein_embedding_list = []
for index, row in df.iterrows():
  if row["Entry"] not in embeddings_dict:
    continue
  if not df2_full_exists:
    maskedindeces, protein_embedding = embedding_masking_model(row['substituted_seq'], model, df, embeddings_dict,percentage=math.ceil(row['percent_deleted']))
    if not maskedindeces:
      continue
    protein_embedding_list.append(protein_embedding)
    allindexes.append(index)
    numinterpro = int(len(row['InterPro'])/10)
    allnuminterpro.append(numinterpro)
    correctmasks = set(np.arange(row['del_start'],row['del_end']+1))
    truncatedpredictions = set(maskedindeces[:len(correctmasks)])
    allmasked.append(truncatedpredictions)
    identical_count = len(truncatedpredictions.intersection(correctmasks))
    percent_identity = (identical_count / len(correctmasks))
    allpercentidentities.append(percent_identity)
    modified_prompt = list(row['substituted_seq'])
    for index in maskedindeces:
        modified_prompt[index] = "_"
    modified_prompt = "".join(modified_prompt)
    masked_sequence.append(modified_prompt)
  else:
    if index not in df2['Index'].tolist():
      continue
    else:
      modified_prompt = df2[df2['Index']==index]['Masked Sequences'].tolist()[0]
  protein_prompt = ESMProtein(sequence=modified_prompt)

  #make the function annotations
  interpro_ids = embeddings_dict[df.iloc[index]['Entry']]
  functionlist = []
  for interpro_id in interpro_ids['InterPro_ids']:
    functionlist.append(FunctionAnnotation(label=interpro_id, start=1, end=len(modified_prompt)))


  #generate w/function annotations
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
          blosum_val =  matrix[gen_residue][target_residue]
          blosum_score += blosum_val
      blosum_score = blosum_score / len(generated_sequence)
      sequence_similarity.append(blosum_score)
  torch.cuda.empty_cache()

if not df2_full_exists:
  shortenedpercentmasks = [allpercentmasks[i] for i in allindexes]
  df2 = pd.DataFrame({
      'Number of Interpro Terms': allnuminterpro,
      'Percentage Deleted': shortenedpercentmasks,
      'Percent Correct': allpercentidentities,
      'Index': allindexes,
      'Masked sites': allmasked,
      'Sequence Similarity': sequence_similarity,
      'Generated Sequences': generated_sequence_list,
      'Masked Sequences': masked_sequence
  })
else:
  df2['Generated Sequences'] = generated_sequence_list
  df2['Sequence Similarity'] = sequence_similarity

# Save the DataFrame as a CSV file
df2.to_csv('with_seq_similarity_embedding_output_full.csv')

allnuminterpro = []
allpercentmasks = df['percent_deleted'].tolist()
allpercentidentities = []
allindexes = []
allmasked = []
sequence_similarity = []
masked_sequence = []
generated_sequence_list = []
selected_proteins = df[df.index.isin(df2['Index'])]
for index,row in selected_proteins.iterrows():
  if row["Entry"] not in embeddings_dict:
    continue
  if not df3_full_exists:
    maskedindeces = get_random_indices(row['substituted_seq'], percentage=math.ceil(row['percent_deleted']))
    if not maskedindeces:
      continue
    allindexes.append(index)
    numinterpro = int(len(row['InterPro'])/10)
    allnuminterpro.append(numinterpro)
    correctmasks = set(np.arange(row['del_start'],row['del_end']+1))
    truncatedpredictions = set(maskedindeces[:len(correctmasks)])
    allmasked.append(truncatedpredictions)
    identical_count = len(truncatedpredictions.intersection(correctmasks))
    percent_identity = (identical_count / len(correctmasks))
    allpercentidentities.append(percent_identity)
    modified_prompt = list(row['substituted_seq'])
    for index in maskedindeces:
        modified_prompt[index] = "_"
    modified_prompt = "".join(modified_prompt)
    masked_sequence.append(modified_prompt)
  else:
    if index not in df3['Index'].tolist():
      continue
    else:
      modified_prompt = df3[df3['Index']==index]['Masked Sequences'].tolist()[0]
  protein_prompt = ESMProtein(sequence=modified_prompt)
  #make the function annotations
  interpro_ids = embeddings_dict[df.iloc[index]['Entry']]
  functionlist = []
  for interpro_id in interpro_ids['InterPro_ids']:
    functionlist.append(FunctionAnnotation(label=interpro_id, start=1, end=len(modified_prompt)))


  #generate w/function annotations
  protein_prompt.function_annotations = get_keywords_from_interpro(functionlist)
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
          blosum_val =  matrix[gen_residue][target_residue]
          blosum_score += blosum_val
      blosum_score = blosum_score / len(generated_sequence)
      sequence_similarity.append(blosum_score)

  torch.cuda.empty_cache()

if not df3_full_exists:
  shortenedpercentmasks = [allpercentmasks[i] for i in allindexes]
  df3 = pd.DataFrame({
      'Number of Interpro Terms': allnuminterpro,
      'Percentage Deleted': shortenedpercentmasks,
      'Percent Correct': allpercentidentities,
      'Index': allindexes,
      'Masked sites': allmasked,
      'Sequence Similarity': sequence_similarity,
      'Generated Sequences': generated_sequence_list,
      'Masked Sequences': masked_sequence
  })
else:
  df3['Generated Sequences'] = generated_sequence_list
  df3['Sequence Similarity'] = sequence_similarity


df3.to_csv('with_seq_similarity_random_output_full.csv')

# prompt: graph 'Percent Correct' column for df2 and df3 over the rows

import numpy as np
import matplotlib.pyplot as plt

# Assuming df2 and df3 are pandas DataFrames with columns 'Percentage Deleted' and 'Percent Correct'

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
plt.show()

import numpy as np
import matplotlib.pyplot as plt

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
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Assuming df2 and df3 are pandas DataFrames with necessary columns
df2 = df2.dropna(subset=['Percentage Deleted', 'Sequence Similarity', 'Number of Interpro Terms'])
df3 = df3.dropna(subset=['Percentage Deleted', 'Sequence Similarity', 'Number of Interpro Terms'])

# Drop rows with -infinity values
df2 = df2[(df2['Percentage Deleted'] != -np.inf) & (df2['Sequence Similarity'] != -np.inf)]
df3 = df3[(df3['Percentage Deleted'] != -np.inf) & (df3['Sequence Similarity'] != -np.inf)]

# Define color mapping function
def map_color(value):
    if value < 2:
        return 'red'
    elif 2 <= value <= 4:
        return 'yellow'
    elif 4 < value <= 6:
        return 'green'
    else:
        return 'blue'

# Map colors for df2 and df3
colors2 = df2['Number of Interpro Terms'].apply(map_color)
colors3 = df3['Number of Interpro Terms'].apply(map_color)

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
#plt.scatter(df2['Percentage Deleted'], df2['Sequence Similarity'], c=colors2, label='Embedding Masking Model', marker='o', edgecolor='black')
plt.scatter(df3['Percentage Deleted'], df3['Sequence Similarity'], label='Random Masking Model', marker='x', edgecolor='black')
plt.scatter(df2['Percentage Deleted'], df2['Sequence Similarity'], c=colors2, label='Embedding Masking Model', marker='o', edgecolor='black')

plt.plot(x_vals2, y_vals2, color='blue', linestyle='--', label='Fit: Embedding Masking')
plt.plot(x_vals3, y_vals3, color='orange', linestyle='--', label='Fit: Random Masking')

plt.xlabel('Percentage Deleted')
plt.ylabel('Sequence Similarity')
plt.title('Sequence Similarity vs. Percentage Deleted (Sum of Squares Distance)')
plt.legend()
plt.grid(True)
plt.show()

