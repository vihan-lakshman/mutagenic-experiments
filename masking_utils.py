
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
