import torch

from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.tokenization import InterProQuantizedTokenizer
from esm.utils.types import FunctionAnnotation

def get_label_embedding(model, interpro_label, sequence):
    hostProtein = ESMProtein(sequence=sequence)
    embedding_function = model.encoder.function_embed
    hostProtein.function_annotations = get_keywords_from_interpro(
        [FunctionAnnotation(label=interpro_label, start=1, end=len(sequence))])
    hostProtein_tensor = model.encode(hostProtein)
    device = hostProtein_tensor.function.device
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
