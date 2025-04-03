from dataclasses import dataclass
from typing import List

import clip
import faiss
import numpy as np


@dataclass
class RetrievalResult:
  """
  Class representing result item from text-image retrieval.
  """
  img_path: str
  distance: float

  def __str__(self) -> str:
    """
    Override to string for human readable output.

    :return: String representation of object
    """
    return f"Result:\n  img_path: {self.img_path},\n  distance: {self.distance}\n"

  def to_dict(self) -> dict:
    """
    Convert object to dictionary so that it can be saved to json.

    :return: Dictionary of object fields (all values as strings).
    """
    return {
      "img_path": self.img_path,
      "distance": str(self.distance),
    }

class TextToImageRetriever:
  """
  Class representing a text-image retrieval pipeline.
  """
  def __init__(self, embedding_model, device, embeddings):
    self.embedding_model = embedding_model
    self.device = device
    self.img_paths = np.array(list(embeddings.keys()))

    self.embeddings = embeddings  # image paths mapped to embedding tensors

    embeddings = list(self.embeddings.values())

    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    img_features = embeddings/norms
    img_features = np.vstack(img_features)

    # Build index
    d = img_features.shape[1]  # Embedding dimension
    self.index = faiss.IndexFlatIP(d)
    self.index.add(img_features)

  def retrieve(self, query, n) -> List[RetrievalResult]:
    """
    Retrieve top N elements closest to query.

    :param query: Query string
    :param n: Number of elements to retrieve
    :return: List of retrieval results.
    """
    # Get embedding of query
    tokenized_text = clip.tokenize([query]).to(self.device)
    text_features = self.embedding_model.encode_text(tokenized_text)

    # Normalize embedding
    text_features /= text_features.norm(dim=-1, keepdim=True)

    distances, indices = self.index.search(text_features.cpu().detach().numpy(), k=n)
    indices = indices[0]
    distances = distances[0]
    img_paths = self.img_paths[indices]
    results = []

    # Construct results based on metadata
    for i in range(len(img_paths)):
      distance = distances[i]
      img_path = img_paths[i]
      results.append(RetrievalResult(img_path, distance))

    return results
