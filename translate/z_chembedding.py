import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from rdkit import Chem

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


class ChemBERTaEmbedder:
    def __init__(self, model_name=MODEL_NAME, pooling="cls", use_cuda=False):
        """
        Initialize the ChemBERTa model and tokenizer.

        Parameters:
          model_name: Hugging Face model name or path.
          pooling: "cls" for CLS token pooling, "average" for mean pooling.
          use_cuda: Whether to use GPU acceleration if available.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.pooling = pooling

    @staticmethod
    def validate_smiles(smiles):
        """Check if the given SMILES string is valid using RDKit."""
        return Chem.MolFromSmiles(smiles) is not None

    def embed(self, smiles_input):
        """
        Compute embeddings for a single SMILES string or a list of SMILES strings.

        Parameters:
          smiles_input: A single SMILES string or a list of SMILES strings.

        Returns:
          embeddings: NumPy array of shape (N, hidden_dim).
        """
        # Convert single SMILES string to list
        if isinstance(smiles_input, str):
            smiles_input = [smiles_input]

        # Validate all SMILES
        valid_smiles = [s for s in smiles_input if self.validate_smiles(s)]
        if not valid_smiles:
            raise ValueError("No valid SMILES provided.")

        # Tokenize
        encodings = self.tokenizer.batch_encode_plus(
            valid_smiles, padding=True, add_special_tokens=True, return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            if self.pooling == "cls":
                embeddings = self.model.pooler(outputs.last_hidden_state)  # (N, hidden_dim)
            elif self.pooling == "average":
                token_embeddings = outputs.last_hidden_state  # (N, L, H)
                mask = attention_mask.unsqueeze(-1).float()  # (N, L, 1)
                sum_embeddings = torch.sum(token_embeddings * mask, dim=1)  # (N, H)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)  # (N, 1)
                embeddings = sum_embeddings / sum_mask
            else:
                raise ValueError("pooling must be either 'cls' or 'average'")

        return embeddings.cpu().numpy()


def main():
    sample_smiles = [
        "Fc1ccc(OCC2CCC3CN(c4ncc(Cl)cc4Cl)CCN3C2)cc1",
        "CCCCn1c(=O)n(CCCc2ccccc2)c(=O)c2[nH]c(Cl)nc21",
        "OCCCn1c(=O)n(CCCc2ccccc2)c(=O)c2[nH]c(Cl)nc21",
        "CCCCn1c(=O)n(CCCc2cccc2)c(=O)c2[nH]c(Cl)nc21",
        "Fc1ccc(OCC2CCC3CN(c4ncc(Br)cc4Cl)CCN3C2)cc1",
        "INVALID_SMILES"  # Invalid test case
    ]

    embedder = ChemBERTaEmbedder(use_cuda=False)

    # Test single SMILES embedding
    single_embedding = embedder.embed(sample_smiles[0])
    print("Single SMILES embedding shape:", single_embedding.shape)

    # Test multiple SMILES embedding
    try:
        embeddings = embedder.embed(sample_smiles)
        print("Embeddings shape:", embeddings.shape)
    except ValueError as e:
        print(f"Error: {e}")

    # Compute cosine distance matrix
    if "embeddings" in locals():
        distance_matrix = cosine_distances(embeddings)
        print("Cosine Distance Matrix:\n", np.round(distance_matrix, 3))

        tsne = TSNE(n_components=2, random_state=42, perplexity=3)
        embeddings_2d = tsne.fit_transform(embeddings)
        print("t-SNE Embeddings:\n", embeddings_2d)


if __name__ == "__main__":
    main()
