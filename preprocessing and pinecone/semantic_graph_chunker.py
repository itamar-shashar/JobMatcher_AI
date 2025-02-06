import math
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from networkx.algorithms import community as nx_community
import community as community_louvain  # pip install python-louvain
import spacy


class SemanticGraphChunker:
    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        window_size=5,         # Increased window size helps connect more sentences.
        penalty_alpha=0.5,     # Lower decay so distant sentences still contribute.
        sim_threshold=0.5,     # Threshold for edges creation
        min_chunk_sentences=8  # Minimum number of sentences per chunk.
    ):
        """
        Initialize the chunker.

        Args:
            model_name (str): The SentenceTransformer model name.
            window_size (int): Number of sentences before and after to consider for graph edges.
            penalty_a (float): Decay factor in the penalty exp(-a * |i-j|) for sentence distance.
            sim_threshold (float): Minimum weighted similarity to add an edge.
            min_chunk_sentences (int): Minimum sentences desired in each final chunk. Adjacent communities
                                       will be merged if needed.
        """
        self.model = SentenceTransformer(model_name)
        self.window_size = window_size
        self.penalty_a = penalty_alpha
        self.sim_threshold = sim_threshold
        self.min_chunk_sentences = min_chunk_sentences
        # Load spaCy's English model with unnecessary components disabled for speed.
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger"])

        # Pre-compute penalty factors for distances 1 through window_size.
        self.penalty = {d: math.exp(-self.penalty_a * d) for d in range(1, self.window_size + 1)}

    def chunk_text(self, text):
        """
        Chunk the input text into larger semantically coherent chunks.
        Each chunk is returned as a single string (its sentences joined together).

        Args:
            text (str): The input text to be chunked.

        Returns:
            List[str]: A list of chunk strings.
        """
        # 1. Split text into sentences.
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # 2. Compute normalized embeddings.
        embeddings = self._embed_sentences(sentences)

        # 3. Build the graph over sentences.
        graph = self._build_graph(sentences, embeddings)

        # 4. Detect communities (preliminary chunks) in the graph.
        communities = self._detect_communities(graph)

        # 5. Process and merge communities: sort by original order, then merge small adjacent chunks.
        chunks = self._process_communities(communities, sentences)

        return chunks

    def _split_sentences(self, text):
        """
        Use spaCy to segment text into sentences.

        Args:
            text (str): The input text.

        Returns:
            List[str]: List of sentences.
        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _embed_sentences(self, sentences):
        """
        Compute normalized sentence embeddings.

        Args:
            sentences (List[str]): List of sentences.

        Returns:
            np.ndarray: Array of normalized embeddings.
        """
        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        # Normalize each embedding so cosine similarity is just the dot product.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms


    def _build_graph(self, sentences, embeddings):
        """
        Build a graph where nodes are sentences and edges connect sentences within a window.
        Edge weights are the cosine similarity multiplied by a distance penalty.
        Only edges above a similarity threshold are kept.
        This version uses vectorized operations to compute similarities for each allowed offset.

        Args:
            sentences (List[str]): List of sentences.
            embeddings (np.ndarray): Normalized embeddings.

        Returns:
            networkx.Graph: The constructed graph.
        """
        G = nx.Graph()
        n = len(sentences)

        # Add nodes.
        for i, sentence in enumerate(sentences):
            G.add_node(i, sentence=sentence)

        # Use vectorized computations over offsets.
        edge_list = []
        for d in range(1, self.window_size + 1):
            if n - d <= 0:
                break

            # Compute cosine similarities for all pairs separated by offset d.
            # embeddings[:-d] and embeddings[d:] both have shape (n-d, embedding_dim).
            sims = np.sum(embeddings[:-d] * embeddings[d:], axis=1)  # vectorized dot product
            # Apply distance penalty.
            weighted_sims = sims * self.penalty[d]

            # Get indices where the weighted similarity exceeds the threshold.
            valid_idx = np.where(weighted_sims >= self.sim_threshold)[0]
            if valid_idx.size > 0:
                i_indices = valid_idx
                j_indices = valid_idx + d
                weights = weighted_sims[valid_idx]
                # Convert the arrays to a list of edge tuples.
                edges = list(zip(i_indices.tolist(), j_indices.tolist(), weights.tolist()))
                edge_list.extend(edges)

        G.add_weighted_edges_from(edge_list)
        return G

    def _detect_communities(self, graph):
        """
        Use the Louvain algorithm to detect communities in the graph.

        Args:
            graph (networkx.Graph): The sentence graph.

        Returns:
            List[set]: A list of sets where each set contains node indices belonging to a community.
        """
        # Compute partition using Louvain.
        partition = community_louvain.best_partition(graph, weight="weight")
        # Aggregate nodes by community id.
        communities = {}
        for node, comm in partition.items():
            communities.setdefault(comm, set()).add(node)
        return list(communities.values())


    def _process_communities(self, communities, sentences):
        """
        Process communities by sorting sentence indices, merging adjacent communities if they are too small,
        and then joining the sentences into a coherent string for each chunk.

        Args:
            communities (List[set]): Communities from the graph.
            sentences (List[str]): Original sentences.

        Returns:
            List[str]: Final chunk strings.
        """
        # Convert each community to a sorted list.
        sorted_chunks = [sorted(list(comm)) for comm in communities]
        sorted_chunks.sort(key=lambda chunk: chunk[0])

        # Merge adjacent chunks if they are contiguous and too small.
        merged_chunks = self._merge_small_chunks(sorted_chunks, self.min_chunk_sentences)

        # Join sentences in each chunk.
        chunks = [" ".join(sentences[i] for i in chunk) for chunk in merged_chunks]
        return chunks

    def _merge_small_chunks(self, sorted_chunks, min_size):
        """
        Merge adjacent (contiguous) communities if one of them is smaller than the minimum size.
        This procedure runs iteratively until no further merges occur.

        Args:
            sorted_chunks (List[List[int]]): List of communities as sorted lists of indices.
            min_size (int): Minimum number of sentences per chunk.

        Returns:
            List[List[int]]: Merged communities.
        """
        merged = sorted_chunks[:]  # Make a copy.
        changed = True
        while changed:
            changed = False
            new_merged = []
            i = 0
            while i < len(merged):
                # If there is a next chunk and the two are contiguous,
                # and if the current chunk is smaller than the minimum size,
                # then merge them.
                if (i < len(merged) - 1 and merged[i][-1] + 1 == merged[i + 1][0] and len(merged[i]) < min_size):
                    new_merged.append(merged[i] + merged[i + 1])
                    i += 2
                    changed = True
                else:
                    new_merged.append(merged[i])
                    i += 1
            merged = new_merged
        return merged
