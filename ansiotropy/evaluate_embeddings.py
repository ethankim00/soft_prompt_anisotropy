from ansiotropy.embeddings.generate_embeddings import SoftPromptEmbeddingsExtractor

from ansiotropy.metrics import get_average_mev, intra_sentence_cosine_similarity, inter_context_cosine_similarity, word_cosine_similarity

#TODO add full experimental parameters

if __name__ == "__main__":
    extractor = SoftPromptEmbeddingsExtractor(model_path = "4718005879.ckpt", dataset = "cb")
    embedding_dict = extractor.save_soft_prompt_embeddings()
    self_sim, sim_dict = word_cosine_similarity(embedding_dict, center=True)
    print(self_sim)
    print(intra_sentence_cosine_similarity(embedding_dict, center=False))
    #print(get_average_mev(embedding_dict, center=True))