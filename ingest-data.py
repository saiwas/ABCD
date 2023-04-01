import pickle
import pandas as pd
import embedding

def compute_doc_embeddings(df: pd.DataFrame):
    embeddings_arr = []
    for _, r in df.iterrows():
        embeddings_arr.append(embedding.len_safe_get_embedding(r["Content"]))

    df["embeddings"] = embeddings_arr
    return df

def main():
    # covert pdf to txt
    df = embedding.pdf_to_txt("data/GDPR.pdf")

    # Generate embedding vector
    print("Generating embedding vector....")
    document_embeddings_df = compute_doc_embeddings(df)

    print("Saving embedding vector to cache files ....")
    document_embeddings_df.to_csv("cache/test.csv")
    document_embeddings_df.to_pickle(path=embedding.EMBEDDING_CACHE_PATH, protocol=pickle.HIGHEST_PROTOCOL)
  
if __name__ == "__main__":
    # calling the main function
    main()