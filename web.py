import numpy as np
import gradio as gr
import pandas as pd

import tiktoken
import embedding
import pickle
import openai

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"
COMPLETIONS_MODEL = "text-davinci-003"

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def vector_similarity(x: list[float], y: list[float]) -> float:
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, context_embeddings: dict[int, np.array]) -> list[(float, (str, str))]:
    query_embedding = embedding.len_safe_get_embedding(query)
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in context_embeddings.items()
    ], reverse=True)

    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        chosen_sections_len += len(encoding.encode(document_section["Content"])) + separator_len

        
        if (chosen_sections_len > MAX_SECTION_LEN) and (len(chosen_sections_indexes) > 0):
            break

        chosen_sections.append(SEPARATOR + document_section["Content"].replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections: ${chosen_sections_indexes}")

    # header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    header = """Summarize the context provided and answer the question as truthfully as possible based on the summary, and if you're not confident with the answer, please say "I don't know."\n\nContext:\n"""
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

def answer_query_with_context(query: str, df: pd.DataFrame, document_embeddings: dict[str, np.array],show_prompt: bool = False) -> str:
    prompt = construct_prompt(
      query,
      document_embeddings,
      df
    )
    
    if show_prompt:
        print(prompt)

    # return prompt
    response = openai.Completion.create(
        prompt=prompt,
        **{
        "temperature": 0.3,
        "max_tokens": 2048 + len(query.split()),
        "model": COMPLETIONS_MODEL,
    })
    return response["choices"][0]["text"].strip(" \n")

def load_all_document_embeddings(df: pd.DataFrame):
    return {
       index: r["embeddings"] for index, r in df.iterrows()
    }

def chatbot(input_text):
    df = pd.read_pickle(embedding.EMBEDDING_CACHE_PATH)
    document_embeddings = load_all_document_embeddings(df)

    answer = answer_query_with_context(input_text, df, document_embeddings, True)
    return answer

def main():
    iface = gr.Interface(fn=chatbot,
      inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
      outputs="text",
      title="Custom-trained AI Chatbot")

    iface.launch(share=False)

      
if __name__ == "__main__":
    # calling the main function
    main()
