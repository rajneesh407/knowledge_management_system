import os
from backend.prompt_library import (
    summarization_prompt_text_or_table,
    description_prompt_image,
    answer_using_context_prompt,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from backend.utils import parse_docs_for_images_and_texts
from langchain_core.output_parsers import StrOutputParser
from backend.config import HF_TOKEN


def initialize_client(client_name="hugging_face"):
    if client_name == "hugging_face":
        from huggingface_hub import InferenceClient

        return InferenceClient(token=HF_TOKEN)
    else:
        raise AssertionError(f"We currently do not support client: {client_name}")


def summarization_model_text(client, input_text, model_name="llama_8b"):
    if model_name == "llama_8b":
        output = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=input_text,
            max_tokens=2048,
        )
        return output.choices[0].message.content
    else:
        raise AssertionError(f"We currently do not support model: {model_name}")


def summarization_model_image(client, base64_image, model_name="llama_11b"):
    if model_name == "llama_11b":
        output = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": description_prompt_image,
                        },
                    ],
                },
            ],
        )
        return output.choices[0].message.content
    else:
        raise AssertionError(f"We currently do not support model: {model_name}")


def get_embedding_function(model_name="all-MiniLM-L6-v2"):
    if model_name == "all-MiniLM-L6-v2":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        raise AssertionError(f"We currently do not support model: {model_name}")


def build_prompt_for_response_model(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.page_content

    answer_prompt = answer_using_context_prompt.format(
        context=context_text, question=user_question
    )

    prompt_content = [{"type": "text", "text": answer_prompt}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )
    return [{"role": "user", "content": prompt_content}]


def get_response_from_chain(retriever, model):
    return {
        "context": retriever | RunnableLambda(parse_docs_for_images_and_texts),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt_for_response_model)
            | RunnableLambda(model)
            | StrOutputParser()
        )
    )


def response_model(client, retriever, model_name="llama_11b"):
    if model_name == "llama_11b":

        def model(prompt_list):
            output = client.chat.completions.create(
                model="meta-llama/Llama-3.2-11B-Vision-Instruct",
                messages=prompt_list,
                max_tokens=800,
                temperature=0.1,
            )
            return output.choices[0].message.content

        return get_response_from_chain(retriever, model)
