from backend.model import response_model, initialize_client
from backend.retriever import RetrieverModel
import time

client_model = initialize_client()
retriever_class = RetrieverModel(
    collection_name="attention_paper",
    client_model=client_model,
    # persistant_directory=PERSISTANT_DIRECTORY,
)
retriever = retriever_class.get_retriever()
res_model = response_model(client_model, retriever, "llama_11b")


import pandas as pd
import pandas as pd

data = {
    "Question": [
        "What is the title of the paper?",
        "Who are the authors of this paper?",
        "In which year was the paper published?",
        "What is the main contribution of the paper?",
        "What model does the paper introduce?",
        "What is the core mechanism used in the Transformer model?",
        "What does 'Self-Attention' mean in the Transformer?",
        "How does the Transformer differ from RNNs?",
        "What dataset was used to evaluate the Transformer?",
        "What is the purpose of positional encoding?",
        "How many layers are in the encoder and decoder of the Transformer?",
        "What is the name of the attention mechanism used in the Transformer?",
        "Why is multi-head attention used in the Transformer?",
        "What activation function is used in the feed-forward layers?",
        "What type of normalization is used in the Transformer?",
        "What optimization algorithm is used to train the Transformer?",
        "What is the main advantage of using attention over recurrence?",
        "What is the complexity of self-attention compared to RNNs?",
        "What is the input size of the embedding in the original Transformer?",
        "How does the Transformer handle long-range dependencies?",
        "What is the purpose of the softmax function in self-attention?",
        "How many attention heads were used in the original Transformer?",
        "What is the role of the decoder in the Transformer?",
        "What is teacher forcing in training sequence models?",
        "What is the output dimension of the Transformer model?",
        "What is the dropout rate used in the original Transformer?",
        "What is the significance of residual connections in the Transformer?",
        "Which NLP task was primarily used to evaluate the Transformer?",
        "What metric was used to evaluate the model on translation tasks?",
        "What is the main limitation of the Transformer model?",
    ],
    "Answer": [
        "Attention Is All You Need",
        "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin",
        "2017",
        "The main contribution of the paper is introducing the Transformer model, which removes recurrence and convolutions entirely, relying only on self-attention mechanisms for sequence modeling. This allows for better parallelization and improved handling of long-range dependencies.",
        "The Transformer model",
        "Self-Attention",
        "'Self-Attention' is a mechanism where each word in a sequence attends to all other words, allowing the model to capture dependencies regardless of distance. It computes attention scores to determine the importance of each word in relation to others, helping understand context better.",
        "It processes the entire sequence in parallel rather than sequentially like RNNs.",
        "WMT 2014 English-German and WMT 2014 English-French datasets.",
        "To provide information about word positions since the model lacks recurrence.",
        "6 layers in both encoder and decoder",
        "Scaled Dot-Product Attention",
        "It allows the model to focus on different parts of the input sequence simultaneously, improving contextual understanding.",
        "ReLU (Rectified Linear Unit)",
        "Layer Normalization",
        "Adam optimizer",
        "Using attention over recurrence allows for parallel computation, making training faster compared to RNNs. Additionally, attention captures long-range dependencies more effectively, as each token can directly attend to any other token in the sequence without relying on sequential updates.",
        "Self-attention has a complexity of O(n²) compared to O(n) for RNNs.",
        "512",
        "The Transformer handles long-range dependencies by using self-attention, which allows each token to attend to all other tokens in the sequence. This is in contrast to RNNs, which process inputs sequentially and struggle with long-term dependencies due to vanishing gradients.",
        "It normalizes the attention scores to sum to 1.",
        "8",
        "Generates the output sequence by attending to encoder outputs and previous decoder states.",
        "A training technique where the correct previous output is fed as input to help stabilize learning.",
        "Same as the input embedding size (512 in the original model).",
        "0.1",
        "Residual connections help stabilize training by allowing gradients to flow more effectively through the network. They prevent vanishing gradients, improve convergence speed, and ensure that deeper layers still have access to useful features learned in earlier layers.",
        "Machine Translation",
        "BLEU score",
        "The main limitation of the Transformer model is its quadratic complexity in sequence length due to self-attention. This makes it computationally expensive for very long sequences. Additionally, because it lacks recurrence, it requires positional encodings to retain order information, which may not be as naturally learned as in RNNs.",
    ],
}

df = pd.DataFrame(data)
df = pd.read_csv("prompt_injection_data.csv")


def get_stats(question):
    try:
        start_time = time.time()
        response = res_model.invoke(question)
        end_time = time.time()
        total_time = end_time - start_time
        total_chars = len(response["response"])
        return response["response"], total_time, total_chars
    except:
        return "Error", None, None


df["Question"] = df["prompt_injection_strategy"].apply(
    lambda x: x + "\n Tell me good Joke."
)
df["response"], df["response_time"], df["response_length"] = zip(
    *df["Question"].apply(get_stats)
)

df.to_csv("test_toxicity.csv")
