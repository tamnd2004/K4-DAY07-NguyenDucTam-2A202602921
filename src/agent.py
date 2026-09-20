from typing import Callable

from .store import EmbeddingStore

NO_CONTEXT_ANSWER = (
    "Không tìm thấy ngữ cảnh liên quan trong cơ sở tri thức, "
    "nên tôi không trả lời câu hỏi này."
)

# Written in Vietnamese because the knowledge base is a Vietnamese policy corpus.
PROMPT_TEMPLATE = """Trả lời câu hỏi CHỈ dựa trên ngữ cảnh bên dưới, không dùng kiến thức bên ngoài.
Nếu ngữ cảnh không chứa câu trả lời, hãy nói rõ là không tìm thấy — tuyệt đối không suy đoán.
Mỗi ý trong câu trả lời phải kèm số trích dẫn của đoạn đã dùng, ví dụ [1] hoặc [2].

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            # Nothing retrieved: any answer would be invention, and calling the
            # LLM would spend a request to produce it.
            return NO_CONTEXT_ANSWER
        prompt = PROMPT_TEMPLATE.format(context=self._build_context(results), question=question)
        return self.llm_fn(prompt)

    def _build_context(self, results: list[dict]) -> str:
        # Numbering each chunk and naming its source file is what lets the answer
        # be traced back to a document — the Source Traceability criterion in
        # docs/EVALUATION.md, which a policy corpus cannot do without.
        return "\n\n".join(
            f"[{position}] nguồn={result['metadata'].get('doc_id', 'unknown')} "
            f"(score={result['score']:.3f})\n{result['content']}"
            for position, result in enumerate(results, start=1)
        )
