QUERY_INSTRUCTION = (
    "Instruct: Given an integration operations question, retrieve authoritative runbooks, "
    "API documentation, incident reports, postmortems, SLAs, and architecture notes that "
    "provide evidence for the answer.\n\nQuery: {question}"
)


def embedding_query(question: str) -> str:
    return QUERY_INSTRUCTION.format(question=question.strip())
