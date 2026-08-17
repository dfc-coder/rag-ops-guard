# ADR 004 — Explicit LangGraph orchestration

Status: **Superseded** by the generic conversational ReAct pivot (`docs/specs/generic-react-pivot.md`).

## Former decision

The initial architecture used LangGraph `StateGraph` as the query workflow orchestrator.

## Superseding decision

The product now has one canonical conversational owner: `ConversationAgent`, reached through `rag_ops_guard.app.conversation_agent()`. The legacy graph/router pipeline and `src/rag_ops_guard/graph/` package were removed rather than retained as a fallback.

LangGraph is therefore not a direct runtime dependency of this project. Conditional behavior such as safety blocking, optional document tools, clarification, segmented generation and citation validation is owned by the canonical conversational agent and covered by behavioral plus architecture-fitness tests.
