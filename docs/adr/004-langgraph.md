# ADR 004 — Explicit LangGraph orchestration

## Decision

Use LangGraph `StateGraph` as the query workflow orchestrator.

## Why

The RAG has real conditional behavior—clarification, safety blocking, abstention, generation and citation validation. An explicit graph makes those paths observable and independently testable rather than hiding them inside one prompt or chain.
