# AI Recruiter Production Monorepo

This repository is structured for a production-grade, modular AI Recruiter platform.

The architecture aligns with the project objective:
- Multimodal AI interviewer (NLP + Voice + Vision + Avatar)
- Agentic orchestration with explicit service boundaries
- Objective scoring and interview report generation
- HR dashboard and candidate-facing application
- Recruiter persona fine-tuning pipeline using data_v2 assets

## Top-Level Domains
- apps: Candidate and HR user interfaces
- services: Runtime backend microservices and AI agents
- platform: Shared runtime platform capabilities (memory, messaging, observability, security)
- ml: Fine-tuning, evaluation, and serving workflows for interviewer models
- api: Interface contracts by protocol (REST, gRPC, WebSocket)
- contracts: Event contracts and schema contracts
- data: Canonical data lifecycle zones (raw, interim, processed, exports)
- infrastructure: Docker, Kubernetes, Terraform, monitoring, and CI templates
- configs: Environment and feature flag configuration templates
- tests: Quality gates (unit, integration, e2e, load, safety)
- scripts: Developer, data, and release operational scripts
- docs: Architecture, runbooks, security, product, and ML documentation

## Legacy + Research Assets
- Existing data_v2 and Research folders are preserved and mapped into this structure.
- See docs/datasets/data_v2_mapping.md for migration and usage guidance.
