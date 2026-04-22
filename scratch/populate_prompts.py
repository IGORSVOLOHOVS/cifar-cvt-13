import os

prompts = {
    "02-adr.md": "# Prompt: Architectural Decision Record (ADR)\nDocument a critical architectural decision using the MADR template. Explain the context, options considered, and the final justification based on quality attribute trade-offs.",
    "03-c4-modeling.md": "# Prompt: C4 System Modeling\nCreate Mermaid.js diagrams for C4 Level 1 (System Context), Level 2 (Container), and Level 3 (Component). Focus on boundaries and communication protocols.",
    "04-skeleton.md": "# Prompt: Technical Skeleton & Config\nGenerate the project structure and primary configuration files (CMakeLists.txt or pyproject.toml). Ensure strict linting and build standards are initialized.",
    "05-ddd-modeling.md": "# Prompt: Domain-Driven Design Modeling\nIdentify Value Objects, Entities, and Aggregates. Define the Ubiquitous Language and Bounded Contexts for the core domain.",
    "06-api-contracts.md": "# Prompt: API & Contract Design\nDesign the public interface/API. Focus on header-only definitions (C++) or abstract base classes (Python) to establish strict contracts before implementation.",
    "07-docker-ci.md": "# Prompt: Docker & CI Pipeline\nCreate a multi-stage Dockerfile and a GitHub Actions / GitLab CI pipeline. Focus on reproducible builds and automated quality gates.",
    "08-testing-infra.md": "# Prompt: Testing Infrastructure Setup\nConfigure the testing framework (doctest or pytest). Set up mock objects, test fixtures, and coverage measurement tools.",
    "09-functional-core.md": "# Prompt: Atomic Functional Core\nImplement the core business logic as a set of pure, stateless functions. Avoid side effects and ensure maximum testability.",
    "10-monadic-errors.md": "# Prompt: Monadic Error Flow\nIntegrate `std::expected` (C++) or Result patterns (Python) into the logic. Ensure all error paths are handled explicitly without exceptions.",
    "11-concurrency-state.md": "# Prompt: Concurrency & Message Passing\nImplement state management using the Actor Model or Message Passing. Avoid shared-state mutexes in favor of thread isolation.",
    "12-imperative-shell.md": "# Prompt: Imperative Shell & Adapters\nImplement adapters for external systems (Database, Network, File System). Ensure isolation from the Functional Core.",
    "13-observability.md": "# Prompt: Structured Logging & Telemetry\nIntegrate structured logging and performance telemetry. Ensure every major event is traceable and measurable.",
    "14-perf-baselines.md": "# Prompt: Performance Budgeting\nEstablish performance budgets and baselines for critical paths using micro-benchmarking tools (nanobench or pytest-benchmark).",
    "15-optimization.md": "# Prompt: High-Performance Honing\nAnalyze and optimize critical paths for cache locality, SIMD utilization, or algorithmic complexity.",
    "16-security-audit.md": "# Prompt: Comprehensive Security Audit\nRun static and dynamic security scans (Bandit, Sanitizers). Audit for common vulnerabilities (OWASP, Buffer Overflows, Python Injection).",
    "17-lint-cleanup.md": "# Prompt: Technical Debt & Smells Audit\nIdentify and resolve code smells, duplication, and technical debt. Ensure strict adherence to the project's Clean Code rules.",
    "18-modernization.md": "# Prompt: Language Standard Upgrade\nVerify that the codebase fully utilizes the latest language features (C++23 Ranges/Concepts or Python 3.12+ type features).",
    "19-docs-auto.md": "# Prompt: Automated API Documentation\nGenerate professional API documentation (Doxygen or pdoc). Ensure all public interfaces are fully documented with examples.",
    "20-readme-crafting.md": "# Prompt: User-Centric Documentation\nWrite high-quality READMEs, Tutorials, and Installation guides. Focus on the user experience and clear onboarding.",
    "21-validation.md": "# Prompt: Full Validation Sweep\nRun the complete verification suite (Format -> Lint -> Test -> Build). Ensure zero regressions before merging.",
    "22-compliance.md": "# Prompt: Licensing & Standards Audit\nAudit for license compliance (OSS scanners) and adherence to industry standards (MISRA, PEP8, ISO).",
    "23-packaging.md": "# Prompt: Distribution Artifact Generation\nConfigure packaging (Conan, CPack, or Wheels). Build and verify final distributable artifacts.",
    "24-release-prep.md": "# Prompt: Release & Changelog Automation\nGenerate a detailed changelog and release notes. Automate version tagging and deployment notifications.",
    "25-handoff.md": "# Prompt: Final Project Handoff\nPerform a deep \"Brain Extraction\" of the entire project status. Provide a comprehensive summary for the next maintenance session."
}

base_path = "workflow_prompts"
for filename, content in prompts.items():
    with open(os.path.join(base_path, filename), "w", encoding="utf-8") as f:
        f.write(content)

print(f"Created {len(prompts)} prompt files in {base_path}")
