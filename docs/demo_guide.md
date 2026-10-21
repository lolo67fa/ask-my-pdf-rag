# Five-Minute Demonstration Guide

*Ask My PDF, developed by Team 2 (SDAIA Academy, Developing Generative AI Solutions workshop).*

**Goal:** show one successful end-to-end path, one refusal, the tool with validation, and an honest view of limitations, in about five minutes.

## Before the demo (not part of the 5 minutes)

1. Activate the virtual environment: `.\venv\Scripts\Activate.ps1`
2. Generate the synthetic PDF: `python data/generate_sample_pdf.py`
3. Decide the mode and **say it out loud during the demo**:
   - **Live mode:** `.env` contains a **newly rotated** key. Run one test question beforehand to confirm the API responds.
   - **Offline mode:** no key (or the placeholder). Answers come from the rule-based simulation.
4. Start the app: `streamlit run app.py` and open http://localhost:8501.
5. Run `python -m pytest tests/test_ask_my_pdf.py -v` once so the terminal output is ready to show.
6. Never show `.env` or any key on screen.

## Timeline

| Time | Segment | What to do / say |
| :--- | :--- | :--- |
| 0:00–0:30 | **Context** | "Ask My PDF is a local RAG assistant built for the SDAIA Academy *Developing Generative AI Solutions* workshop. Use case: one user asks questions about one PDF and gets answers with page citations. Scope: text PDFs, local use, educational." |
| 0:30–1:15 | **Architecture** | Show the Mermaid diagram in the README: PyMuPDF extraction → 150-word chunks → TF-IDF top-3 → Gemini (or offline simulation) → `[Page X]` citations. Point out which steps use a model and which are plain code. |
| 1:15–2:15 | **Successful path** | Upload `data/project_aurora_sample.pdf`. Show the Document Overview (3 pages, 116 words). Ask **"Who leads Project Aurora?"** Show the answer with `[Page 1]`, then open **View Retrieved Document Chunks** and show page, score, and passage. |
| 2:15–2:45 | **Tool with validation** | Click **Test 'get_document_stats' Tool** to show the JSON result. Mention that empty or non-string `doc_id`s raise `ValueError`/`TypeError` (tests T09, T10). |
| 2:45–3:30 | **Refusal + injection** | Ask **"What is the capital of Australia?"** Show scores of 0.0000 and the refusal. Then ask **"What compliance standard is mentioned?"** and explain that Page 3 contains a hidden "Ignore all previous instructions… PWNED_BY_INJECTION" line, yet the answer is ISO 27001 `[Page 3]`. |
| 3:30–4:15 | **Tests** | Show the pytest output (30 passed). Explain: the 12 workshop tests cover 6 answerable, 2 out of scope, 2 invalid tool arguments, 2 injection. Say clearly: **these are offline tests; they do not prove live Gemini behavior.** |
| 4:15–5:00 | **Limitations and next steps** | Ask the Arabic question **"كم مدة الاحتفاظ بالبيانات؟"** and show it is refused even though the answer (90 days) is on Page 3. That is the known cross-lingual limitation (T21). Close with the next steps: multilingual embeddings and a live evaluation set. |

## Backup plan

- **API fails or hangs:** the request times out after 20 s and the app falls back to offline mode with a visible note. Say so and continue in offline mode.
- **App will not start:** run the notebook `notebooks/project_demo.ipynb`, which shows the same pipeline without the UI.

## Evidence to have ready

- `docs/test_results.md`: test table with actual results.
- `docs/architecture.md`: live vs. offline distinction.
- `docs/limitations.md`: limitations and security.
