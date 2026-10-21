# Limitations, Security Considerations, and Future Improvements

*Ask My PDF, developed by Team 2 (SDAIA Academy, Developing Generative AI Solutions workshop).*

## 1. Known limitations

### Retrieval
- **Lexical retrieval only.** TF-IDF matches words, not meaning. Synonyms or paraphrases ("boss" vs. "lead") can score 0.0 even when the answer is present.
- **Cross-lingual retrieval is a small keyword lexicon.** `translate_arabic_query_to_english` maps about ten Arabic keyword groups to English search terms. Several mapped term lists include vocabulary from the synthetic test PDF (e.g. "Dr Elena Rostova", "March 15 2025", "99.4%"). On other English PDFs these extra terms are just noise. Arabic questions outside the lexicon get 0.0 similarity and are refused even when the answer exists (shown by test T21). **The earlier Arabic-to-English failure is only partly fixed.**
- **Arabic PDFs** are indexed with an English stop-word list and no Arabic normalization (e.g. alef/hamza variants), so Arabic-document retrieval quality is untested.
- **Summary routing** is pattern-based; it samples the first chunk of up to 4 pages spread across the document, so most of a long document is not read for the summary (the offline answer adds a coverage note). The 0.95 score shown in the UI is assigned, not computed.
- **Fixed parameters.** Chunk size (150 words), overlap (30), and top-k (3) are hard-coded and were not tuned on real documents.

### Extraction
- Scanned/image-only PDFs are rejected (no OCR).
- Multi-column layouts, tables, headers and footers may be extracted in a confusing order.
- Word counts are whitespace-based counts of the extracted text.

### Generation
- **Live Gemini was checked with 9 questions only** (8 completed). Grounding, citations, refusals and injection resistance rely on the system instruction and are not part of the automated tests.
- **The zero-evidence refusal rule is not strictly followed by Gemini:** in live case L08 all retrieval scores were 0.0000 and Gemini still answered. Refusing in code when every score is 0 would enforce the rule.
- **Free-tier limits:** live testing hit `429 RESOURCE_EXHAUSTED` after about 5 quick requests and `503 UNAVAILABLE` when the model was overloaded; there is no automatic retry.
- No model output is guaranteed correct. Citations can be wrong or missing even when instructed; users should check the cited page.
- The model name (`gemini-2.5-flash`) is hard-coded; model availability and pricing may change.
- Only the current question is sent to the model; there is no conversation memory.

### Offline simulation
- The offline engine is a **rule-based test double**. Its fact templates (lead, launch date, SLA, chunk size, ISO 27001) only answer questions about the synthetic Project Aurora PDF. On other PDFs it returns the first sentence of the top chunk or a refusal.
- Prompt-injection detection is a short keyword list; rephrased attacks are not detected in offline mode.
- On a timeout or any unclassified API error, the app falls back to the offline engine and appends a notice. A user could miss that notice and think the model answered.
- API errors are classified by searching the error text for "401", "403", "404" or "429"; an unrelated message that contains one of these numbers could be misclassified.

### Tool
- `get_document_stats` tolerates case, whitespace, and a missing `.pdf` in the name. (Earlier versions returned the only loaded document's stats for *any* unknown name; fixed and covered by T29.)
- The registry is a module-level dictionary. With several browser sessions on the same Streamlit server, one session's upload replaces the other's registry entry. Fine for a single local user; not suitable for multi-user deployment.

### Application
- One PDF at a time; chat resets on a new upload. Re-uploading a file with the same name is not reprocessed.
- No upload size limit beyond Streamlit's default.

## 2. Security considerations

| Topic | Current state |
| :--- | :--- |
| API key storage | Read from `.env` (gitignored) via `python-dotenv`. `.env.example` contains only a placeholder. The UI shows whether a key is present, never the key. |
| Previously exposed key | A Gemini API key was earlier shown in the app interface. The team **revoked it and replaced it** with a new key kept only in the local `.env`. Neither key is in this repository or its history (checked before each push). |
| Prompt injection from PDF text | Excerpts are labeled "UNTRUSTED REFERENCE MATERIAL" in the prompt and the system instruction forbids following embedded instructions. This lowers risk but **does not eliminate it**; Gemini ignored the embedded instruction in one live test (L06). |
| Prompt injection from the user | Same system instruction (live) and a keyword guard (offline). |
| Grounding | System instruction requires answers only from excerpts with `[Page X]` citations and a fixed refusal sentence. Retrieved passages and scores are displayed so users can check the evidence. |
| Tool safety | The only tool is read-only, takes one validated string, and returns two integers. No file system, shell, or network access is exposed to the model. |
| Error handling | 401/403/404/429 errors show clear messages; extraction errors are shown in the UI; 20 s request timeout; single tool round-trip with no loop. Unexpected exceptions show `str(err)` in the UI, which could reveal internal details. |
| HTML rendering | `app.py` uses `unsafe_allow_html=True` and inserts the uploaded **filename** into HTML without escaping. Low risk for a local single-user app, but should be escaped before any shared deployment. |
| Data privacy | Uploaded PDFs are processed in memory. In live mode, retrieved excerpts and the question are sent to Google's Gemini API. Do not upload confidential documents. |
| Repository | Only the synthetic demo PDF is committed; `.gitignore` blocks other PDFs, `.env`, virtual environments, caches and logs. |

## 3. Future improvements

1. **Semantic or multilingual retrieval:** multilingual embeddings (or BM25 + embeddings hybrid) to replace the Arabic keyword lexicon.
2. **Live evaluation set:** a small, opt-in pytest marker (e.g. `@pytest.mark.live`) that runs the 12 scenarios against Gemini when a key is present, recording results separately from offline tests.
3. **Retry on temporary API errors:** retry once on 503 (model overloaded) before falling back to offline mode.
4. **Per-session registry:** store document stats in `st.session_state` instead of a module-level dict.
5. **Visible mode indicator per answer:** label each answer as "Gemini" or "Offline simulation".
6. **Escape user-controlled strings** before inserting them into HTML.
7. **OCR** for scanned PDFs, and layout-aware extraction for tables and columns.
8. **Better summaries:** use more than four sampled chunks (e.g. map-reduce summarization) for long documents.
9. **Citation checking:** verify that every `[Page X]` in a live answer refers to a page among the retrieved chunks.
