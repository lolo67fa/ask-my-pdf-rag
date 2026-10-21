# Limitations, Security Considerations, and Future Improvements

## 1. Known limitations

### Retrieval
- **Lexical retrieval only.** TF-IDF matches words, not meaning. Synonyms or paraphrases ("boss" vs. "lead") can score 0.0 even when the answer is present.
- **Cross-lingual retrieval is a small keyword lexicon.** `translate_arabic_query_to_english` maps about ten Arabic keyword groups to English search terms. Several mapped term lists include vocabulary from the synthetic test PDF (e.g. "Dr Elena Rostova", "March 15 2025", "99.4%"). On other English PDFs these extra terms are just noise. Arabic questions outside the lexicon get 0.0 similarity and are refused even when the answer exists (shown by test T21). **The earlier Arabic-to-English failure is only partly fixed.**
- **Arabic PDFs** are indexed with an English stop-word list and no Arabic normalization (e.g. alef/hamza variants), so Arabic-document retrieval quality is untested.
- **Summary routing** is keyword-based; it returns the first chunk of up to 3 pages, so long documents are summarized only from their opening pages. The 0.95 score shown in the UI is assigned, not computed.
- **Fixed parameters.** Chunk size (150 words), overlap (30), and top-k (3) are hard-coded and were not tuned on real documents.

### Extraction
- Scanned/image-only PDFs are rejected (no OCR).
- Multi-column layouts, tables, headers and footers may be extracted in a confusing order.
- Word counts are whitespace-based counts of the extracted text.

### Generation
- **Live Gemini behavior was not verified in this review.** Grounding, citation format, refusals and injection resistance in live mode rely on the system instruction; none of this is tested automatically.
- No model output is guaranteed correct. Citations can be wrong or missing even when instructed; users should check the cited page.
- The model name (`gemini-2.5-flash`) is hard-coded; model availability and pricing may change.
- Only the current question is sent to the model; there is no conversation memory.

### Offline simulation
- The offline engine is a **rule-based test double**. Its fact templates (lead, launch date, SLA, chunk size, ISO 27001) only answer questions about the synthetic Project Aurora PDF. On other PDFs it returns the first sentence of the top chunk or a refusal.
- Prompt-injection detection is a short keyword list; rephrased attacks are not detected in offline mode.
- On any non-key, non-quota API error, the app **silently falls back** to the offline engine and appends a note. A user could miss that note and think the model answered.

### Tool
- `get_document_stats` resolves **any unknown `doc_id` to the only registered document** when exactly one is registered. This is lenient on purpose but weaker than strict validation.
- The registry is a module-level dictionary. With several browser sessions on the same Streamlit server, one session's upload replaces the other's registry entry. Fine for a single local user; not suitable for multi-user deployment.

### Application
- One PDF at a time; chat resets on a new upload. Re-uploading a file with the same name is not reprocessed.
- No upload size limit beyond Streamlit's default.

## 2. Security considerations

| Topic | Current state |
| :--- | :--- |
| API key storage | Read from `.env` (gitignored) via `python-dotenv`. `.env.example` contains only a placeholder. The UI shows whether a key is present, never the key. |
| Previously exposed key | A Gemini API key was earlier shown in the app interface. **Treat it as compromised: revoke it in Google AI Studio and create a new one** before any further live use or deployment. This repository does not contain that key (checked before commit). |
| Prompt injection from PDF text | Excerpts are labeled "UNTRUSTED REFERENCE MATERIAL" in the prompt and the system instruction forbids following embedded instructions. This lowers risk but **does not eliminate it**; it is not verified against the live model. |
| Prompt injection from the user | Same system instruction (live) and a keyword guard (offline). |
| Grounding | System instruction requires answers only from excerpts with `[Page X]` citations and a fixed refusal sentence. Retrieved passages and scores are displayed so users can check the evidence. |
| Tool safety | The only tool is read-only, takes one validated string, and returns two integers. No file system, shell, or network access is exposed to the model. |
| Error handling | Invalid key and quota errors show clear messages; extraction errors are shown in the UI; 60 s request timeout. Unexpected exceptions show `str(err)` in the UI, which could reveal internal details. |
| HTML rendering | `app.py` uses `unsafe_allow_html=True` and inserts the uploaded **filename** into HTML without escaping. Low risk for a local single-user app, but should be escaped before any shared deployment. |
| Data privacy | Uploaded PDFs are processed in memory. In live mode, retrieved excerpts and the question are sent to Google's Gemini API. Do not upload confidential documents. |
| Repository | Only the synthetic demo PDF is committed; `.gitignore` blocks other PDFs, `.env`, virtual environments, caches and logs. |

## 3. Future improvements

1. **Semantic or multilingual retrieval:** multilingual embeddings (or BM25 + embeddings hybrid) to replace the Arabic keyword lexicon.
2. **Live evaluation set:** a small, opt-in pytest marker (e.g. `@pytest.mark.live`) that runs the 12 scenarios against Gemini when a key is present, recording results separately from offline tests.
3. **Strict tool validation option:** reject unknown `doc_id`s instead of falling back.
4. **Per-session registry:** store document stats in `st.session_state` instead of a module-level dict.
5. **Visible mode indicator per answer:** label each answer as "Gemini" or "Offline simulation".
6. **Escape user-controlled strings** before inserting them into HTML.
7. **OCR** for scanned PDFs, and layout-aware extraction for tables and columns.
8. **Better summaries:** sample chunks across the whole document rather than the first pages.
9. **Citation checking:** verify that every `[Page X]` in a live answer refers to a page among the retrieved chunks.
