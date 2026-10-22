# Deployment Guide: Streamlit Community Cloud

*Ask My PDF, developed by Team 2 (SDAIA Academy, Developing Generative AI Solutions workshop).*

**Status:** prepared, **not deployed**. Nothing has been published, no cloud resources were created, and no secrets were uploaded.

## 1. Readiness checks (2026-10-22)

| Check | How it was verified | Result |
| :--- | :--- | :--- |
| Entry point | Repository layout | `app.py` at the repository root |
| Dependencies resolve on the cloud platform | `pip download` of `requirements.txt` for Linux x86_64, Python 3.12, binary wheels only | All pinned packages resolve (e.g. `pymupdf-1.28.2-cp310-abi3-manylinux_2_28`, `scikit_learn-1.7.2-cp312-manylinux2014`) |
| Versions match the tested environment | `requirements.txt` pins the exact versions used for the 35 passing tests | Pinned: streamlit 1.65.0, pymupdf 1.28.2, scikit-learn 1.7.2, numpy 2.2.6, google-genai 2.32.0, python-dotenv 1.2.4, pytest 9.1.1 |
| No Windows-specific code | Search for drive letters, backslash paths, Windows-only modules | None found; all paths use `pathlib` relative to the app file |
| Logo loads | Headless `AppTest` run | Rendered; file name `Sdaia-Academy.png` matches exactly (Linux is case-sensitive) |
| Key from Streamlit secrets | Headless `AppTest` with a fake secret and no environment variable | Detected; key never displayed |
| Missing key | Headless `AppTest` with no key | App starts in offline mode with a clear warning |
| Upload size | `.streamlit/config.toml` | Limited to 25 MB |
| App on Linux itself | **Not tested** (no Linux environment available) | Expected to work based on the checks above |

> Why pinned versions: without pins, the cloud would install PyMuPDF 2.0, scikit-learn 1.9 and NumPy 2.5, which were never tested with this code.

## 2. Deploy step by step

1. Go to **https://share.streamlit.io** and sign in with the GitHub account that owns `lolo67fa/ask-my-pdf-rag`.
2. When prompted, allow Streamlit to access **private repositories** (required because the repository is private).
3. Click **Create app** → deploy from a GitHub repository, and enter:
   - **Repository:** `lolo67fa/ask-my-pdf-rag`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings**:
   - **Python version:** 3.12 (the version the dependencies were checked against).
   - **Secrets:** paste exactly one line, with your real key:
     ```toml
     GEMINI_API_KEY = "your-new-key"
     ```
     The format is shown in `.streamlit/secrets.toml.example`. Never put the key in a file in the repository.
5. Click **Deploy** and wait for the build to finish (the first build installs all dependencies and may take a few minutes).
6. **Check sharing before telling anyone the URL.** In the app's **Share** settings, keep viewing restricted to invited people until the team decides otherwise. Anyone who can use the app spends the team's Gemini quota.

## 3. Smoke test after deployment

1. The sidebar shows **"Gemini API key detected"**, and the SDAIA Academy logo appears in the header.
2. Upload `data/project_aurora_sample.pdf` (download it from the repository first). The overview shows **3 pages, 116 words**.
3. Ask **"Who leads Project Aurora?"**. Expect Dr. Elena Rostova `[Page 1]` with the caption **"Answered by Gemini 2.5 Flash"**.
4. Ask **"What is the capital of Australia?"**. Expect a refusal.
5. Space questions a few seconds apart: the free tier returned `429` after about 5 rapid requests during testing.

If the caption says *"local offline engine (Gemini not configured)"*, the secret is missing or misspelled: check **Settings → Secrets** (key name `GEMINI_API_KEY`) and reboot the app.

## 4. Updating secrets or rotating the key

**App menu → Settings → Secrets**: edit the value and save; the app restarts with the new key. Revoke old keys in Google AI Studio.

## 5. Known deployment caveats (not blockers)

| Caveat | Impact | Mitigation |
| :--- | :--- | :--- |
| Shared tool registry (per server process) | With simultaneous users, the stats tool could briefly see another session's document | The app re-registers the session's document before every answer; a per-session registry is listed as future work |
| Gemini free-tier quota | Heavy use returns `429`; the app shows a clear message | Restrict sharing; space out demo questions |
| Uploaded content goes to Google | In live mode, excerpts are sent to the Gemini API | Use only non-confidential PDFs |
| Model overload (`503`) | Answer comes from the offline engine, labeled as such | Retry the question later |
| App not run on Linux before deployment | Small risk of an unforeseen platform issue | Run the smoke test above right after deploying |
