# Contributing to Sanice

First off, thank you for considering contributing to Sanice! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

## Code of Conduct & Respect
**Respect is non-negotiable.** We are a global community. Harassment, insults, or lack of patience with beginners will not be tolerated. We expect constructive feedback and kindness.

---

## 🌐 The Golden Rule: Internationalization (I18N)
**Sanice is a polyglot library.** This is our most important technical rule:

> **Every new feature MUST be accessible in all 4 supported languages:**
> 1.  🇧🇷 Portuguese (PT)
> 2.  🇺🇸 English (EN)
> 3.  🇨🇳 Chinese (ZH)
> 4.  🇮🇳 Hindi (HI)

**How to do it:**
1.  If you create a method called `def super_clean(self):`...
2.  You **must** add it to the `METHOD_ALIASES` dictionary in `core.py`.
3.  You **must** provide the translations/transliterations for the other 3 languages.
    * *Tip: Use Google Translate or ChatGPT if you are not fluent, but do not leave it blank.*
4.  You **must** add any log messages to the `I18N` dictionary.

**Pull Requests that break this rule will be rejected automatically.**

---

## Development Setup

1.  **Fork** the repository on GitHub.
2.  **Clone** your fork locally:
    ```bash
    git clone https://github.com/YOUR_USER/sanice.git
    cd sanice
    ```
3.  **Install dependencies** in editable mode (with dev tools):
    ```bash
    pip install -e ".[dev,api,db]"
    ```
4.  **Create a branch** for your feature:
    ```bash
    git checkout -b feature/my-amazing-feature
    ```

## Testing
**Code without tests is broken by design.**

1.  **New Features:** If you add a new feature (e.g., a new method in `core.py`), you **must** add a corresponding test case in `tests/`.
2.  **Bug Fixes:** If you fix a bug, write a regression test to ensure it doesn't come back.
3.  **Run the full suite:** Before submitting, run the tests to ensure nothing is broken.
    ```bash
    pytest
    ```

## Pull Request Process

1.  **Documentation:** Update the `README.md` with details of changes (if applicable).
2.  **Dependencies:** If your feature introduces a new Python library:
    * Add it to `install_requires` in `setup.py` if it's essential.
    * Add it to `extras_require` in `setup.py` if it's optional (like database drivers).
3.  **Versioning:** Update the version in `sanice/__init__.py` (e.g., `1.0.8` -> `1.0.9`) **only if instructed** (usually maintainers handle releases).
4.  **CI/CD:** The PR will trigger a GitHub Action. **If the tests fail, the PR cannot be merged.**