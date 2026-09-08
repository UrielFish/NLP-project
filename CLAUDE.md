# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is brand new and currently empty — no code, no dependencies, no structure yet. It is being set up as an NLP sentiment/text classification project intended for a CV/internship application, so there is no existing architecture or commands to document yet.

## Working with the user

The user has some ML/data science background from a previous project (an ML trading strategy involving technical indicators, walk-forward validation, and backtesting), but is new to NLP and text data specifically. When working in this repo:
- Explain NLP-specific concepts in small, plain-language steps rather than assuming prior background (e.g. tokenization, vectorization, embeddings, class imbalance in text data) — don't assume familiarity just because the user has done ML before.
- Prefer proceeding in small increments and checking in before moving on, rather than making large unreviewed changes.

## Mandatory workflow: small increments only

This is non-negotiable for this project. Never complete a large chunk of the project silently in one go.

- Before writing any code, explain the concept involved and state your plan for the piece you're about to build.
- Implement only one small piece at a time.
- After each piece, pause and stop — do not continue on to the next piece — so the user can ask questions.
- Wait for the user to confirm understanding (or ask follow-ups) before moving on to the next piece.

## Project plan (for reference — confirm with user before starting each stage)

1. **Data & first look**: Load a review/text dataset (e.g. Amazon reviews, IMDB reviews), inspect its structure and label distribution, understand what raw text data looks like before any processing.
2. **Preprocessing**: Clean text (lowercasing, punctuation/stopword handling as appropriate), and explain why text preprocessing choices affect downstream model performance.
3. **Vectorization**: Convert text to numeric features — TF-IDF first (simpler, interpretable), then word embeddings — explaining what each representation captures and loses.
4. **Baseline models**: Train classical models (logistic regression, Naive Bayes) as an interpretable baseline before anything more complex.
5. **Neural baseline / comparison**: Introduce a simple neural approach and compare fairly against the classical baseline.
6. **Evaluation**: Use precision, recall, and F1 (not just accuracy), and explain why — especially given likely class imbalance in review/sentiment data.
7. **Write-up**: Clean repo structure, README with methodology/results/limitations, and CV/interview framing — following the same standard of honest reporting as the trading project (a clear negative or mixed result, reported rigorously, is a fine outcome).

This file should be rewritten/expanded once the project is actually scaffolded (once there's a real structure, dependencies, and commands to run/build/test).
