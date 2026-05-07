# Usage Documentation Rules

This section defines rules for usage documentation in this repository.

## Purpose

Documentation in `docs/usage/` describes how an external user consumes an already connected library, NuGet package, module, tool, or integration.

It is not documentation for developing that library internally.

Usage documentation is package-user documentation first. The reader is assumed to be writing their own application after installing the package.

Each important dependency or integration should have its own directory:

```text
docs/usage/<PackageOrIntegrationName>/
```

Example:

```text
docs/usage/NeuroModFlowNet.ONNX/
```

The directory name may be the public package name, not the internal repository or project layout.

## Standard Files

Each usage documentation directory should contain:

```text
Usage.ru.md
Usage.en.md
AI-Usage.md
```

## Usage.ru.md

`Usage.ru.md` is the primary human-oriented document.

The Russian version is the source of truth.

If `Usage.ru.md` and `Usage.en.md` conflict, follow `Usage.ru.md`.

## Usage.en.md

`Usage.en.md` is the English human-oriented version.

It must preserve the structure, section order, meaning, and emphasis of `Usage.ru.md`.

Do not change the Russian document structure while creating or updating the English version.

## AI-Usage.md

`AI-Usage.md` is for coding agents.

It must be written in English.

It should contain:

* canonical usage patterns;
* strict rules;
* forbidden bypasses;
* public package-consumer examples;
* API names that are verified against the current code;
* TODO markers where current project details are unclear.

`AI-Usage.md` should be shorter, more direct, and more imperative than the human-oriented documents.

## Structure Preservation

Existing documentation structure must not be changed without explicit approval.

If the user placed sections or bullet points in a specific order, treat that order as intentional.

Do not:

* reorder sections;
* merge sections;
* rename sections;
* remove user-authored text;
* substantially rewrite user-authored text;
* “improve style” in a way that changes structure, meaning, or emphasis.

Allowed changes:

* fix obvious typos;
* add new points inside suitable existing sections;
* clarify wording without changing meaning;
* add TODO markers where information is missing or unclear.

## User-Authored Text

The user may manually add text to any usage documentation file.

Manual user edits are authoritative.

In general, do not substantially rewrite user-authored text without explicit approval.

If a major rewrite seems necessary, propose the change first instead of applying it directly.

## Grounding Rules

Document only real APIs, types, methods, options, paths, and project-specific usage patterns present in the current repository.

Do not invent APIs.

Do not guess missing behavior.

If something is unclear, add a clearly marked TODO instead of guessing.

## Public Usage Boundary

Human-oriented usage documents must not mention repository internals.

Do not include:

* source file paths;
* internal project names;
* lab, sample, test, or tool project layout;
* implementation details that are only useful for maintaining the repository;
* instructions for changing this repository.

Allowed in human usage documents:

* public type names;
* public method names;
* public enum values;
* package-consumer code examples;
* runtime requirements visible to the package user;
* TODO markers for unclear public behavior.
