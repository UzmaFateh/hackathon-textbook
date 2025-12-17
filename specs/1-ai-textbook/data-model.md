# Data Model: AI-Native Software Development Textbook

## Entities

### Chapter
- **Fields**:
  - id: string (unique identifier)
  - title: string (chapter title)
  - content: string (Markdown/MDX content)
  - wordCount: number (minimum 2000)
  - type: enum (theory | practical | mixed)
  - difficulty: enum (beginner | intermediate | advanced)
  - prerequisites: array of string (required knowledge)
  - exercises: array of object (practice problems)
  - citations: array of object (APA-formatted references)

### Tutorial
- **Fields**:
  - id: string (unique identifier)
  - title: string (tutorial title)
  - content: string (Markdown/MDX content)
  - steps: array of object (step-by-step instructions)
  - codeSnippets: array of object (executable code examples)
  - prerequisites: array of string (required setup/tools)
  - expectedOutcome: string (what user should achieve)

### Citation
- **Fields**:
  - id: string (unique identifier)
  - type: enum (journal | documentation | book | website)
  - title: string (source title)
  - authors: array of string (author names)
  - year: number (publication year)
  - publisher: string (journal/publisher name)
  - url: string (optional - source URL)
  - doi: string (optional - for academic sources)
  - apaFormatted: string (properly formatted APA citation)

### CodeSnippet
- **Fields**:
  - id: string (unique identifier)
  - language: string (programming language)
  - code: string (the actual code)
  - description: string (what the code does)
  - executionResult: string (expected output or behavior)
  - validationScript: string (script to test the code)
  - isValidated: boolean (whether it passes validation)

### UserProgress
- **Fields**:
  - userId: string (user identifier)
  - completedChapters: array of string (chapter IDs)
  - completedTutorials: array of string (tutorial IDs)
  - bookmarkedSections: array of string (section IDs)
  - quizResults: array of object (results from chapter exercises)

## Relationships

- Chapter contains multiple Citations (1 to many)
- Tutorial contains multiple CodeSnippets (1 to many)
- Chapter may contain multiple Exercises (1 to many)
- UserProgress tracks progress across Chapters and Tutorials (many to many)

## Validation Rules

- Chapter.wordCount >= 2000 (from spec requirement)
- Chapter.type must be one of [theory, practical, mixed]
- Citation must have proper APA formatting
- CodeSnippet must pass validationScript test
- Tutorial.steps must not be empty

## State Transitions

- Chapter: draft → review → approved → published
- CodeSnippet: untested → validated → verified
- Tutorial: planned → in-progress → complete → validated