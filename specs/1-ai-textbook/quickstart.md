# Quickstart Guide: AI-Native Software Development Textbook

## Prerequisites

- Node.js (v18 or higher)
- npm or yarn package manager
- Git for version control
- Basic knowledge of Markdown and MDX syntax

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ai-textbook
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Start the Development Server
```bash
npm start
```

This will start the Docusaurus development server at `http://localhost:3000`.

## Creating New Content

### Adding a New Chapter
1. Create a new MDX file in `docs/chapters/`
2. Follow the naming convention: `chapter-{number}-{topic}.mdx`
3. Include frontmatter with metadata:
```md
---
title: Chapter Title
description: Brief description of the chapter
tags: [tag1, tag2]
difficulty: intermediate
wordCount: 2500
type: mixed
---
```

### Adding a New Tutorial
1. Create a new MDX file in `docs/tutorials/`
2. Follow the naming convention: `tutorial-{number}-{topic}.mdx`
3. Include frontmatter with metadata:
```md
---
title: Tutorial Title
description: Brief description of the tutorial
tags: [tag1, tag2]
prerequisites: ["List prerequisites here"]
expectedTime: 30 minutes
---
```

### Adding Code Snippets
Use the following syntax for executable code examples:
```mdx
```javascript
// This code will be validated during build
console.log("Hello, AI-Native Development!");
```

## Validation Process

### Running Code Snippet Validation
```bash
npm run validate-code
```

### Checking Readability
```bash
npm run readability-check
```

### Running Plagiarism Detection
```bash
npm run plagiarism-check
```

### Full Validation Suite
```bash
npm run validate-all
```

## Building for Production

```bash
npm run build
```

The built site will be available in the `build/` directory.

## Deployment

The site can be deployed to any static hosting service. For GitHub Pages:

```bash
npm run deploy
```