#!/usr/bin/env node

/**
 * Script to validate code snippets in the textbook
 * This is a basic validation script that checks for common issues in code snippets
 */

const fs = require('fs');
const path = require('path');

// Configuration
const DOCS_DIR = './docs';
const IGNORE_PATTERNS = ['.git', 'node_modules', 'build'];

// Validation rules
const validationRules = [
  {
    name: 'checkForConsoleLogs',
    description: 'Check for development console logs',
    validate: (content) => {
      const matches = content.match(/console\.(log|debug|info|warn|error)\(/g);
      return {
        valid: !matches,
        message: matches ? `Found ${matches.length} console statement(s)` : null,
        matches: matches || []
      };
    }
  },
  {
    name: 'checkForIncompleteCode',
    description: 'Check for incomplete code markers',
    validate: (content) => {
      const incompleteMarkers = ['TODO', 'FIXME', 'XXX', 'TODO:', 'FIXME:', 'XXX:'];
      let foundMarkers = [];

      for (const marker of incompleteMarkers) {
        const regex = new RegExp(marker, 'gi');
        const matches = content.match(regex);
        if (matches) {
          foundMarkers = foundMarkers.concat(matches);
        }
      }

      return {
        valid: foundMarkers.length === 0,
        message: foundMarkers.length > 0 ? `Found ${foundMarkers.length} incomplete code marker(s): ${foundMarkers.join(', ')}` : null,
        matches: foundMarkers
      };
    }
  },
  {
    name: 'checkForTrailingWhitespace',
    description: 'Check for trailing whitespace',
    validate: (content) => {
      const lines = content.split('\n');
      const trailingWhitespaceLines = [];

      for (let i = 0; i < lines.length; i++) {
        if (/\s+$/.test(lines[i])) {
          trailingWhitespaceLines.push(i + 1);
        }
      }

      return {
        valid: trailingWhitespaceLines.length === 0,
        message: trailingWhitespaceLines.length > 0 ? `Found trailing whitespace on lines: ${trailingWhitespaceLines.join(', ')}` : null,
        matches: trailingWhitespaceLines
      };
    }
  }
];

function isValidFile(file) {
  const ext = path.extname(file).toLowerCase();
  return ext === '.md' || ext === '.mdx';
}

function scanDirectory(dir) {
  const results = [];

  const items = fs.readdirSync(dir);

  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);

    // Skip ignored patterns
    if (IGNORE_PATTERNS.some(pattern => fullPath.includes(pattern))) {
      continue;
    }

    if (stat.isDirectory()) {
      results.push(...scanDirectory(fullPath));
    } else if (isValidFile(item)) {
      results.push(fullPath);
    }
  }

  return results;
}

function validateFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const results = {
    filePath,
    validations: []
  };

  for (const rule of validationRules) {
    const validation = rule.validate(content);
    results.validations.push({
      name: rule.name,
      description: rule.description,
      valid: validation.valid,
      message: validation.message,
      matches: validation.matches
    });
  }

  return results;
}

function runValidation() {
  console.log('Starting code snippet validation...\n');

  const files = scanDirectory(DOCS_DIR);
  console.log(`Found ${files.length} files to validate\n`);

  let totalIssues = 0;
  const allResults = [];

  for (const file of files) {
    const result = validateFile(file);
    allResults.push(result);

    const issues = result.validations.filter(v => !v.valid);
    if (issues.length > 0) {
      console.log(`\n⚠️  Issues found in: ${file}`);
      for (const issue of issues) {
        console.log(`   - ${issue.description}: ${issue.message}`);
        totalIssues++;
      }
    }
  }

  console.log(`\nValidation complete!`);
  console.log(`Total issues found: ${totalIssues}`);

  if (totalIssues === 0) {
    console.log('✅ All code snippets passed validation!');
    process.exit(0);
  } else {
    console.log('❌ Some code snippets failed validation.');
    process.exit(1);
  }
}

// Run validation if this script is executed directly
if (require.main === module) {
  runValidation();
}

module.exports = {
  validateFile,
  scanDirectory,
  validationRules
};