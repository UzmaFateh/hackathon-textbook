#!/usr/bin/env node

/**
 * Build and validation workflow script
 * This script orchestrates the build process and runs all validation checks
 */

const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const SCRIPTS_DIR = './scripts';
const BUILD_DIR = './build';

// Import validation functions
const { validateFile, scanDirectory } = require('./validate-code-snippets');

async function runCommand(command, description) {
  console.log(`\n${description}`);
  console.log(`Running: ${command}\n`);

  return new Promise((resolve, reject) => {
    const child = spawn(command, { shell: true, stdio: 'inherit' });

    child.on('close', (code) => {
      if (code === 0) {
        console.log(`✅ ${description} completed successfully`);
        resolve(code);
      } else {
        console.log(`❌ ${description} failed with code ${code}`);
        reject(new Error(`${description} failed`));
      }
    });
  });
}

async function checkLinks() {
  console.log('\n🔍 Checking for broken links...');

  try {
    // This is a simplified link check - in a real implementation, you'd use a tool like lychee
    const files = scanDirectory('./docs');
    let brokenLinks = 0;

    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');
      // Simple regex to find markdown links
      const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
      let match;

      while ((match = linkRegex.exec(content)) !== null) {
        const link = match[2];
        // Check if it's an external link (skip for now)
        if (link.startsWith('http') || link.startsWith('mailto:')) {
          continue;
        }

        // Check if it's a relative link to a file
        if (link.startsWith('./') || link.startsWith('../')) {
          const absolutePath = path.resolve(path.dirname(file), link);
          if (!fs.existsSync(absolutePath)) {
            console.log(`⚠️  Broken relative link in ${file}: ${link}`);
            brokenLinks++;
          }
        }
      }
    }

    if (brokenLinks === 0) {
      console.log('✅ No broken relative links found');
    } else {
      console.log(`❌ Found ${brokenLinks} broken relative link(s)`);
    }

    return brokenLinks === 0;
  } catch (error) {
    console.log(`❌ Link checking failed: ${error.message}`);
    return false;
  }
}

async function checkReadability() {
  console.log('\n📚 Checking readability (Flesch-Kincaid Grade Level)...');

  // This is a simplified readability check
  // In a real implementation, you'd use a library like 'flesch-kincaid' or 'readability'
  try {
    const files = scanDirectory('./docs');
    let readabilityIssues = 0;

    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');

      // Simple word count and sentence count estimation
      const words = content.split(/\s+/).filter(word => word.length > 0);
      const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);

      // Very rough estimation - in a real implementation, use proper readability library
      if (words.length > 0) {
        const avgWordsPerSentence = sentences.length > 0 ? words.length / sentences.length : 0;

        // If average sentence length is too long, it might be hard to read
        if (avgWordsPerSentence > 25) {
          console.log(`⚠️  Long sentences detected in ${file} (avg: ${avgWordsPerSentence.toFixed(1)} words/sentence)`);
          readabilityIssues++;
        }
      }
    }

    if (readabilityIssues === 0) {
      console.log('✅ Readability check passed (no obvious issues detected)');
    } else {
      console.log(`⚠️  ${readabilityIssues} readability issue(s) detected (long sentences)`);
    }

    return true; // Don't fail the build for readability issues
  } catch (error) {
    console.log(`⚠️  Readability checking failed: ${error.message}`);
    return true; // Don't fail the build for readability check failure
  }
}

async function runPlagiarismCheck() {
  console.log('\n🔍 Checking for potential plagiarism...');

  // This is a placeholder for plagiarism detection
  // In a real implementation, you'd integrate with a plagiarism detection service
  console.log('⚠️  Plagiarism checking is not fully implemented in this script');
  console.log('   (In a real implementation, this would check against known sources)');

  return true; // Don't fail the build for now
}

async function main() {
  console.log('🚀 Starting build and validation workflow...\n');

  try {
    // 1. Clean previous build
    if (fs.existsSync(BUILD_DIR)) {
      console.log('🗑️  Cleaning previous build...');
      fs.rmSync(BUILD_DIR, { recursive: true, force: true });
    }

    // 2. Run validation on code snippets
    console.log('\n🔍 Validating code snippets...');
    const files = scanDirectory('./docs');
    let totalIssues = 0;

    for (const file of files) {
      const result = validateFile(file);
      const issues = result.validations.filter(v => !v.valid);

      if (issues.length > 0) {
        console.log(`⚠️  Issues in ${file}:`);
        for (const issue of issues) {
          console.log(`   - ${issue.description}: ${issue.message}`);
          totalIssues++;
        }
      }
    }

    if (totalIssues > 0) {
      console.log(`\n❌ Found ${totalIssues} code snippet issue(s)`);
      process.exit(1);
    } else {
      console.log('✅ All code snippets passed validation');
    }

    // 3. Check for broken links
    const linksOk = await checkLinks();
    if (!linksOk) {
      console.log('\n❌ Broken links detected');
      process.exit(1);
    }

    // 4. Check readability
    await checkReadability();

    // 5. Run plagiarism check
    await runPlagiarismCheck();

    // 6. Build the site
    await runCommand('npm run build', 'Building Docusaurus site');

    console.log('\n🎉 Build and validation workflow completed successfully!');
    console.log('✅ Site built and all validations passed');
  } catch (error) {
    console.error('\n💥 Build and validation workflow failed:', error.message);
    process.exit(1);
  }
}

// Run the workflow if this script is executed directly
if (require.main === module) {
  main();
}