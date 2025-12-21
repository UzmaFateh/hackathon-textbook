import re
from typing import Dict, Any


class MarkdownParser:
    def __init__(self):
        pass

    def parse(self, content: str) -> str:
        """
        Parse markdown content and extract the main text content
        This is a simplified parser that removes markdown formatting
        but preserves the semantic meaning
        """
        # Remove frontmatter if present (between --- delimiters)
        content = self._remove_frontmatter(content)

        # Remove markdown links but keep the link text: [text](url) -> text
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)

        # Remove markdown images: ![alt](url) -> alt
        content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', content)

        # Remove markdown headers but keep the text: ## Header -> Header
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)

        # Remove bold and italic formatting
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # **bold**
        content = re.sub(r'\*(.*?)\*', r'\1', content)      # *italic*
        content = re.sub(r'__(.*?)__', r'\1', content)      # __bold__
        content = re.sub(r'_(.*?)_', r'\1', content)        # _italic_

        # Remove code blocks but preserve content
        content = re.sub(r'```.*?\n(.*?)```', r'\1', content, flags=re.DOTALL)

        # Remove inline code
        content = re.sub(r'`(.*?)`', r'\1', content)

        # Remove blockquotes
        content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)

        # Remove list markers
        content = re.sub(r'^\s*[\*\-\+]\s+', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)

        # Normalize whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)  # Remove extra blank lines
        content = content.strip()

        return content

    def _remove_frontmatter(self, content: str) -> str:
        """
        Remove YAML frontmatter from the beginning of the document
        """
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                return parts[2]  # Return content after the second ---
        return content

    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract metadata from markdown frontmatter
        """
        metadata = {}
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                # Simple YAML-like parsing
                for line in frontmatter.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip().strip('"\'')
        return metadata