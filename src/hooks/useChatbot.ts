import { useState, useEffect } from 'react';
import { chatApi } from '@site/src/services/api';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  sources?: string[];
  timestamp: Date;
}

export const useChatbot = (selectedText?: string, shouldShowIntro: boolean = false) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [hasIntroBeenShown, setHasIntroBeenShown] = useState(false);

  // Add initial message when selectedText changes
  useEffect(() => {
    if (selectedText) {
      setMessages(prev => [
        ...prev,
        {
          id: `initial-${Date.now()}`,
          content: `You selected: "${selectedText}"\nHow can I help you with this?`,
          role: 'assistant',
          timestamp: new Date(),
        }
      ]);
    }
  }, [selectedText]);

  // Add intro message when chat session starts (only if no messages exist yet and shouldShowIntro is true)
  useEffect(() => {
    if (messages.length === 0 && shouldShowIntro && !hasIntroBeenShown && !selectedText) {
      const introMessage: Message = {
        id: `intro-${Date.now()}`,
        content: "Hello! I'm your Textbook Assistant. How can I help you with the documentation today?",
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, introMessage]);
      setHasIntroBeenShown(true);
    }
  }, [messages.length, shouldShowIntro, hasIntroBeenShown, selectedText]);

  const sendMessage = async (query: string) => {
    // Add user message to the chat
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content: query,
      role: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Send the message to the backend
      const response = await chatApi.sendMessage({
        query,
        selected_text: selectedText,
        session_id: sessionId || undefined,
      });

      // Update session ID if it's the first message
      if (!sessionId) {
        setSessionId(response.session_id);
      }

      // Add assistant response to the chat
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        content: response.response,
        role: 'assistant',
        sources: response.sources,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message to the chat
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        content: 'Sorry, I encountered an error while processing your request.',
        role: 'assistant',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return {
    messages,
    sendMessage,
    isLoading,
    sessionId,
  };
};