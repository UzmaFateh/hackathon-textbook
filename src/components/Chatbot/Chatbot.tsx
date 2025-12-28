import React, { useState } from 'react';
import FloatingButton from './FloatingButton';
import ChatWindow from './ChatWindow';
import './Chatbot.css';

const Chatbot: React.FC = () => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [chatKey, setChatKey] = useState(0); // Key to force remount of ChatWindow

  // Function to handle text selection
  React.useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection()?.toString().trim() || '';
      setSelectedText(selectedText);
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
  };

  const closeChat = () => {
    setIsChatOpen(false);
    // Reset the key to force remount ChatWindow and trigger intro message on next open
    setChatKey(prev => prev + 1);
  };

  return (
    <>
      <FloatingButton onClick={toggleChat} />
      <ChatWindow
        key={chatKey} // This forces remount when chat is closed and reopened
        isOpen={isChatOpen}
        onClose={closeChat}
        selectedText={selectedText}
      />
    </>
  );
};

export default Chatbot;