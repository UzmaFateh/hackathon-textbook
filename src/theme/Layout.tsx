import React from 'react';
import Layout from '@theme-original/Layout';
import Chatbot from '../components/Chatbot/Chatbot'; // Import the chatbot component
import type { Props } from '@theme/Layout';

// Create a wrapper around the default Layout to add the chatbot
export default function LayoutWrapper(props: Props): JSX.Element {
  return (
    <>
      <Layout {...props}>
        {props.children}
        {/* Add the chatbot component to appear on all pages */}
        <Chatbot />
      </Layout>
    </>
  );
}