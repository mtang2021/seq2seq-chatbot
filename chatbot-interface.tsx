import React, { useState, useRef, useEffect } from 'react';

const ChatbotInterface = () => {
  const [messages, setMessages] = useState([
    { text: "Welcome to the Neural Chatbot! I'm trained on movie dialog data. Try saying something to start a conversation.", sender: 'bot' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [decodeMethod, setDecodeMethod] = useState('greedy');
  const [activeTab, setActiveTab] = useState('chat');
  const [temperature, setTemperature] = useState(0.9);
  const [topP, setTopP] = useState(0.9);
  const [beamSize, setBeamSize] = useState(5);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const simulateBotResponse = async (userMessage) => {
    setLoading(true);
    
    const response = await new Promise(resolve => {
      setTimeout(() => {
        // Simulate different responses based on the decode method
        const responses = {
          'greedy': "This is a simulated greedy response! In a real implementation, this would call your backend API with the trained model.",
          'top-p': `This is a simulated top-p sampling response with temperature ${temperature} and p=${topP}! In a real implementation, this would use your neural model.`,
          'beam': `This is a simulated beam search response with beam size ${beamSize}! In a real implementation, this would generate multiple candidates and select the best one.`
        };
        resolve(responses[decodeMethod]);
      }, 1000);
    });
    
    setMessages(prevMessages => [...prevMessages, { text: response, sender: 'bot' }]);
    setLoading(false);
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (input.trim() === '') return;
    
    const userMessage = input;
    setInput('');
    setMessages(prevMessages => [...prevMessages, { text: userMessage, sender: 'user' }]);
    
    await simulateBotResponse(userMessage);
  };

  return (
    <div className="flex flex-col w-full h-screen bg-gray-100">
      {/* Tabs */}
      <div className="bg-white shadow">
        <div className="flex">
          <button 
            className={`px-4 py-2 font-medium ${activeTab === 'chat' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-600'}`}
            onClick={() => setActiveTab('chat')}
          >
            Chat
          </button>
          <button 
            className={`px-4 py-2 font-medium ${activeTab === 'docs' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-600'}`}
            onClick={() => setActiveTab('docs')}
          >
            Documentation
          </button>
        </div>
      </div>

      {activeTab === 'chat' ? (
        <>
          {/* Chat container */}
          <div className="flex-1 overflow-auto p-4">
            <div className="max-w-4xl mx-auto">
              {messages.map((message, index) => (
                <div 
                  key={index} 
                  className={`mb-4 ${message.sender === 'user' ? 'text-right' : 'text-left'}`}
                >
                  <div 
                    className={`inline-block p-3 rounded-lg ${
                      message.sender === 'user' 
                        ? 'bg-blue-500 text-white' 
                        : 'bg-white text-gray-800 shadow'
                    }`}
                  >
                    {message.text}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="text-left mb-4">
                  <div className="inline-block p-3 rounded-lg bg-white text-gray-800 shadow">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Settings panel */}
          <div className="bg-white p-4 border-t border-gray-200">
            <div className="max-w-4xl mx-auto mb-4">
              <div className="flex flex-wrap items-center space-x-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Decoding Method</label>
                  <select 
                    className="block w-full py-2 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                    value={decodeMethod}
                    onChange={(e) => setDecodeMethod(e.target.value)}
                  >
                    <option value="greedy">Greedy Search</option>
                    <option value="top-p">Top-p Sampling</option>
                    <option value="beam">Beam Search</option>
                  </select>
                </div>

                {decodeMethod === 'top-p' && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Temperature: {temperature}</label>
                      <input
                        type="range"
                        min="0.1"
                        max="2"
                        step="0.1"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        className="w-32"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Top-p: {topP}</label>
                      <input
                        type="range"
                        min="0.1"
                        max="1"
                        step="0.1"
                        value={topP}
                        onChange={(e) => setTopP(parseFloat(e.target.value))}
                        className="w-32"
                      />
                    </div>
                  </>
                )}

                {decodeMethod === 'beam' && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Beam Size: {beamSize}</label>
                    <input
                      type="range"
                      min="2"
                      max="10"
                      step="1"
                      value={beamSize}
                      onChange={(e) => setBeamSize(parseInt(e.target.value))}
                      className="w-32"
                    />
                  </div>
                )}
              </div>
            </div>

            {/* Input area */}
            <div className="max-w-4xl mx-auto">
              <form onSubmit={handleSend} className="flex items-center">
                <input
                  type="text"
                  value={input}
                  onChange={handleInputChange}
                  placeholder="Type a message..."
                  className="flex-1 p-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={loading}
                />
                <button
                  type="submit"
                  className="bg-blue-500 text-white p-2 rounded-r-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-blue-300"
                  disabled={loading || input.trim() === ''}
                >
                  Send
                </button>
              </form>
            </div>
          </div>
        </>
      ) : (
        /* Documentation content */
        <div className="flex-1 overflow-auto p-4">
          <div className="max-w-4xl mx-auto bg-white shadow rounded-lg p-6">
            <h1 className="text-2xl font-bold mb-4">Neural Chatbot Documentation</h1>
            
            <section className="mb-6">
              <h2 className="text-xl font-semibold mb-2">Overview</h2>
              <p className="mb-2">
                This chatbot is built using neural sequence-to-sequence (Seq2Seq) models trained on movie dialogue data. 
                It can generate responses to your messages using different decoding strategies.
              </p>
            </section>

            <section className="mb-6">
              <h2 className="text-xl font-semibold mb-2">Dataset</h2>
              <p className="mb-2">
                The chatbot is trained on the <strong>Cornell Movie Dialog Corpus</strong>, which contains conversations 
                extracted from movie scripts. The dataset includes over 220,000 conversational exchanges between more 
                than 10,000 pairs of movie characters.
              </p>
              <p>
                For this implementation, we're using a filtered subset of single-turn conversations 
                (one message and one response) with at least 10 tokens each to ensure meaningful exchanges.
              </p>
            </section>

            <section className="mb-6">
              <h2 className="text-xl font-semibold mb-2">Model Architecture</h2>
              <p className="mb-2">
                The system uses two main neural network architectures:
              </p>
              <ul className="list-disc pl-6 mb-2">
                <li><strong>Baseline Seq2Seq Model:</strong> A basic encoder-decoder architecture that encodes the input message and generates a response.</li>
                <li><strong>Seq2Seq with Attention:</strong> An enhanced model that uses an attention mechanism to focus on relevant parts of the input when generating each word of the response.</li>
              </ul>
              <p>
                Both models use GRU (Gated Recurrent Unit) neural networks and shared word embeddings between the encoder and decoder.
              </p>
            </section>

            <section className="mb-6">
              <h2 className="text-xl font-semibold mb-2">Decoding Methods</h2>
              <p className="mb-2">
                The interface offers three different ways to generate responses:
              </p>
              <ul className="list-disc pl-6">
                <li>
                  <strong>Greedy Search:</strong> At each step, the model selects the most probable next word. 
                  This produces consistent but sometimes repetitive responses.
                </li>
                <li>
                  <strong>Top-p Sampling (Nucleus Sampling):</strong> The model randomly samples from the most probable 
                  words that together reach a cumulative probability of p. This creates more diverse and natural-sounding responses.
                  <ul className="list-circle pl-6 mt-1">
                    <li><strong>Temperature</strong> controls randomness - higher values (>1) increase diversity, lower values (&lt;1) make responses more focused.</li>
                    <li><strong>Top-p</strong> value controls how much of the probability distribution to consider - lower values focus on only the most likely words.</li>
                  </ul>
                </li>
                <li>
                  <strong>Beam Search:</strong> The model keeps track of the k most probable sequences at each step. 
                  This creates more globally coherent responses by considering multiple possibilities in parallel.
                  <ul className="list-circle pl-6 mt-1">
                    <li><strong>Beam Size</strong> controls how many candidate sequences are maintained - larger values explore more possibilities.</li>
                  </ul>
                </li>
              </ul>
            </section>

            <section>
              <h2 className="text-xl font-semibold mb-2">Key Technical Terms</h2>
              <ul className="space-y-2">
                <li>
                  <strong>Seq2Seq (Sequence-to-Sequence):</strong> A neural network architecture that converts one sequence (like a question) 
                  into another sequence (like an answer).
                </li>
                <li>
                  <strong>Attention Mechanism:</strong> Allows the model to focus on different parts of the input when generating each 
                  word of the output, similar to how humans pay attention to specific words in a conversation.
                </li>
                <li>
                  <strong>GRU (Gated Recurrent Unit):</strong> A type of neural network designed to process sequences and remember 
                  information over time.
                </li>
                <li>
                  <strong>Embedding:</strong> A technique that converts words into dense vectors of numbers that capture semantic meaning.
                </li>
                <li>
                  <strong>Bidirectional Encoder:</strong> Processes the input sequence in both forward and backward directions to capture more context.
                </li>
              </ul>
            </section>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatbotInterface;
