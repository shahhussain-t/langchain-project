import React, { useState } from 'react';
import './QuestionAnswer.css';

const QuestionAnswer = ({ guestId }) => {
  const [question, setQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [conversations, setConversations] = useState([]);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  const askQuestion = async () => {
    if (!question.trim() || !guestId) {
      return;
    }

    const currentQuestion = question.trim();
    setQuestion('');
    setIsAsking(true);

    // Add question to conversations immediately
    const newConversation = {
      id: Date.now(),
      question: currentQuestion,
      answer: null,
      sources: [],
      isLoading: true,
      timestamp: new Date()
    };

    setConversations(prev => [newConversation, ...prev]);

    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: currentQuestion,
          guest_id: guestId,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get answer');
      }

      const result = await response.json();
      
      // Update the conversation with the answer
      setConversations(prev => 
        prev.map(conv => 
          conv.id === newConversation.id 
            ? { ...conv, answer: result.answer, sources: result.sources, isLoading: false }
            : conv
        )
      );

    } catch (error) {
      console.error('Question error:', error);
      
      // Update conversation with error
      setConversations(prev => 
        prev.map(conv => 
          conv.id === newConversation.id 
            ? { 
                ...conv, 
                answer: `❌ Error: ${error.message}`, 
                sources: [], 
                isLoading: false,
                isError: true 
              }
            : conv
        )
      );
    } finally {
      setIsAsking(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  const clearConversations = () => {
    if (window.confirm('Clear all conversations?')) {
      setConversations([]);
    }
  };

  const handleGenerateReport = async (originalQuestion, aiResponse) => {
    if (!guestId) return;
    
    setIsGeneratingReport(true);
    
    try {
      // Extract topic from the original question or AI response
      const reportTopic = originalQuestion.length > 100 ? 
        originalQuestion.substring(0, 100) + '...' : originalQuestion;
      
      const response = await fetch('http://localhost:8000/generate-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          guest_id: guestId,
          report_topic: reportTopic,
          include_email: false
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate report');
      }
      
      const result = await response.json();
      
      // Add the generated report as a new conversation
      const reportConversation = {
        id: Date.now(),
        question: `📊 Generated Report: ${reportTopic}`,
        answer: result.report,
        sources: [],
        isLoading: false,
        timestamp: new Date(),
        isReport: true
      };
      
      setConversations(prev => [reportConversation, ...prev]);
      
    } catch (error) {
      console.error('Report generation error:', error);
      alert(`Error generating report: ${error.message}`);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  return (
    <div className="question-answer">
      <div className="question-input-section">
        <div className="input-container">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about your uploaded documents..."
            className="question-input"
            rows={3}
            disabled={isAsking}
          />
          <button 
            onClick={askQuestion}
            disabled={!question.trim() || isAsking}
            className="ask-btn"
          >
            {isAsking ? (
              <>
                <span className="spinner"></span>
                Thinking...
              </>
            ) : (
              <>
                <span className="ask-icon">🤔</span>
                Ask Question
              </>
            )}
          </button>
        </div>
        <p className="input-hint">
          Press Enter to ask, or Shift+Enter for new line
        </p>
      </div>

      {conversations.length > 0 && (
        <div className="conversations-section">
          <div className="conversations-header">
            <h3>💬 Conversations ({conversations.length})</h3>
            <button onClick={clearConversations} className="clear-conversations-btn">
              Clear All
            </button>
          </div>
          
          <div className="conversations-list">
            {conversations.map((conv) => (
              <div key={conv.id} className="conversation">
                <div className="question-block">
                  <div className="question-header">
                    <span className="question-icon">❓</span>
                    <span className="question-label">Question:</span>
                    <span className="timestamp">
                      {conv.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="question-text">{conv.question}</div>
                </div>

                <div className="answer-block">
                  <div className="answer-header">
                    <span className="answer-icon">🤖</span>
                    <span className="answer-label">Answer:</span>
                  </div>
                  
                  {conv.isLoading ? (
                    <div className="loading-answer">
                      <span className="spinner"></span>
                      Analyzing documents and generating answer...
                    </div>
                  ) : (
                    <>
                      <div className={`answer-text ${conv.isError ? 'error' : ''}`}>
                        {conv.answer}
                      </div>
                      
                      {/* Report Generation Button */}
                      {!conv.isError && conv.answer && (
                        conv.answer.toLowerCase().includes('generate report') ||
                        conv.answer.toLowerCase().includes('create a report') ||
                        conv.answer.toLowerCase().includes('detailed report') ||
                        conv.answer.toLowerCase().includes('comprehensive analysis')
                      ) && (
                        <div className="report-generation-section">
                          <button 
                            className="generate-report-btn"
                            onClick={() => handleGenerateReport(conv.question, conv.answer)}
                            disabled={isGeneratingReport}
                          >
                            {isGeneratingReport ? '⏳ Generating Report...' : '📊 Generate Detailed Report'}
                          </button>
                        </div>
                      )}
                      
                      {conv.sources && conv.sources.length > 0 && (
                        <div className="sources-section">
                          <h4 className="sources-title">📚 Sources:</h4>
                          <div className="sources-list">
                            {conv.sources.map((source, index) => (
                              <div key={index} className="source-item">
                                <div className="source-header">
                                  <span className="source-doc">📄 {source.document_name}</span>
                                  <span className="source-chunk">Chunk {source.chunk_index + 1}</span>
                                  <span className="source-score">
                                    Relevance: {(source.relevance_score * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="source-snippet">
                                  "{source.snippet}..."
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default QuestionAnswer;