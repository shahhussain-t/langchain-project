import React, { useState, useEffect } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import QuestionAnswer from './components/QuestionAnswer';
import UserProfile from './components/UserProfile';
import DocumentManager from './components/DocumentManager';
import useDocumentStore from './store/documentStore';
import { getOrGenerateGuestId } from './utils/guestId';
import { Bot, FileText, MessageCircle, Trash2, User } from 'lucide-react';

function App() {
  const [guestId, setGuestId] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [documentChangeKey, setDocumentChangeKey] = useState(0);
  const { documentStats } = useDocumentStore();
  
  // Compute hasDocuments from Zustand store
  const hasDocuments = documentStats && documentStats.documents && documentStats.documents.length > 0;

  useEffect(() => {
    // Initialize or retrieve guest ID
    const currentGuestId = getOrGenerateGuestId();
    setGuestId(currentGuestId);
  }, []);

  const handleFileUpload = (fileInfo) => {
    setUploadedFiles(prev => [...prev, fileInfo]);
    // Trigger document stats refresh
    setDocumentChangeKey(prev => prev + 1);
  };



  const clearAllData = async () => {
    if (window.confirm('Are you sure you want to clear all your uploaded documents? This action cannot be undone.')) {
      try {
        setIsLoading(true);
        const response = await fetch(`http://localhost:8000/clear/${guestId}`, {
          method: 'DELETE',
        });
        
        if (response.ok) {
          setUploadedFiles([]);
          setDocumentChangeKey(prev => prev + 1);
          alert('All documents cleared successfully!');
        } else {
          throw new Error('Failed to clear documents');
        }
      } catch (error) {
        console.error('Error clearing documents:', error);
        alert('Error clearing documents. Please try again.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Bot className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">AI Document Assistant</h1>
                <p className="text-sm text-gray-600">Upload documents and ask questions to get intelligent answers</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <User className="h-4 w-4" />
                <span>Guest: {guestId.slice(0, 8)}...</span>
              </div>
              {uploadedFiles.length > 0 && (
                <button 
                  className="inline-flex items-center px-3 py-2 border border-red-300 text-sm leading-4 font-medium rounded-md text-red-700 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  onClick={clearAllData}
                  disabled={isLoading}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  {isLoading ? 'Clearing...' : 'Clear All'}
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* User Profile Section */}
        <div className="mb-8">
          <UserProfile guestId={guestId} />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
          {/* Left Column - Upload and Documents */}
          <div className="space-y-5">
            {/* Upload Section */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 bg-green-100 rounded-lg">
                  <FileText className="h-5 w-5 text-green-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-900">Upload Documents</h2>
              </div>
              
              <FileUpload 
                guestId={guestId} 
                onFileUpload={handleFileUpload}
              />
              
              {uploadedFiles.length > 0 && (
                <div className="mt-6 pt-6 border-t border-gray-200">
                  <h3 className="text-sm font-medium text-gray-900 mb-3">
                    Recently Uploaded ({uploadedFiles.length})
                  </h3>
                  <div className="space-y-2">
                    {uploadedFiles.map((file, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <FileText className="h-4 w-4 text-gray-400" />
                          <span className="text-sm font-medium text-gray-900">{file.document_name}</span>
                        </div>
                        <span className="text-xs text-gray-500">{file.chunks_processed} chunks</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Document Manager */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200">
              <DocumentManager 
                guestId={guestId} 
                key={documentChangeKey}
              />
            </div>
          </div>

          {/* Right Column - Q&A */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center space-x-3 mb-6">
              <div className="p-2 bg-purple-100 rounded-lg">
                <MessageCircle className="h-5 w-5 text-purple-600" />
              </div>
              <h2 className="text-xl font-semibold text-gray-900">Ask Questions</h2>
            </div>
            
            {hasDocuments ? (
              <QuestionAnswer guestId={guestId} />
            ) : (
              <div className="text-center py-8">
                <div className="p-3 bg-gray-100 rounded-full w-12 h-12 mx-auto mb-3 flex items-center justify-center">
                  <FileText className="h-6 w-6 text-gray-400" />
                </div>
                <p className="text-gray-500 text-sm">
                  Please upload at least one document before asking questions.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            Powered by LangChain, Pinecone, and OpenAI
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;