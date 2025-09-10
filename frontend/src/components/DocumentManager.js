import React, { useState, useEffect } from 'react';
import useDocumentStore from '../store/documentStore';
import './DocumentManager.css';

const DocumentManager = ({ guestId }) => {
  const [deleteStatus, setDeleteStatus] = useState('');
  const [conversationStats, setConversationStats] = useState(null);
  const [deletingItems, setDeletingItems] = useState(new Set());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const { documentStats, isLoading, setGuestId, fetchDocumentStats, deleteDocument: deleteDoc } = useDocumentStore();

  // Fetch conversation statistics
  const fetchConversationStats = async () => {
    try {
      const response = await fetch(`http://localhost:8000/stats/${guestId}`);
      if (response.ok) {
        const data = await response.json();
        // Use the conversation chunks directly from the stats endpoint
        setConversationStats({ 
          conversation_chunks: data.conversation_chunks || 0,
          total_chunks: data.total_chunks || 0,
          document_chunks: data.document_chunks || 0,
          other_chunks: data.other_chunks || 0
        });
      } else {
        console.error('Failed to fetch conversation stats:', response.status);
        setConversationStats({ conversation_chunks: 0 });
      }
    } catch (error) {
      console.error('Error fetching conversation stats:', error);
      setConversationStats({ conversation_chunks: 0 });
    }
  };

  // Delete all conversations
  const deleteAllConversations = async () => {
    if (!window.confirm('Are you sure you want to delete all conversation history?')) {
      return;
    }

    setDeleteStatus('🗑️ Deleting conversations...');
    try {
      const response = await fetch(`http://localhost:8000/conversations/${guestId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        const result = await response.json();
        setDeleteStatus(`✅ Deleted ${result.conversation_chunks_deleted} conversation chunks! Refreshing...`);
        
        // Force complete refresh
        setTimeout(async () => {
          await Promise.all([fetchDocumentStats(), fetchConversationStats()]);
          setDeleteStatus('✅ Conversations deleted and data refreshed!');
          setTimeout(() => setDeleteStatus(''), 2000);
        }, 500);
      } else {
        const error = await response.json();
        setDeleteStatus(`❌ Error: ${error.detail}`);
        setTimeout(() => setDeleteStatus(''), 5000);
      }
    } catch (error) {
      setDeleteStatus(`❌ Error: ${error.message}`);
      setTimeout(() => setDeleteStatus(''), 5000);
    }
  };

  // Clear all user data
  const clearAllData = async () => {
    if (!window.confirm('Are you sure you want to delete ALL your data (documents and conversations)? This cannot be undone.')) {
      return;
    }

    setDeleteStatus('🗑️ Clearing all data...');
    try {
      const response = await fetch(`http://localhost:8000/clear/${guestId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        const result = await response.json();
        setDeleteStatus(`✅ Cleared all data: ${result.document_chunks_deleted} document chunks, ${result.conversation_chunks_deleted} conversation chunks! Refreshing...`);
        
        // Force complete refresh
        setTimeout(async () => {
          await Promise.all([fetchDocumentStats(), fetchConversationStats()]);
          setDeleteStatus('✅ All data cleared and refreshed!');
          setTimeout(() => setDeleteStatus(''), 3000);
        }, 500);
      } else {
        const error = await response.json();
        setDeleteStatus(`❌ Error: ${error.detail}`);
        setTimeout(() => setDeleteStatus(''), 5000);
      }
    } catch (error) {
      setDeleteStatus(`❌ Error: ${error.message}`);
      setTimeout(() => setDeleteStatus(''), 5000);
    }
  };



  const deleteDocument = async (documentName) => {
    if (!window.confirm(`Are you sure you want to delete "${documentName}"?`)) {
      return;
    }

    // Add to deleting items for animation
    setDeletingItems(prev => new Set([...prev, documentName]));
    setDeleteStatus('🗑️ Deleting document...');
    const result = await deleteDoc(documentName);
    
    if (result.success) {
      setDeleteStatus('✅ Document deleted successfully! Refreshing...');
      // Wait for animation to complete before refreshing
      setTimeout(async () => {
        setDeletingItems(prev => {
          const newSet = new Set(prev);
          newSet.delete(documentName);
          return newSet;
        });
        await fetchDocumentStats();
        setDeleteStatus('✅ Document deleted and data refreshed!');
        setTimeout(() => setDeleteStatus(''), 2000);
      }, 300);
    } else {
      // Remove from deleting items on error
      setDeletingItems(prev => {
        const newSet = new Set(prev);
        newSet.delete(documentName);
        return newSet;
      });
      setDeleteStatus(`❌ Error: ${result.error}`);
      setTimeout(() => setDeleteStatus(''), 5000);
    }
  };

  useEffect(() => {
    if (guestId) {
      setGuestId(guestId);
      setIsRefreshing(true);
      Promise.all([fetchDocumentStats(), fetchConversationStats()]).finally(() => {
        setTimeout(() => setIsRefreshing(false), 200);
      });
    }
  }, [guestId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Refresh conversation stats when document stats change
  useEffect(() => {
    if (documentStats && guestId) {
      fetchConversationStats();
    }
  }, [documentStats]); // eslint-disable-line react-hooks/exhaustive-deps







  if (isLoading) {
    return (
      <div className="document-manager">
        <h3>📊 Document Statistics</h3>
        <div className="loading">Loading document statistics...</div>
      </div>
    );
  }

  if (!documentStats) {
    return (
      <div className="document-manager">
        <h3>📊 Document Statistics</h3>
        <div className="no-data">No document data available</div>
      </div>
    );
  }

  return (
    <div className="document-manager">
      <h3>📊 Document Statistics</h3>
      
      <div className="stats-summary">
        <div className="stat-item">
          <span className="stat-label">Total Documents:</span>
          <span className="stat-value">{documentStats.total_documents}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Total Chunks:</span>
          <span className={`stat-value ${documentStats.total_chunks > 10 ? 'warning' : ''}`}>
            {documentStats.total_chunks}
          </span>
        </div>
      </div>

      {documentStats.warning_message && (
        <div className="warning-message">
          ⚠️ {documentStats.warning_message}
        </div>
      )}

      {deleteStatus && (
        <div className={`delete-status ${deleteStatus.includes('✅') ? 'success' : 'error'}`}>
          {deleteStatus}
        </div>
      )}

      <div className={`documents-list ${isRefreshing ? 'loading' : ''}`}>
        <h4>📄 Your Documents</h4>
        {documentStats.documents.length === 0 ? (
          <div className="no-documents">No documents uploaded yet</div>
        ) : (
          <div className="document-items">
            {documentStats.documents.map((doc, index) => (
              <div 
                key={index} 
                className={`document-item ${deletingItems.has(doc.name) ? 'deleting' : ''}`}
              >
                <div className="document-info">
                  <div className="document-name">{doc.name}</div>
                  <div className="document-details">
                    <span className="chunk-count">{doc.chunks} chunks</span>
                    {doc.upload_date !== 'Unknown' && (
                      <span className="upload-date">
                        Uploaded: {new Date(doc.upload_date).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                </div>
                <button 
                  className="delete-btn"
                  onClick={() => deleteDocument(doc.name)}
                  disabled={deletingItems.has(doc.name)}
                  title="Delete this document"
                >
                  {deletingItems.has(doc.name) ? '⏳ Deleting...' : '🗑️ Delete'}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="conversation-section">
        <h4>💬 Data Management</h4>
        <div className="conversation-stats">
          <div className="stat-item">
            <span className="stat-label">Total Chunks:</span>
            <span className="stat-value">
              {conversationStats ? conversationStats.total_chunks : 'Loading...'}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Document Chunks:</span>
            <span className="stat-value">
              {conversationStats ? conversationStats.document_chunks : 'Loading...'}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Conversation Chunks:</span>
            <span className="stat-value">
              {conversationStats ? conversationStats.conversation_chunks : 'Loading...'}
            </span>
          </div>
          {conversationStats && conversationStats.other_chunks > 0 && (
            <div className="stat-item">
              <span className="stat-label">Other Chunks:</span>
              <span className="stat-value">
                {conversationStats.other_chunks}
              </span>
            </div>
          )}
        </div>
        <div className="conversation-actions">
          <button 
            className="delete-conversations-btn"
            onClick={deleteAllConversations}
            disabled={isLoading || !conversationStats || conversationStats.conversation_chunks === 0}
            title="Delete all conversation history"
          >
            🗑️ Clear Conversations
          </button>
          <button 
            className="clear-all-btn"
            onClick={clearAllData}
            disabled={isLoading || !conversationStats || conversationStats.total_chunks === 0}
            title="Delete all data (documents and conversations)"
          >
            🗑️ Clear All Data
          </button>
        </div>
      </div>

      <div className="refresh-section">
        <button 
          className="refresh-btn"
          onClick={() => {
            setIsRefreshing(true);
            Promise.all([fetchDocumentStats(), fetchConversationStats()]).finally(() => {
              setTimeout(() => setIsRefreshing(false), 200);
            });
          }}
          disabled={isLoading || isRefreshing}
        >
          {isRefreshing ? '⏳ Refreshing...' : '🔄 Refresh Statistics'}
        </button>
      </div>
    </div>
  );
};

export default DocumentManager;