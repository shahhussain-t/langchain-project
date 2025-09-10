import React, { useState } from 'react';
import './FileUpload.css';

const FileUpload = ({ guestId, onFileUpload }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [dragActive, setDragActive] = useState(false);

  const allowedTypes = ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
  const allowedExtensions = ['.pdf', '.txt', '.docx'];

  const validateFile = (file) => {
    if (!file) return false;
    
    const isValidType = allowedTypes.includes(file.type) || 
                       allowedExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    
    if (!isValidType) {
      setUploadStatus('❌ Please select a PDF, TXT, or DOCX file.');
      return false;
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      setUploadStatus('❌ File size must be less than 10MB.');
      return false;
    }
    
    return true;
  };

  const handleFileSelect = (file) => {
    if (validateFile(file)) {
      setSelectedFile(file);
      setUploadStatus('');
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      handleFileSelect(file);
    }
  };

  const uploadFile = async () => {
    if (!selectedFile || !guestId) {
      setUploadStatus('❌ Please select a file and ensure guest ID is available.');
      return;
    }

    setIsUploading(true);
    setUploadStatus('📤 Uploading and processing document...');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('guest_id', guestId);

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const result = await response.json();
      setUploadStatus(`✅ ${result.message} (${result.chunks_processed} chunks processed)`);
      
      // Notify parent component
      onFileUpload(result);
      
      // Reset form
      setSelectedFile(null);
      document.getElementById('file-input').value = '';
      
    } catch (error) {
      console.error('Upload error:', error);
      let errorMessage = error.message;
      
      // Handle specific chunk limit error
      if (error.message.includes('chunks uploaded')) {
        errorMessage = error.message + ' You can delete some documents using the Document Manager below.';
      }
      
      setUploadStatus(`❌ Upload failed: ${errorMessage}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="file-upload">
      <div 
        className={`upload-area ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className="upload-content">
          <div className="upload-icon">📁</div>
          <p className="upload-text">
            Drag and drop your document here, or{' '}
            <label htmlFor="file-input" className="file-label">
              browse files
            </label>
          </p>
          <p className="upload-hint">
            Supported formats: PDF, TXT, DOCX (max 10MB)
          </p>
          <input
            id="file-input"
            type="file"
            accept=".pdf,.txt,.docx"
            onChange={handleFileChange}
            className="file-input"
          />
        </div>
      </div>

      {selectedFile && (
        <div className="selected-file">
          <div className="file-info">
            <span className="file-icon">📄</span>
            <div className="file-details">
              <div className="file-name">{selectedFile.name}</div>
              <div className="file-size">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </div>
            </div>
          </div>
          <button 
            className="upload-btn"
            onClick={uploadFile}
            disabled={isUploading}
          >
            {isUploading ? (
              <>
                <span className="spinner"></span>
                Processing...
              </>
            ) : (
              'Upload & Process'
            )}
          </button>
        </div>
      )}

      {uploadStatus && (
        <div className={`upload-status ${uploadStatus.startsWith('✅') ? 'success' : uploadStatus.startsWith('❌') ? 'error' : 'info'}`}>
          {uploadStatus}
        </div>
      )}
    </div>
  );
};

export default FileUpload;