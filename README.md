# 🤖 AI Document Assistant MVP

A full-stack multi-step reasoning AI assistant that allows guest users to upload documents (PDF, TXT, DOCX), ask questions about them, and get intelligent answers with source references. Built with LangChain, Pinecone, FastAPI, and React.js.

## ✨ Features

- **📄 Document Upload**: Support for PDF, TXT, and DOCX files
- **🔍 Intelligent Q&A**: Ask questions and get contextual answers from your documents
- **📚 Source References**: See exactly which parts of your documents were used to generate answers
- **👤 Guest User Isolation**: Each user has a separate, secure space for their documents
- **⚡ Real-time Processing**: Fast document processing and question answering
- **📱 Responsive Design**: Works seamlessly on desktop and mobile devices
- **🎨 Modern UI**: Clean, intuitive interface with loading indicators and error handling

## 🏗️ Architecture

### Backend (Python + FastAPI)
- **FastAPI**: High-performance web framework
- **LangChain**: Document processing and Q&A chains
- **Pinecone**: Vector database for semantic search
- **OpenAI**: Embeddings and language model
- **Document Processing**: PDF (pdfplumber), DOCX (python-docx), TXT

### Frontend (React.js)
- **React 18**: Modern React with hooks
- **Responsive Design**: Mobile-first CSS
- **Guest Management**: UUID-based user isolation
- **File Upload**: Drag-and-drop interface
- **Real-time UI**: Loading states and error handling

## 📋 Prerequisites

- **Python 3.8+**
- **Node.js 16+**
- **OpenAI API Key**
- **Pinecone Account** (free tier available)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository (if using git)
git clone <repository-url>
cd langchain

# Or if you have the files locally, navigate to the project directory
cd path/to/langchain
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the `backend` directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
# Note: Index name and host are configured in the code
```

**Getting API Keys:**

1. **OpenAI API Key**:
   - Go to [OpenAI Platform](https://platform.openai.com/)
   - Sign up/login and navigate to API Keys
   - Create a new secret key

2. **Pinecone API Key**:
   - Go to [Pinecone](https://www.pinecone.io/)
   - Sign up for free account
   - Create a new index with these specifications:
     - **Index Name**: `ai-assistant`
     - **Dimensions**: `512` (for text-embedding-3-small model)
     - **Metric**: `cosine`
     - **Cloud**: `AWS`
     - **Region**: `us-east-1`
     - **Type**: `Dense`
     - **Capacity mode**: `Serverless`
   - Get your API key from the dashboard

### 4. Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install
```

### 5. Run the Application

**Terminal 1 - Backend:**
```bash
cd backend
# Make sure virtual environment is activated
python main.py
# Or alternatively:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

### 6. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📖 Usage Guide

### Uploading Documents

1. **Open the application** in your browser
2. **Drag and drop** or **click to browse** for files
3. **Supported formats**: PDF, TXT, DOCX (max 10MB)
4. **Click "Upload & Process"** to process the document
5. **Wait for confirmation** - documents are split into chunks and embedded

### Asking Questions

1. **Upload at least one document** first
2. **Type your question** in the question box
3. **Press Enter** or click "Ask Question"
4. **View the answer** with source references
5. **Check sources** to see which document parts were used

### Managing Data

- **Guest ID**: Automatically generated and stored in browser
- **Data Isolation**: Your documents are private to your session
- **Clear Data**: Use "Clear All Documents" to remove all your data
- **New Session**: Clear browser data or use incognito mode

## 🔧 API Endpoints

### Upload Document
```http
POST /upload
Content-Type: multipart/form-data

file: <document_file>
guest_id: <unique_guest_id>
```

### Ask Question
```http
POST /ask
Content-Type: application/json

{
  "question": "Your question here",
  "guest_id": "unique_guest_id"
}
```

### Clear User Data
```http
DELETE /clear/{guest_id}
```

### Health Check
```http
GET /health
```

## 🛠️ Development

### Project Structure

```
langchain/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Environment variables
├── frontend/
│   ├── public/
│   │   ├── index.html       # HTML template
│   │   └── manifest.json    # PWA manifest
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUpload.js     # File upload component
│   │   │   ├── FileUpload.css    # Upload styles
│   │   │   ├── QuestionAnswer.js # Q&A component
│   │   │   └── QuestionAnswer.css# Q&A styles
│   │   ├── utils/
│   │   │   └── guestId.js        # Guest ID management
│   │   ├── App.js               # Main app component
│   │   ├── App.css              # Main app styles
│   │   ├── index.js             # React entry point
│   │   └── index.css            # Global styles
│   └── package.json             # Node.js dependencies
└── README.md                    # This file
```

### Adding New Features

1. **New Document Types**: Extend `process_document()` function
2. **Enhanced Q&A**: Modify LangChain chains in backend
3. **User Authentication**: Replace guest system with proper auth
4. **Document Management**: Add document listing and deletion
5. **Advanced Search**: Implement filters and search options

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|----------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM | Yes | - |
| `PINECONE_API_KEY` | Pinecone API key for vector database | Yes | - |
| `PINECONE_ENV` | Pinecone environment | No | `us-west1-gcp-free` |
| `PINECONE_INDEX_NAME` | Pinecone index name | No | `ai-assistant` |

## 🚨 Troubleshooting

### Common Issues

**Backend won't start:**
- Check if virtual environment is activated
- Verify all environment variables are set
- Ensure Python 3.8+ is installed
- Check if port 8000 is available

**Frontend won't start:**
- Verify Node.js 16+ is installed
- Run `npm install` to install dependencies
- Check if port 3000 is available

**Upload fails:**
- Check file size (max 10MB)
- Verify file format (PDF, TXT, DOCX)
- Check backend logs for errors
- Ensure OpenAI API key is valid

**Questions return no results:**
- Verify documents were uploaded successfully
- Check if Pinecone index was created
- Ensure guest_id is consistent
- Check backend logs for errors

**CORS errors:**
- Ensure backend is running on port 8000
- Check CORS middleware configuration
- Verify frontend is accessing correct backend URL

### Logs and Debugging

**Backend logs:**
```bash
# Run with debug logging
uvicorn main:app --reload --log-level debug
```

**Frontend debugging:**
- Open browser developer tools (F12)
- Check Console tab for JavaScript errors
- Check Network tab for API call failures

## 📈 Performance Optimization

### Backend
- **Batch Processing**: Documents are processed in chunks
- **Async Operations**: FastAPI handles concurrent requests
- **Vector Caching**: Pinecone provides fast similarity search
- **Connection Pooling**: Reuse database connections

### Frontend
- **Code Splitting**: React lazy loading for components
- **Memoization**: Prevent unnecessary re-renders
- **Debounced Input**: Reduce API calls during typing
- **Progressive Loading**: Show content as it loads

## 🔒 Security Considerations

- **Guest Isolation**: Each user's data is completely isolated
- **Input Validation**: File types and sizes are validated
- **API Rate Limiting**: Consider adding rate limits for production
- **HTTPS**: Use HTTPS in production environments
- **Environment Variables**: Never commit API keys to version control

## 🚀 Deployment

### Backend Deployment
- **Docker**: Containerize the FastAPI application
- **Cloud Platforms**: Deploy to AWS, GCP, or Azure
- **Environment**: Set production environment variables
- **Monitoring**: Add logging and monitoring solutions

### Frontend Deployment
- **Build**: Run `npm run build` to create production build
- **Static Hosting**: Deploy to Netlify, Vercel, or S3
- **CDN**: Use CDN for better performance
- **Environment**: Configure production API endpoints

## 📝 License

This project is for educational and demonstration purposes. Please ensure you comply with the terms of service for OpenAI and Pinecone when using their APIs.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check browser console for frontend errors
4. Review backend logs for server errors

---

**Built with ❤️ using LangChain, Pinecone, FastAPI, and React.js**