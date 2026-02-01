import { useState, useEffect } from 'react'
import './App.css'
import { API_BASE_URL } from './config'
import Header from './components/Header'
import DocumentUpload from './components/DocumentUpload'
import DocumentViewer from './components/DocumentViewer'
import ChatInterface from './components/ChatInterface'
import ReviewPanel from './components/ReviewPanel'

function App() {
  const [documents, setDocuments] = useState([])
  const [selectedDocument, setSelectedDocument] = useState(null)
  const [activeTab, setActiveTab] = useState('upload')
  const [isLoading, setIsLoading] = useState(false)

  // Fetch documents on mount
  useEffect(() => {
    fetchDocuments()
  }, [])

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/documents/`)
      if (response.ok) {
        const data = await response.json()
        setDocuments(data.documents || [])
      }
    } catch (error) {
      console.error('Failed to fetch documents:', error)
    }
  }

  const handleDocumentUpload = async (file) => {
    setIsLoading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const data = await response.json()
        setDocuments(prev => [{
          document_id: data.document_id,
          filename: data.filename,
          status: data.status,
          created_at: data.created_at
        }, ...prev])
        setActiveTab('documents')
      }
    } catch (error) {
      console.error('Upload failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSelectDocument = async (documentId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/documents/${documentId}/results`)
      if (response.ok) {
        const data = await response.json()
        setSelectedDocument(data)
        setActiveTab('viewer')
      }
    } catch (error) {
      console.error('Failed to fetch document:', error)
    }
  }

  return (
    <div className="app">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        hasDocument={!!selectedDocument}
      />

      <main className="main-content">
        {activeTab === 'upload' && (
          <DocumentUpload
            onUpload={handleDocumentUpload}
            isLoading={isLoading}
            documents={documents}
            onSelectDocument={handleSelectDocument}
            onRefresh={fetchDocuments}
          />
        )}

        {activeTab === 'viewer' && selectedDocument && (
          <DocumentViewer document={selectedDocument} />
        )}

        {activeTab === 'chat' && selectedDocument && (
          <ChatInterface document={selectedDocument} />
        )}

        {activeTab === 'review' && selectedDocument && (
          <ReviewPanel document={selectedDocument} />
        )}

        {(activeTab === 'viewer' || activeTab === 'chat' || activeTab === 'review') && !selectedDocument && (
          <div className="no-document">
            <div className="no-document-icon">📄</div>
            <h2>No Document Selected</h2>
            <p>Upload or select a document to get started</p>
            <button
              className="btn btn-primary"
              onClick={() => setActiveTab('upload')}
            >
              Go to Upload
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
