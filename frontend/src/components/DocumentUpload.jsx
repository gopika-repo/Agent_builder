import { useState, useRef } from 'react'
import './DocumentUpload.css'

function DocumentUpload({ onUpload, isLoading, documents, onSelectDocument, onRefresh }) {
    const [isDragging, setIsDragging] = useState(false)
    const fileInputRef = useRef(null)

    const handleDragOver = (e) => {
        e.preventDefault()
        setIsDragging(true)
    }

    const handleDragLeave = () => {
        setIsDragging(false)
    }

    const handleDrop = (e) => {
        e.preventDefault()
        setIsDragging(false)

        const file = e.dataTransfer.files[0]
        if (file) {
            onUpload(file)
        }
    }

    const handleFileSelect = (e) => {
        const file = e.target.files[0]
        if (file) {
            onUpload(file)
        }
    }

    const getStatusBadge = (status) => {
        const statusMap = {
            'completed': { class: 'badge-success', label: 'Completed' },
            'pending': { class: 'badge-info', label: 'Pending' },
            'processing': { class: 'badge-warning', label: 'Processing' },
            'failed': { class: 'badge-error', label: 'Failed' }
        }
        return statusMap[status] || statusMap['pending']
    }

    return (
        <div className="upload-page">
            {/* Upload Section */}
            <section className="upload-section animate-slide-up">
                <div
                    className={`upload-zone ${isDragging ? 'dragging' : ''} ${isLoading ? 'loading' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => !isLoading && fileInputRef.current?.click()}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".pdf,.png,.jpg,.jpeg,.tiff"
                        onChange={handleFileSelect}
                        hidden
                    />

                    {isLoading ? (
                        <div className="upload-loading">
                            <div className="spinner"></div>
                            <span>Processing document...</span>
                        </div>
                    ) : (
                        <>
                            <div className="upload-icon">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <path d="M12 16V4m0 0L8 8m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
                                    <path d="M20 17v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                            </div>
                            <h3>Upload Document</h3>
                            <p>Drag & drop or click to browse</p>
                            <div className="upload-formats">
                                <span className="format-badge">PDF</span>
                                <span className="format-badge">PNG</span>
                                <span className="format-badge">JPG</span>
                                <span className="format-badge">TIFF</span>
                            </div>
                        </>
                    )}
                </div>
            </section>

            {/* Features Section */}
            <section className="features-section animate-slide-up" style={{ animationDelay: '100ms' }}>
                <h2>Powered by 6-Agent Intelligence</h2>
                <div className="features-grid">
                    <div className="feature-card">
                        <div className="feature-icon">👁️</div>
                        <h4>Vision Analysis</h4>
                        <p>YOLO-powered layout detection for tables, figures, and charts</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">📝</div>
                        <h4>Hybrid OCR</h4>
                        <p>Tesseract + EasyOCR for maximum text extraction accuracy</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">🧩</div>
                        <h4>Layout Graph</h4>
                        <p>Spatial relationship analysis and reading order detection</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">🧠</div>
                        <h4>AI Reasoning</h4>
                        <p>LLM-powered summarization and entity extraction</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">🔗</div>
                        <h4>Multi-Modal Fusion</h4>
                        <p>Cross-modal validation and structured output generation</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">✅</div>
                        <h4>Confidence Scoring</h4>
                        <p>Field-level confidence with human review flagging</p>
                    </div>
                </div>
            </section>

            {/* Recent Documents */}
            <section className="documents-section animate-slide-up" style={{ animationDelay: '200ms' }}>
                <div className="section-header">
                    <h2>Recent Documents</h2>
                    <button className="btn btn-ghost" onClick={onRefresh}>
                        🔄 Refresh
                    </button>
                </div>

                {documents.length === 0 ? (
                    <div className="no-documents">
                        <p>No documents yet. Upload your first document to get started!</p>
                    </div>
                ) : (
                    <div className="documents-list">
                        {documents.map((doc) => {
                            const badge = getStatusBadge(doc.status)
                            return (
                                <div
                                    key={doc.document_id}
                                    className="document-row"
                                    onClick={() => doc.status === 'completed' && onSelectDocument(doc.document_id)}
                                >
                                    <div className="document-info">
                                        <span className="document-icon">📄</span>
                                        <div className="document-details">
                                            <span className="document-name">{doc.filename}</span>
                                            <span className="document-date">
                                                {new Date(doc.created_at).toLocaleDateString()}
                                            </span>
                                        </div>
                                    </div>
                                    <span className={`badge ${badge.class}`}>{badge.label}</span>
                                </div>
                            )
                        })}
                    </div>
                )}
            </section>
        </div>
    )
}

export default DocumentUpload
