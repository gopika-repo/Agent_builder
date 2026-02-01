import { useState } from 'react'
import './DocumentViewer.css'

function DocumentViewer({ document }) {
    const [activeSection, setActiveSection] = useState('summary')
    const [showConfidenceOverlay, setShowConfidenceOverlay] = useState(false)

    const sections = [
        { id: 'summary', label: 'Summary', icon: '📊' },
        { id: 'text', label: 'Full Text', icon: '📝' },
        { id: 'tables', label: 'Tables', icon: '📋' },
        { id: 'entities', label: 'Entities', icon: '🏷️' },
        { id: 'structure', label: 'Structure', icon: '🧩' }
    ]

    const textAnalysis = document.text_analysis || {}
    const fusedOutput = document.fused_output || {}
    const confidenceScores = document.confidence_scores || {}

    const getConfidenceColor = (confidence) => {
        if (confidence >= 0.8) return 'confidence-high'
        if (confidence >= 0.6) return 'confidence-medium'
        return 'confidence-low'
    }

    return (
        <div className="viewer-page">
            {/* Sidebar */}
            <aside className="viewer-sidebar">
                <div className="sidebar-header">
                    <h3>📄 {document.filename}</h3>
                    <span className="page-count">{document.page_count} pages</span>
                </div>

                <nav className="sidebar-nav">
                    {sections.map(section => (
                        <button
                            key={section.id}
                            className={`nav-item ${activeSection === section.id ? 'active' : ''}`}
                            onClick={() => setActiveSection(section.id)}
                        >
                            <span className="nav-icon">{section.icon}</span>
                            <span>{section.label}</span>
                        </button>
                    ))}
                </nav>

                {/* Confidence Overview */}
                <div className="confidence-panel">
                    <h4>Confidence Overview</h4>
                    <div className="confidence-score-large">
                        <span className={`score ${getConfidenceColor(confidenceScores.overall_confidence || 0)}`}>
                            {Math.round((confidenceScores.overall_confidence || 0) * 100)}%
                        </span>
                        <span className="score-label">Overall Confidence</span>
                    </div>

                    <div className="confidence-breakdown">
                        <div className="confidence-item">
                            <span>Vision</span>
                            <div className="progress">
                                <div className="progress-bar" style={{ width: '85%' }}></div>
                            </div>
                        </div>
                        <div className="confidence-item">
                            <span>OCR</span>
                            <div className="progress">
                                <div className="progress-bar" style={{ width: '92%' }}></div>
                            </div>
                        </div>
                        <div className="confidence-item">
                            <span>Layout</span>
                            <div className="progress">
                                <div className="progress-bar" style={{ width: '78%' }}></div>
                            </div>
                        </div>
                    </div>

                    {confidenceScores.items_needing_review?.length > 0 && (
                        <div className="review-alert">
                            <span className="alert-icon">⚠️</span>
                            <span>{confidenceScores.items_needing_review.length} items need review</span>
                        </div>
                    )}
                </div>
            </aside>

            {/* Main Content */}
            <main className="viewer-main">
                {activeSection === 'summary' && (
                    <div className="section-content animate-fade-in">
                        <div className="section-header-row">
                            <h2>Document Summary</h2>
                            <span className={`badge badge-${textAnalysis.document_type === 'financial_report' ? 'info' : 'success'}`}>
                                {textAnalysis.document_type || 'Document'}
                            </span>
                        </div>

                        <div className="summary-card">
                            <h4>Overview</h4>
                            <p>{textAnalysis.summary || 'Processing summary...'}</p>
                        </div>

                        <div className="key-points-card">
                            <h4>🎯 Key Points</h4>
                            <ul className="key-points-list">
                                {(textAnalysis.key_points || []).map((point, i) => (
                                    <li key={i}>
                                        <span className="point-bullet">•</span>
                                        {point}
                                    </li>
                                ))}
                            </ul>
                        </div>

                        <div className="topics-card">
                            <h4>📌 Topics</h4>
                            <div className="topics-list">
                                {(textAnalysis.topics || []).map((topic, i) => (
                                    <span key={i} className="topic-tag">{topic}</span>
                                ))}
                            </div>
                        </div>

                        <div className="stats-grid">
                            <div className="stat-card">
                                <span className="stat-value">{document.page_count || 0}</span>
                                <span className="stat-label">Pages</span>
                            </div>
                            <div className="stat-card">
                                <span className="stat-value">{fusedOutput.tables?.length || 0}</span>
                                <span className="stat-label">Tables</span>
                            </div>
                            <div className="stat-card">
                                <span className="stat-value">
                                    {fusedOutput.elements?.filter(e => ['figure', 'image', 'chart'].includes(e.type)).length || 0}
                                </span>
                                <span className="stat-label">Figures</span>
                            </div>
                            <div className="stat-card">
                                <span className="stat-value">{textAnalysis.entities?.length || 0}</span>
                                <span className="stat-label">Entities</span>
                            </div>
                        </div>
                    </div>
                )}

                {activeSection === 'text' && (
                    <div className="section-content animate-fade-in">
                        <h2>Full Text</h2>
                        <div className="text-content">
                            <pre>{document.ocr_results?.full_text || fusedOutput.full_text || 'No text extracted'}</pre>
                        </div>
                    </div>
                )}

                {activeSection === 'tables' && (
                    <div className="section-content animate-fade-in">
                        <h2>Extracted Tables</h2>
                        {(fusedOutput.tables || []).length === 0 ? (
                            <div className="empty-state">
                                <span className="empty-icon">📋</span>
                                <p>No tables detected in this document</p>
                            </div>
                        ) : (
                            <div className="tables-list">
                                {(fusedOutput.tables || []).map((table, i) => (
                                    <div key={i} className="table-card">
                                        <div className="table-header">
                                            <h4>Table {i + 1}</h4>
                                            <span className="badge badge-info">Page {table.page_number + 1}</span>
                                        </div>
                                        <table className="data-table">
                                            <thead>
                                                <tr>
                                                    {(table.headers || []).map((h, j) => (
                                                        <th key={j}>{h}</th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {(table.rows || []).slice(0, 5).map((row, j) => (
                                                    <tr key={j}>
                                                        {row.map((cell, k) => (
                                                            <td key={k}>{cell}</td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                        {table.rows?.length > 5 && (
                                            <p className="more-rows">+ {table.rows.length - 5} more rows</p>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {activeSection === 'entities' && (
                    <div className="section-content animate-fade-in">
                        <h2>Named Entities</h2>
                        <div className="entities-grid">
                            {(textAnalysis.entities || []).map((entity, i) => (
                                <div key={i} className={`entity-card entity-${entity.type}`}>
                                    <span className="entity-type">{entity.type}</span>
                                    <span className="entity-text">{entity.text}</span>
                                    <div className="confidence-indicator">
                                        <span className={`confidence-dot ${getConfidenceColor(entity.confidence)}`}></span>
                                        <span>{Math.round(entity.confidence * 100)}%</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {activeSection === 'structure' && (
                    <div className="section-content animate-fade-in">
                        <h2>Document Structure</h2>
                        <div className="structure-tree">
                            {(fusedOutput.structure?.sections || []).map((section, i) => (
                                <div key={i} className={`structure-node level-${section.level}`}>
                                    <span className="node-icon">{section.level === 1 ? '📁' : '📄'}</span>
                                    <span className="node-title">{section.title}</span>
                                    <span className="node-page">Page {section.page + 1}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </main>
        </div>
    )
}

export default DocumentViewer
