import React, { useState, useEffect } from 'react';
import './MultiDocumentCompare.css';

/**
 * Multi-Document Comparison Component
 * 
 * Enables querying and comparing across multiple documents
 */
const MultiDocumentCompare = ({ documents = [], onCompare }) => {
    const [selectedDocs, setSelectedDocs] = useState([]);
    const [query, setQuery] = useState('');
    const [compareMode, setCompareMode] = useState(false);
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const toggleDocument = (docId) => {
        setSelectedDocs(prev =>
            prev.includes(docId)
                ? prev.filter(id => id !== docId)
                : [...prev, docId]
        );
    };

    const handleQuery = async () => {
        if (selectedDocs.length === 0) {
            setError('Please select at least one document');
            return;
        }

        if (!query.trim()) {
            setError('Please enter a query');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/advanced/multi-document/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    document_ids: selectedDocs,
                    compare: compareMode,
                    mode: 'standard'
                })
            });

            if (!response.ok) throw new Error('Query failed');

            const data = await response.json();
            setResults(data);

            if (onCompare) {
                onCompare(data);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="multi-doc-compare">
            <div className="compare-header">
                <h3>🔍 Multi-Document Intelligence</h3>
                <p>Query and compare across multiple documents</p>
            </div>

            {/* Document Selection */}
            <div className="doc-selection">
                <h4>Select Documents</h4>
                <div className="doc-list">
                    {documents.map(doc => (
                        <div
                            key={doc.id}
                            className={`doc-item ${selectedDocs.includes(doc.id) ? 'selected' : ''}`}
                            onClick={() => toggleDocument(doc.id)}
                        >
                            <div className="doc-checkbox">
                                {selectedDocs.includes(doc.id) ? '✓' : ''}
                            </div>
                            <div className="doc-info">
                                <span className="doc-name">{doc.filename}</span>
                                <span className="doc-pages">{doc.page_count || '?'} pages</span>
                            </div>
                        </div>
                    ))}
                </div>
                {selectedDocs.length > 0 && (
                    <div className="selection-summary">
                        {selectedDocs.length} document{selectedDocs.length > 1 ? 's' : ''} selected
                    </div>
                )}
            </div>

            {/* Query Input */}
            <div className="query-section">
                <div className="query-input-wrapper">
                    <input
                        type="text"
                        placeholder="Ask a question across documents..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
                    />
                    <button
                        className="query-btn"
                        onClick={handleQuery}
                        disabled={loading}
                    >
                        {loading ? '...' : '🔎'}
                    </button>
                </div>

                <label className="compare-toggle">
                    <input
                        type="checkbox"
                        checked={compareMode}
                        onChange={(e) => setCompareMode(e.target.checked)}
                    />
                    <span>Enable cross-document comparison</span>
                </label>
            </div>

            {/* Error Message */}
            {error && (
                <div className="error-message">
                    ⚠️ {error}
                </div>
            )}

            {/* Results */}
            {results && (
                <div className="compare-results">
                    <h4>Results</h4>

                    {compareMode && results.result?.comparison ? (
                        <div className="comparison-view">
                            {Object.entries(results.result.comparison).map(([docId, findings]) => (
                                <div key={docId} className="doc-findings">
                                    <h5>📄 {docId}</h5>
                                    <div className="findings-list">
                                        {findings.map((finding, idx) => (
                                            <div key={idx} className="finding-item">
                                                <span className="finding-type">{finding.content_type}</span>
                                                <p className="finding-content">{finding.content}</p>
                                                <div className="finding-meta">
                                                    Page {finding.page_number + 1} •
                                                    Confidence: {(finding.score * 100).toFixed(0)}%
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="merged-results">
                            {results.result?.reranked_results?.map((result, idx) => (
                                <div key={idx} className="result-item">
                                    <div className="result-header">
                                        <span className={`result-type type-${result.content_type}`}>
                                            {result.content_type}
                                        </span>
                                        <span className="result-doc">{result.document_id}</span>
                                        <span className="result-score">
                                            {(result.score * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                    <p className="result-content">{result.content}</p>
                                    {result.bbox && (
                                        <div className="result-location">
                                            Page {result.page_number + 1} •
                                            Region: ({Math.round(result.bbox[0])}, {Math.round(result.bbox[1])})
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}

                    {results.result?.reasoning && (
                        <div className="reasoning-box">
                            <h5>🧠 AI Reasoning</h5>
                            <p>{results.result.reasoning}</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default MultiDocumentCompare;
