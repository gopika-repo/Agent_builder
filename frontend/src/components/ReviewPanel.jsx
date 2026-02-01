import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../config';
import './ReviewPanel.css'

function ReviewPanel({ document }) {
    const [flaggedItems, setFlaggedItems] = useState([])
    const [reviewSummary, setReviewSummary] = useState(null)
    const [selectedItem, setSelectedItem] = useState(null)
    const [correction, setCorrection] = useState('')
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        loadReviewData()
    }, [document.document_id])

    const loadReviewData = async () => {
        setIsLoading(true)
        try {
            const [flagsRes, summaryRes] = await Promise.all([
                fetch(`${API_BASE_URL}/api/review/${document.document_id}/flags`),
                fetch(`${API_BASE_URL}/api/review/${document.document_id}/summary`)
            ]);

            if (flagsRes.ok) {
                const flagsData = await flagsRes.json()
                setFlaggedItems(flagsData.items || [])
            }

            if (summaryRes.ok) {
                const summaryData = await summaryRes.json()
                setReviewSummary(summaryData)
            }
        } catch (error) {
            console.error('Failed to load review data:', error)
        } finally {
            setIsLoading(false)
        }
    }

    const submitCorrection = async (fieldId, correctionType) => {
        const item = flaggedItems.find(i => i.field_id === fieldId)
        if (!item) return

        try {
            const response = await fetch(
                `${API_BASE_URL}/api/review/${document.document_id}/correct`,
                {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        field_id: fieldId,
                        original_value: item.current_value,
                        corrected_value: correctionType === 'edit' ? correction : item.current_value,
                        correction_type: correctionType
                    })
                }
            )

            if (response.ok) {
                // Remove from list and update summary
                setFlaggedItems(prev => prev.filter(i => i.field_id !== fieldId))
                setSelectedItem(null)
                setCorrection('')
                loadReviewData()
            }
        } catch (error) {
            console.error('Failed to submit correction:', error)
        }
    }

    const getConfidenceClass = (confidence) => {
        if (confidence >= 0.8) return 'confidence-high'
        if (confidence >= 0.6) return 'confidence-medium'
        return 'confidence-low'
    }

    if (isLoading) {
        return (
            <div className="review-loading">
                <div className="spinner"></div>
                <p>Loading review data...</p>
            </div>
        )
    }

    return (
        <div className="review-page">
            {/* Review Summary */}
            <section className="review-summary animate-slide-up">
                <h2>Human Review Dashboard</h2>

                <div className="summary-cards">
                    <div className="summary-card">
                        <div className="summary-icon">📋</div>
                        <div className="summary-content">
                            <span className="summary-value">{reviewSummary?.total_flagged || 0}</span>
                            <span className="summary-label">Total Flagged</span>
                        </div>
                    </div>

                    <div className="summary-card">
                        <div className="summary-icon">✅</div>
                        <div className="summary-content">
                            <span className="summary-value">{reviewSummary?.reviewed || 0}</span>
                            <span className="summary-label">Reviewed</span>
                        </div>
                    </div>

                    <div className="summary-card">
                        <div className="summary-icon">⏳</div>
                        <div className="summary-content">
                            <span className="summary-value">{reviewSummary?.pending || 0}</span>
                            <span className="summary-label">Pending</span>
                        </div>
                    </div>

                    <div className="summary-card">
                        <div className="summary-icon">✏️</div>
                        <div className="summary-content">
                            <span className="summary-value">{reviewSummary?.corrections_made || 0}</span>
                            <span className="summary-label">Corrections</span>
                        </div>
                    </div>
                </div>
            </section>

            {/* Flagged Items List */}
            <section className="flagged-items-section animate-slide-up" style={{ animationDelay: '100ms' }}>
                <div className="section-header">
                    <h3>⚠️ Items Requiring Review</h3>
                    <span className="item-count">{flaggedItems.length} items</span>
                </div>

                {flaggedItems.length === 0 ? (
                    <div className="no-items">
                        <div className="no-items-icon">✨</div>
                        <h4>All Clear!</h4>
                        <p>No items require human review. All extractions meet confidence thresholds.</p>
                    </div>
                ) : (
                    <div className="items-grid">
                        {flaggedItems.map((item) => (
                            <div
                                key={item.field_id}
                                className={`flagged-item ${selectedItem?.field_id === item.field_id ? 'selected' : ''}`}
                                onClick={() => {
                                    setSelectedItem(item)
                                    setCorrection(item.current_value)
                                }}
                            >
                                <div className="item-header">
                                    <span className="item-name">{item.field_name}</span>
                                    <div className="confidence-indicator">
                                        <span className={`confidence-dot ${getConfidenceClass(item.confidence)}`}></span>
                                        <span className="confidence-value">{Math.round(item.confidence * 100)}%</span>
                                    </div>
                                </div>

                                <div className="item-value">
                                    {item.current_value.substring(0, 100)}
                                    {item.current_value.length > 100 && '...'}
                                </div>

                                <div className="item-reason">
                                    <span className="reason-icon">💡</span>
                                    <span>{item.review_reason}</span>
                                </div>

                                {item.page_number !== null && (
                                    <span className="item-page">Page {item.page_number + 1}</span>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </section>

            {/* Correction Modal */}
            {selectedItem && (
                <div className="correction-modal animate-fade-in">
                    <div className="modal-content">
                        <div className="modal-header">
                            <h3>Review: {selectedItem.field_name}</h3>
                            <button
                                className="close-btn"
                                onClick={() => {
                                    setSelectedItem(null)
                                    setCorrection('')
                                }}
                            >
                                ×
                            </button>
                        </div>

                        <div className="modal-body">
                            <div className="current-value-section">
                                <label>Current Extracted Value</label>
                                <div className="current-value">{selectedItem.current_value}</div>
                            </div>

                            <div className="confidence-section">
                                <label>Confidence</label>
                                <div className={`confidence-bar ${getConfidenceClass(selectedItem.confidence)}`}>
                                    <div
                                        className="confidence-fill"
                                        style={{ width: `${selectedItem.confidence * 100}%` }}
                                    ></div>
                                    <span>{Math.round(selectedItem.confidence * 100)}%</span>
                                </div>
                            </div>

                            <div className="reason-section">
                                <label>Review Reason</label>
                                <p>{selectedItem.review_reason}</p>
                            </div>

                            <div className="correction-section">
                                <label>Corrected Value (if editing)</label>
                                <textarea
                                    className="correction-input"
                                    value={correction}
                                    onChange={(e) => setCorrection(e.target.value)}
                                    placeholder="Enter corrected value..."
                                />
                            </div>
                        </div>

                        <div className="modal-actions">
                            <button
                                className="btn btn-ghost"
                                onClick={() => submitCorrection(selectedItem.field_id, 'reject')}
                            >
                                ❌ Reject
                            </button>
                            <button
                                className="btn btn-secondary"
                                onClick={() => submitCorrection(selectedItem.field_id, 'confirm')}
                            >
                                ✓ Confirm Original
                            </button>
                            <button
                                className="btn btn-primary"
                                onClick={() => submitCorrection(selectedItem.field_id, 'edit')}
                            >
                                ✏️ Save Correction
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default ReviewPanel
