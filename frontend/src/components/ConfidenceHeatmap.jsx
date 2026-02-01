import React, { useState, useEffect, useRef } from 'react';
import './ConfidenceHeatmap.css';

/**
 * Confidence Heatmap Component
 * 
 * Displays visual confidence overlay on document pages
 * Color-coded: green=high, yellow=medium, red=low
 */
const ConfidenceHeatmap = ({
    documentId,
    pageNumber = 0,
    pageImage,
    onRegionClick
}) => {
    const [regions, setRegions] = useState([]);
    const [statistics, setStatistics] = useState(null);
    const [legend, setLegend] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [hoveredRegion, setHoveredRegion] = useState(null);
    const containerRef = useRef(null);

    // Fetch heatmap data
    useEffect(() => {
        const fetchHeatmap = async () => {
            if (!documentId) return;

            setLoading(true);
            setError(null);

            try {
                const response = await fetch('/api/advanced/heatmap/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ document_id: documentId, page_number: pageNumber })
                });

                if (!response.ok) throw new Error('Failed to load heatmap');

                const data = await response.json();
                setRegions(data.regions || []);
                setStatistics(data.statistics);
                setLegend(data.legend);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchHeatmap();
    }, [documentId, pageNumber]);

    const handleRegionClick = (region) => {
        if (onRegionClick) {
            onRegionClick(region);
        }
    };

    if (loading) {
        return (
            <div className="heatmap-loading">
                <div className="spinner"></div>
                <span>Generating confidence heatmap...</span>
            </div>
        );
    }

    if (error) {
        return (
            <div className="heatmap-error">
                <span>⚠️ {error}</span>
            </div>
        );
    }

    return (
        <div className="confidence-heatmap" ref={containerRef}>
            {/* Legend */}
            {legend && (
                <div className="heatmap-legend">
                    <h4>Confidence Legend</h4>
                    <div className="legend-items">
                        <div className="legend-item">
                            <span className="legend-color high"></span>
                            <span>High ({legend.high?.threshold})</span>
                        </div>
                        <div className="legend-item">
                            <span className="legend-color medium"></span>
                            <span>Medium ({legend.medium?.threshold})</span>
                        </div>
                        <div className="legend-item">
                            <span className="legend-color low"></span>
                            <span>Low ({legend.low?.threshold})</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Statistics */}
            {statistics && (
                <div className="heatmap-stats">
                    <div className="stat">
                        <span className="stat-value">{statistics.total_regions}</span>
                        <span className="stat-label">Regions</span>
                    </div>
                    <div className="stat">
                        <span className="stat-value">{(statistics.average_confidence * 100).toFixed(0)}%</span>
                        <span className="stat-label">Avg Confidence</span>
                    </div>
                    <div className="stat warning">
                        <span className="stat-value">{statistics.low_confidence}</span>
                        <span className="stat-label">Need Review</span>
                    </div>
                </div>
            )}

            {/* Heatmap Overlay */}
            <div className="heatmap-container">
                {pageImage && (
                    <img src={pageImage} alt={`Page ${pageNumber + 1}`} className="page-image" />
                )}

                <div className="heatmap-overlay">
                    {regions.map((region) => (
                        <div
                            key={region.id}
                            className={`confidence-region ${region.requiresReview ? 'needs-review' : ''}`}
                            style={{
                                left: `${region.x}px`,
                                top: `${region.y}px`,
                                width: `${region.width}px`,
                                height: `${region.height}px`,
                                backgroundColor: `${region.color}${Math.round(region.opacity * 255).toString(16).padStart(2, '0')}`,
                                borderColor: region.color
                            }}
                            onMouseEnter={() => setHoveredRegion(region)}
                            onMouseLeave={() => setHoveredRegion(null)}
                            onClick={() => handleRegionClick(region)}
                        >
                            {region.requiresReview && (
                                <span className="review-indicator">!</span>
                            )}
                        </div>
                    ))}
                </div>

                {/* Tooltip */}
                {hoveredRegion && (
                    <div
                        className="region-tooltip"
                        style={{
                            left: `${hoveredRegion.x + hoveredRegion.width / 2}px`,
                            top: `${hoveredRegion.y - 10}px`
                        }}
                    >
                        <div className="tooltip-label">{hoveredRegion.label}</div>
                        <div className="tooltip-confidence">
                            Confidence: {(hoveredRegion.confidence * 100).toFixed(0)}%
                        </div>
                        <div className="tooltip-source">Source: {hoveredRegion.source}</div>
                        {hoveredRegion.requiresReview && (
                            <div className="tooltip-warning">⚠️ Requires human review</div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ConfidenceHeatmap;
