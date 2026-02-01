import './Header.css'

function Header({ activeTab, onTabChange, hasDocument }) {
    const tabs = [
        { id: 'upload', label: 'Documents', icon: '📁' },
        { id: 'viewer', label: 'Viewer', icon: '👁️', disabled: !hasDocument },
        { id: 'chat', label: 'Ask AI', icon: '💬', disabled: !hasDocument },
        { id: 'review', label: 'Review', icon: '✓', disabled: !hasDocument }
    ]

    return (
        <header className="header">
            <div className="header-content">
                <div className="header-brand">
                    <div className="logo">
                        <span className="logo-icon">🔮</span>
                        <span className="logo-text">DocIntel</span>
                    </div>
                    <span className="logo-tagline">Multi-Modal Document Intelligence</span>
                </div>

                <nav className="header-nav">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
                            onClick={() => onTabChange(tab.id)}
                            disabled={tab.disabled}
                        >
                            <span className="nav-icon">{tab.icon}</span>
                            <span className="nav-label">{tab.label}</span>
                        </button>
                    ))}
                </nav>

                <div className="header-actions">
                    <div className="status-indicator online">
                        <span className="status-dot"></span>
                        <span className="status-text">API Connected</span>
                    </div>
                </div>
            </div>
        </header>
    )
}

export default Header
