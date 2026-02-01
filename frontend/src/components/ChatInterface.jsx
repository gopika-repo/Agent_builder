import { useState, useRef, useEffect } from 'react'
import { API_BASE_URL } from '../config'
import './ChatInterface.css'

function ChatInterface({ document }) {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [mode, setMode] = useState('standard')
    const [isLoading, setIsLoading] = useState(false)
    const [suggestions, setSuggestions] = useState([])
    const messagesEndRef = useRef(null)

    // Load initial suggestions
    useEffect(() => {
        loadSuggestions()
    }, [document.document_id])

    // Scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    const loadSuggestions = async () => {
        try {
            const response = await fetch(
                `${API_BASE_URL}/api/chat/${document.document_id}/suggest`
            )
            if (response.ok) {
                const data = await response.json()
                setSuggestions(data.suggestions || [])
            }
        } catch (error) {
            console.error('Failed to load suggestions:', error)
        }
    }

    const sendMessage = async (query = input) => {
        if (!query.trim()) return

        const userMessage = { role: 'user', content: query }
        setMessages(prev => [...prev, userMessage])
        setInput('')
        setIsLoading(true)

        try {
            const response = await fetch(
                `${API_BASE_URL}/api/chat/${document.document_id}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        mode: mode,
                        include_sources: true
                    })
                }
            )

            if (response.ok) {
                const data = await response.json()
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: data.answer,
                    sources: data.sources,
                    mode: data.mode
                }])
            } else {
                throw new Error('Query failed')
            }
        } catch (error) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Sorry, I encountered an error processing your request. Please try again.',
                error: true
            }])
        } finally {
            setIsLoading(false)
        }
    }

    const compareExplanations = async (query) => {
        if (!query.trim()) return

        const userMessage = { role: 'user', content: query, isCompare: true }
        setMessages(prev => [...prev, userMessage])
        setInput('')
        setIsLoading(true)

        try {
            const response = await fetch(
                `${API_BASE_URL}/api/chat/${document.document_id}/explain?query=${encodeURIComponent(query)}`,
                { method: 'POST' }
            )

            if (response.ok) {
                const data = await response.json()
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    isCompare: true,
                    eli5: data.eli5,
                    expert: data.expert,
                    sources: data.sources
                }])
            }
        } catch (error) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Failed to compare explanations',
                error: true
            }])
        } finally {
            setIsLoading(false)
        }
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    return (
        <div className="chat-page">
            {/* Chat Header */}
            <div className="chat-header">
                <div className="chat-title">
                    <h2>💬 Ask AI about this document</h2>
                    <span className="document-name">{document.filename}</span>
                </div>

                {/* Mode Toggle */}
                <div className="mode-toggle">
                    <button
                        className={mode === 'eli5' ? 'active' : ''}
                        onClick={() => setMode('eli5')}
                    >
                        🎈 ELI5
                    </button>
                    <button
                        className={mode === 'standard' ? 'active' : ''}
                        onClick={() => setMode('standard')}
                    >
                        📝 Standard
                    </button>
                    <button
                        className={mode === 'expert' ? 'active' : ''}
                        onClick={() => setMode('expert')}
                    >
                        🎓 Expert
                    </button>
                </div>
            </div>

            {/* Messages */}
            <div className="chat-messages">
                {messages.length === 0 && (
                    <div className="chat-welcome animate-fade-in">
                        <div className="welcome-icon">🔮</div>
                        <h3>Ask anything about your document</h3>
                        <p>I can summarize, explain, find specific information, or compare explanations at different levels.</p>

                        {suggestions.length > 0 && (
                            <div className="suggestions">
                                <span className="suggestions-label">Try asking:</span>
                                <div className="suggestions-list">
                                    {suggestions.slice(0, 4).map((s, i) => (
                                        <button
                                            key={i}
                                            className="suggestion-chip"
                                            onClick={() => sendMessage(s)}
                                        >
                                            {s}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {messages.map((msg, i) => (
                    <div key={i} className={`message ${msg.role} animate-slide-up`}>
                        {msg.role === 'user' ? (
                            <div className="message-content user-message">
                                <div className="message-avatar">👤</div>
                                <div className="message-text">
                                    {msg.content}
                                    {msg.isCompare && (
                                        <span className="compare-badge">Comparing ELI5 vs Expert</span>
                                    )}
                                </div>
                            </div>
                        ) : msg.isCompare ? (
                            <div className="message-content compare-response">
                                <div className="compare-columns">
                                    <div className="compare-column eli5">
                                        <div className="column-header">
                                            <span className="column-icon">🎈</span>
                                            <span>ELI5 (Simple)</span>
                                        </div>
                                        <div className="column-content">{msg.eli5}</div>
                                    </div>
                                    <div className="compare-column expert">
                                        <div className="column-header">
                                            <span className="column-icon">🎓</span>
                                            <span>Expert (Technical)</span>
                                        </div>
                                        <div className="column-content">{msg.expert}</div>
                                    </div>
                                </div>
                                {msg.sources && msg.sources.length > 0 && (
                                    <div className="message-sources">
                                        <span className="sources-label">Sources:</span>
                                        {msg.sources.slice(0, 3).map((s, j) => (
                                            <span key={j} className="source-tag">
                                                {s.type} (p.{s.page + 1})
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className={`message-content assistant-message ${msg.error ? 'error' : ''}`}>
                                <div className="message-avatar">🤖</div>
                                <div className="message-body">
                                    <div className="message-text">{msg.content}</div>
                                    {msg.mode && (
                                        <span className={`mode-badge mode-${msg.mode}`}>
                                            {msg.mode === 'eli5' ? '🎈 ELI5' : msg.mode === 'expert' ? '🎓 Expert' : '📝 Standard'}
                                        </span>
                                    )}
                                    {msg.sources && msg.sources.length > 0 && (
                                        <div className="message-sources">
                                            <span className="sources-label">Sources:</span>
                                            {msg.sources.slice(0, 3).map((s, j) => (
                                                <span key={j} className="source-tag">
                                                    {s.type} (p.{s.page + 1})
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                ))}

                {isLoading && (
                    <div className="message assistant animate-fade-in">
                        <div className="message-content assistant-message loading">
                            <div className="message-avatar">🤖</div>
                            <div className="typing-indicator">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="chat-input-area">
                <div className="input-row">
                    <textarea
                        className="chat-input"
                        placeholder="Ask a question about this document..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={handleKeyPress}
                        disabled={isLoading}
                    />
                    <div className="input-actions">
                        <button
                            className="btn btn-ghost compare-btn"
                            onClick={() => compareExplanations(input)}
                            disabled={!input.trim() || isLoading}
                            title="Compare ELI5 vs Expert"
                        >
                            ⚖️
                        </button>
                        <button
                            className="btn btn-primary send-btn"
                            onClick={() => sendMessage()}
                            disabled={!input.trim() || isLoading}
                        >
                            Send →
                        </button>
                    </div>
                </div>
                <div className="input-hint">
                    Press Enter to send • Use ⚖️ to compare ELI5 vs Expert explanations
                </div>
            </div>
        </div>
    )
}

export default ChatInterface
