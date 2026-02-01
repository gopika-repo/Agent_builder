import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import ChatInterface from './ChatInterface'

// Mock document prop
const mockDocument = {
    document_id: 'test-doc-123',
    filename: 'test-document.pdf'
}

describe('ChatInterface', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        global.fetch.mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({ suggestions: [] })
        })
    })

    it('renders welcome message initially', () => {
        render(<ChatInterface document={mockDocument} />)

        expect(screen.getByText('Ask anything about your document')).toBeInTheDocument()
        expect(screen.getByText(mockDocument.filename)).toBeInTheDocument()
    })

    it('renders mode toggle buttons', () => {
        render(<ChatInterface document={mockDocument} />)

        expect(screen.getByText('🎈 ELI5')).toBeInTheDocument()
        expect(screen.getByText('📝 Standard')).toBeInTheDocument()
        expect(screen.getByText('🎓 Expert')).toBeInTheDocument()
    })

    it('changes mode when toggle buttons are clicked', () => {
        render(<ChatInterface document={mockDocument} />)

        const eli5Button = screen.getByText('🎈 ELI5')
        fireEvent.click(eli5Button)

        expect(eli5Button.className).toContain('active')
    })

    it('renders input textarea', () => {
        render(<ChatInterface document={mockDocument} />)

        expect(screen.getByPlaceholderText('Ask a question about this document...')).toBeInTheDocument()
    })

    it('disables send button when input is empty', () => {
        render(<ChatInterface document={mockDocument} />)

        const sendButton = screen.getByText('Send →')
        expect(sendButton).toBeDisabled()
    })

    it('enables send button when input has text', () => {
        render(<ChatInterface document={mockDocument} />)

        const input = screen.getByPlaceholderText('Ask a question about this document...')
        fireEvent.change(input, { target: { value: 'What is this document about?' } })

        const sendButton = screen.getByText('Send →')
        expect(sendButton).not.toBeDisabled()
    })

    it('sends message and displays user message', async () => {
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ suggestions: [] })
        }).mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({
                answer: 'This is a test response',
                sources: [],
                mode: 'standard'
            })
        })

        render(<ChatInterface document={mockDocument} />)

        const input = screen.getByPlaceholderText('Ask a question about this document...')
        fireEvent.change(input, { target: { value: 'Test question' } })

        const sendButton = screen.getByText('Send →')
        fireEvent.click(sendButton)

        await waitFor(() => {
            expect(screen.getByText('Test question')).toBeInTheDocument()
        })
    })

    it('shows loading indicator while waiting for response', async () => {
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ suggestions: [] })
        }).mockImplementationOnce(() => new Promise(() => { })) // Never resolves

        render(<ChatInterface document={mockDocument} />)

        const input = screen.getByPlaceholderText('Ask a question about this document...')
        fireEvent.change(input, { target: { value: 'Test question' } })

        const sendButton = screen.getByText('Send →')
        fireEvent.click(sendButton)

        await waitFor(() => {
            expect(screen.getByText('Test question')).toBeInTheDocument()
        })
    })

    it('displays suggestions when loaded', async () => {
        global.fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({
                suggestions: ['What is the summary?', 'Show key figures']
            })
        })

        render(<ChatInterface document={mockDocument} />)

        await waitFor(() => {
            expect(screen.getByText('What is the summary?')).toBeInTheDocument()
        })
    })
})
