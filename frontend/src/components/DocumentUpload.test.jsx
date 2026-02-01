import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import DocumentUpload from './DocumentUpload'

// Mock props
const mockProps = {
    onUpload: vi.fn(),
    isLoading: false,
    documents: [],
    onSelectDocument: vi.fn(),
    onRefresh: vi.fn()
}

describe('DocumentUpload', () => {
    beforeEach(() => {
        vi.clearAllMocks()
    })

    it('renders upload zone', () => {
        render(<DocumentUpload {...mockProps} />)

        expect(screen.getByText('Upload Document')).toBeInTheDocument()
        expect(screen.getByText('Drag & drop or click to browse')).toBeInTheDocument()
    })

    it('renders supported format badges', () => {
        render(<DocumentUpload {...mockProps} />)

        expect(screen.getByText('PDF')).toBeInTheDocument()
        expect(screen.getByText('PNG')).toBeInTheDocument()
        expect(screen.getByText('JPG')).toBeInTheDocument()
        expect(screen.getByText('TIFF')).toBeInTheDocument()
    })

    it('renders feature cards', () => {
        render(<DocumentUpload {...mockProps} />)

        expect(screen.getByText('Vision Analysis')).toBeInTheDocument()
        expect(screen.getByText('Hybrid OCR')).toBeInTheDocument()
        expect(screen.getByText('Layout Graph')).toBeInTheDocument()
        expect(screen.getByText('AI Reasoning')).toBeInTheDocument()
        expect(screen.getByText('Multi-Modal Fusion')).toBeInTheDocument()
        expect(screen.getByText('Confidence Scoring')).toBeInTheDocument()
    })

    it('shows loading state when processing', () => {
        render(<DocumentUpload {...mockProps} isLoading={true} />)

        expect(screen.getByText('Processing document...')).toBeInTheDocument()
    })

    it('shows empty state when no documents', () => {
        render(<DocumentUpload {...mockProps} />)

        expect(screen.getByText('No documents yet. Upload your first document to get started!')).toBeInTheDocument()
    })

    it('renders document list with status badges', () => {
        const documents = [
            {
                document_id: 'doc-1',
                filename: 'report.pdf',
                status: 'completed',
                created_at: '2026-01-15T10:00:00Z'
            },
            {
                document_id: 'doc-2',
                filename: 'invoice.png',
                status: 'processing',
                created_at: '2026-01-16T10:00:00Z'
            }
        ]

        render(<DocumentUpload {...mockProps} documents={documents} />)

        expect(screen.getByText('report.pdf')).toBeInTheDocument()
        expect(screen.getByText('invoice.png')).toBeInTheDocument()
        expect(screen.getByText('Completed')).toBeInTheDocument()
        expect(screen.getByText('Processing')).toBeInTheDocument()
    })

    it('calls onSelectDocument when clicking completed document', () => {
        const documents = [
            {
                document_id: 'doc-1',
                filename: 'report.pdf',
                status: 'completed',
                created_at: '2026-01-15T10:00:00Z'
            }
        ]

        render(<DocumentUpload {...mockProps} documents={documents} />)

        const docRow = screen.getByText('report.pdf').closest('.document-row')
        fireEvent.click(docRow)

        expect(mockProps.onSelectDocument).toHaveBeenCalledWith('doc-1')
    })

    it('calls onRefresh when clicking refresh button', () => {
        render(<DocumentUpload {...mockProps} />)

        const refreshButton = screen.getByText('🔄 Refresh')
        fireEvent.click(refreshButton)

        expect(mockProps.onRefresh).toHaveBeenCalled()
    })

    it('handles file selection via click', () => {
        render(<DocumentUpload {...mockProps} />)

        const file = new File(['test'], 'test.pdf', { type: 'application/pdf' })
        const input = document.querySelector('input[type="file"]')

        fireEvent.change(input, { target: { files: [file] } })

        expect(mockProps.onUpload).toHaveBeenCalledWith(file)
    })

    it('handles drag and drop', () => {
        render(<DocumentUpload {...mockProps} />)

        const uploadZone = document.querySelector('.upload-zone')
        const file = new File(['test'], 'test.pdf', { type: 'application/pdf' })

        // Simulate drag over
        fireEvent.dragOver(uploadZone, { preventDefault: vi.fn() })

        // Simulate drop
        fireEvent.drop(uploadZone, {
            preventDefault: vi.fn(),
            dataTransfer: { files: [file] }
        })

        expect(mockProps.onUpload).toHaveBeenCalledWith(file)
    })
})
