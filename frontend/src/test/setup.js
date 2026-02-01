import '@testing-library/jest-dom'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// Cleanup after each test
afterEach(() => {
    cleanup()
})

// Mock fetch globally
global.fetch = vi.fn()

// Mock scrollIntoView
Element.prototype.scrollIntoView = vi.fn()
