# Document Intelligence Frontend ⚛️

A premium, React-based user interface for the Multi-Modal Document Intelligence Platform. Built with modern web standards to ensure a fast, responsive, and accessible user experience.

![React](https://img.shields.io/badge/React-18.3+-61DAFB?logo=react)
![Vite](https://img.shields.io/badge/Vite-5.4+-646CFF?logo=vite)
![Tailwind](https://img.shields.io/badge/CSS-Vanilla-38B2AC?logo=css3)

## ✨ Key Features

- **glassmorphism Design**: Modern, translucent UI components with rich gradients.
- **Real-Time Chat**: Interactive chat interface with streaming responses.
- **Document comparison**: Split-screen view for comparing ELI5 vs Expert explanations.
- **Confidence Visualization**: Color-coded heatmaps showing extraction confidence.
- **Drag & Drop Upload**: Intuitive file management system.

## 🏗️ Architecture

### Component Structure

```
src/
├── components/
│   ├── ChatInterface.jsx   # Main chat & query interface
│   ├── DocumentUpload.jsx  # File upload with drag-and-drop
│   ├── DocumentViewer.jsx  # PDF/Image viewer with overlays
│   ├── ReviewPanel.jsx     # Comparison & Review tools
│   └── ...
├── App.jsx                 # Main layout & routing
└── index.css               # Global styles & variables
```

### State Management
- **Local State**: `useState` for component-level UI state (loading, toggles).
- **Props Drilling**: Data flow is manageable enough to avoid heavy Redux/Context complexity, keeping the app lightweight and performant.

### Styling Strategy
- **Vanilla CSS Variables**: Uses CSS variables (`--primary-color`, etc.) for consistent theming.
- **Scoped CSS**: Component-specific styles are imported directly into JSX files.
- **Animations**: CSS transitions and keyframes for smooth micro-interactions.

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- npm 9+

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at [http://localhost:5173](http://localhost:5173).

## 🧪 Testing

We use **Vitest** and **React Testing Library** for unit testing.

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage
```

## 📦 Building for Production

```bash
npm run build
```

This generates a static build in the `dist/` directory, optimized for production deployment (served via Nginx in Docker).
