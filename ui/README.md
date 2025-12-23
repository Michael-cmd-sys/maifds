# MAIFDS (MindSpore Powered Fraud Defense System) - Frontend UI

A modern, high-performance web interface for the MAIFDS fraud detection platform. Built for the Huawei Innovation Competition to demonstrate advanced AI-powered security capabilities.

![MAIFDS UI](public/logo.png)

## 🚀 Features

*   **Glassmorphism Design**: Premium, translucent UI with smooth framer-motion animations.
*   **Real-time Dashboard**: Live system health, API status, and key performance indicators.
*   **Feature Playground**: Interactive testing suite for all 11 core fraud detection APIs.
*   **Blacklist Management**: Manage blocked entities (phone, URL, device) with visual feedback.
*   **Customer Reputation**: Graph-based risk scoring and suspicious transaction analysis.
*   **Governance Suite**:
    *   **Privacy Dashboard**: PII classification and anonymization metrics (GDPR compliant).
    *   **Audit Logs**: Immutable record of system events.
*   **Alerts System**: Live feed of security threats and community-reported risks.

## 🛠 Tech Stack

*   **Framework**: React 19 + TypeScript + Vite
*   **Styling**: Tailwind CSS v4 + PostCSS
*   **Animations**: Framer Motion
*   **Charts**: Recharts
*   **Routing**: React Router DOM (v6)
*   **State/API**: Axios + React Hooks
*   **Icons**: Lucide React

## 📦 Installation & Setup

1.  **Navigate to the UI directory:**
    ```bash
    cd ui
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Environment Configuration:**
    Create a `.env` file in the root `ui` directory (or rename `.env.example`):
    ```env
    VITE_API_BASE_URL=http://127.0.0.1:8000
    ```

4.  **Run Development Server:**
    ```bash
    npm run dev
    ```
    Access the app at `http://localhost:5173`.

5.  **Build for Production:**
    ```bash
    npm run build
    ```
    The output will be in the `dist` folder.

## 📂 Project Structure

```
ui/
├── src/
│   ├── api/          # Axios client & endpoints
│   ├── components/   # Reusable UI components (Glass cards, inputs, etc.)
│   ├── layouts/      # Dashboard & Page layouts
│   ├── pages/        # Main application views (Overview, Playground, etc.)
│   ├── theme/        # Theme provider (Light/Dark mode)
│   └── utils/        # Helper functions
├── public/           # Static assets
└── index.html        # Entry point
```

## 🎨 Design System

The UI uses a strict color palette defined in `index.css`:
*   **Primary**: `#0B1220` (Deep Navy)
*   **Accent**: `#22D3EE` (Cyan)
*   **Glass**: `rgba(255, 255, 255, 0.1)` with `backdrop-filter: blur(12px)`

---
**Powered by Huawei MindSpore**
