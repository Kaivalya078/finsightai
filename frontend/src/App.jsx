/**
 * FinSight AI - Main App
 * =======================
 * Multi-page application with routing, authentication, and page transitions
 */

import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { AuthProvider } from './context/AuthContext';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ChatPage from './pages/ChatPage';
import AuthCallbackPage from './pages/AuthCallbackPage';
import ProtectedRoute from './components/shared/ProtectedRoute';
import "./App.css";

function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={
          <PageTransition>
            <LandingPage />
          </PageTransition>
        } />
        <Route path="/login" element={
          <PageTransition>
            <LoginPage />
          </PageTransition>
        } />
        <Route path="/register" element={
          <PageTransition>
            <RegisterPage />
          </PageTransition>
        } />
        <Route path="/auth/callback" element={
          <AuthCallbackPage />
        } />
        <Route path="/chat" element={
          <ProtectedRoute>
            <ChatPage />
          </ProtectedRoute>
        } />
      </Routes>
    </AnimatePresence>
  );
}

function PageTransition({ children }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
    >
      {children}
    </motion.div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <AnimatedRoutes />
      </Router>
    </AuthProvider>
  );
}
