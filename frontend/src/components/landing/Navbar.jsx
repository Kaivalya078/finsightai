import { Link } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { Menu, X, Sparkles } from 'lucide-react';
import './../../styles/landing.css';

export default function Navbar() {
    const [isScrolled, setIsScrolled] = useState(false);
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 20);
        };

        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const toggleMobileMenu = () => {
        setIsMobileMenuOpen(!isMobileMenuOpen);
    };

    return (
        <nav className={`landing-nav ${isScrolled ? 'scrolled' : ''}`}>
            <div className="nav-container">
                {/* Logo */}
                <Link to="/" className="nav-logo">
                    <span className="nav-logo-mark">C</span>
                    <span className="logo-text">Cognifin</span>
                </Link>

                {/* Desktop Navigation Links */}
                <div className="nav-links">
                    <a href="#features" className="nav-link">
                        <span>Features</span>
                        <div className="nav-link-underline"></div>
                    </a>
                    <a href="#how-it-works" className="nav-link">
                        <span>How it Works</span>
                        <div className="nav-link-underline"></div>
                    </a>
                    <a href="#about" className="nav-link">
                        <span>About</span>
                        <div className="nav-link-underline"></div>
                    </a>
                </div>

                {/* CTA Button */}
                <Link to="/login" className="nav-cta-btn">
                    <span>Try for Free</span>
                    <div className="btn-glow"></div>
                    <svg className="btn-arrow" width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M1 8h14M9 2l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                </Link>

                {/* Mobile Menu Button */}
                <button className="mobile-menu-btn" onClick={toggleMobileMenu} aria-label="Toggle menu">
                    {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
                </button>
            </div>

            {/* Mobile Menu */}
            <div className={`mobile-menu ${isMobileMenuOpen ? 'open' : ''}`}>
                <div className="mobile-menu-content">
                    <a href="#features" className="mobile-nav-link" onClick={toggleMobileMenu}>
                        Features
                    </a>
                    <a href="#how-it-works" className="mobile-nav-link" onClick={toggleMobileMenu}>
                        How it Works
                    </a>
                    <a href="#about" className="mobile-nav-link" onClick={toggleMobileMenu}>
                        About
                    </a>
                    <Link to="/login" className="mobile-cta-btn" onClick={toggleMobileMenu}>
                        Try for Free →
                    </Link>
                </div>
            </div>
        </nav>
    );
}
