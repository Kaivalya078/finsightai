import { Link } from 'react-router-dom';
import { ArrowRight, Sparkles } from 'lucide-react';
import './../../styles/landing.css';

export default function Hero() {
    return (
        <section className="hero-section">
            {/* Glassmorphism Hero Card */}
            <div className="hero-glass-card">
                {/* Badge */}
                <div className="hero-badge">
                    <Sparkles size={14} />
                    <span>Unlock Your Assets Spark!</span>
                </div>

                {/* Main Heading */}
                <h1 className="hero-title">
                    Instant Insights from
                    <br />
                    <span className="gradient-text">Financial Documents</span>
                </h1>

                {/* Subtitle */}
                <p className="hero-subtitle">
                    Dive into Indian financial documents, where innovative AI technology
                    meets financial expertise. Ask questions, get instant answers backed by evidence.
                </p>

                {/* CTA Buttons */}
                <div className="hero-cta-group">
                    <Link to="/login" className="cta-primary">
                        Try for Free
                        <ArrowRight size={18} />
                    </Link>
                    <a href="#features" className="cta-secondary">
                        Discover More
                    </a>
                </div>
            </div>
        </section>
    );
}
