import { Search, Zap, FileText, TrendingUp } from 'lucide-react';
import { motion } from 'framer-motion';
import './../../styles/landing.css';

const features = [
    {
        icon: Search,
        title: 'Semantic Search',
        description: 'Find exactly what you need with natural language queries across financial documents.'
    },
    {
        icon: Zap,
        title: 'AI-Powered Answers',
        description: 'Get instant, accurate answers powered by GPT-4, grounded in document evidence.'
    },
    {
        icon: FileText,
        title: 'Source Citations',
        description: 'Every answer includes citations to the exact document sections used.'
    },
    {
        icon: TrendingUp,
        title: 'Financial Intelligence',
        description: 'Specialized for SEBI filings, DRHP documents, and annual reports.'
    }
];

export default function Features() {
    return (
        <section className="features-section" id="features">
            <div className="features-container">
                {/* Section Header */}
                <motion.div
                    className="features-header"
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.6 }}
                >
                    <h2 className="features-title">
                        Powerful Features for
                        <span className="gradient-text"> Financial Analysis</span>
                    </h2>
                    <p className="features-subtitle">
                        Everything you need to extract insights from complex financial documents
                    </p>
                </motion.div>

                {/* Feature Grid */}
                <div className="features-grid">
                    {features.map((feature, index) => (
                        <motion.div
                            key={index}
                            className="feature-card"
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                            whileHover={{ y: -5 }}
                        >
                            <div className="feature-icon">
                                <feature.icon size={24} />
                            </div>
                            <h3 className="feature-title">{feature.title}</h3>
                            <p className="feature-description">{feature.description}</p>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
}
