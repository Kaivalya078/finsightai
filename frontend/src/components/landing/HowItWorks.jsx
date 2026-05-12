/**
 * HowItWorks Section
 * ====================
 * 3-step visual process showing how FinSight AI works.
 */

import { motion } from 'framer-motion';
import { Upload, Search, MessageSquare } from 'lucide-react';

const steps = [
    {
        icon: Upload,
        step: '01',
        title: 'Upload Document',
        description: 'Upload your SEBI filing, DRHP, or annual report. We process and index the entire document.',
    },
    {
        icon: Search,
        step: '02',
        title: 'Ask Questions',
        description: 'Ask anything in natural language. Our AI searches through every page to find relevant information.',
    },
    {
        icon: MessageSquare,
        step: '03',
        title: 'Get Grounded Answers',
        description: 'Receive accurate answers with citations pointing to the exact source sections in the document.',
    },
];

export default function HowItWorks() {
    return (
        <section className="how-it-works-section" id="how-it-works">
            <div className="hiw-container">
                <motion.div
                    className="hiw-header"
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.6 }}
                >
                    <h2 className="hiw-title">
                        How It <span className="gradient-text">Works</span>
                    </h2>
                    <p className="hiw-subtitle">
                        Three simple steps to unlock insights from any financial document
                    </p>
                </motion.div>

                <div className="hiw-steps">
                    {steps.map((item, index) => (
                        <motion.div
                            key={index}
                            className="hiw-step"
                            initial={{ opacity: 0, y: 30 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.5, delay: index * 0.15 }}
                        >
                            <div className="hiw-step-number">{item.step}</div>
                            <div className="hiw-step-icon">
                                <item.icon size={28} />
                            </div>
                            <h3 className="hiw-step-title">{item.title}</h3>
                            <p className="hiw-step-desc">{item.description}</p>

                            {/* Connector line */}
                            {index < steps.length - 1 && (
                                <div className="hiw-connector">
                                    <div className="hiw-connector-line"></div>
                                </div>
                            )}
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
}
