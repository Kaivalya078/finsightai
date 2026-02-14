/**
 * EvidenceItem Component
 * ======================
 * A single expandable evidence chunk.
 * Click to expand/collapse the snippet text.
 * 
 * Props:
 *   chunkId - string, e.g. "chunk_12"
 *   snippet - string, the chunk text
 */

import { useState } from "react";

export default function EvidenceItem({ chunkId, snippet }) {
    const [expanded, setExpanded] = useState(false);

    return (
        <div className={`evidence-item ${expanded ? "expanded" : ""}`}>
            <button
                className="evidence-header"
                onClick={() => setExpanded(!expanded)}
            >
                <span className="evidence-id">{chunkId}</span>
                <span className="evidence-preview">
                    {expanded ? "" : snippet.slice(0, 80) + "..."}
                </span>
                <span className="evidence-toggle">{expanded ? "▲" : "▼"}</span>
            </button>
            {expanded && (
                <div className="evidence-body">
                    <p>{snippet}</p>
                </div>
            )}
        </div>
    );
}
