/**
 * EvidenceList Component
 * ======================
 * Displays all evidence chunks used to generate the answer.
 * Each chunk is rendered as an expandable EvidenceItem.
 * 
 * Props:
 *   evidence - array of { chunk_id, snippet }
 */

import EvidenceItem from "./EvidenceItem";

export default function EvidenceList({ evidence }) {
    if (!evidence || evidence.length === 0) return null;

    return (
        <div className="evidence-list">
            <h2>Evidence ({evidence.length} chunks)</h2>
            {evidence.map((item) => (
                <EvidenceItem
                    key={item.chunk_id}
                    chunkId={item.chunk_id}
                    snippet={item.snippet}
                />
            ))}
        </div>
    );
}
