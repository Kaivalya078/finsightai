/**
 * AnswerView Component
 * ====================
 * Displays the AI-generated answer text.
 * 
 * Props:
 *   answer    - string, the generated answer
 *   citations - array of chunk IDs cited
 */

export default function AnswerView({ answer, citations }) {
    if (!answer) return null;

    return (
        <div className="answer-view">
            <h2>Answer</h2>
            <p className="answer-text">{answer}</p>
            {citations.length > 0 && (
                <div className="citations">
                    <span className="citations-label">Sources: </span>
                    {citations.map((id) => (
                        <span key={id} className="citation-badge">{id}</span>
                    ))}
                </div>
            )}
        </div>
    );
}
