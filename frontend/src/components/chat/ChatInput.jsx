/**
 * ChatInput Component
 * ====================
 * Bottom input area for sending messages.
 * Includes PDF upload button that opens an inline form.
 */

import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, X, Upload } from 'lucide-react';

export default function ChatInput({ onSend, onUpload, isLoading, uploadedFile, onClearUpload }) {
    const [question, setQuestion] = useState('');
    const [showUploadForm, setShowUploadForm] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const [companyName, setCompanyName] = useState('');
    const [year, setYear] = useState(String(new Date().getFullYear()));
    const textareaRef = useRef(null);
    const fileInputRef = useRef(null);

    // Auto-resize textarea
    useEffect(() => {
        const el = textareaRef.current;
        if (el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 160) + 'px';
        }
    }, [question]);

    const handleSubmit = (e) => {
        e.preventDefault();
        const trimmed = question.trim();
        if (!trimmed || isLoading) return;
        onSend(trimmed);
        setQuestion('');
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        setSelectedFile(file);
        // Auto-fill company name from filename (remove year/ext)
        const name = file.name.replace(/\.[^.]+$/, '').replace(/[\d_-]+/g, ' ').trim();
        if (!companyName) setCompanyName(name);
    };

    const handleUploadSubmit = (e) => {
        e.preventDefault();
        if (!selectedFile || !companyName.trim()) return;
        onUpload(selectedFile, companyName.trim(), year);
        setShowUploadForm(false);
        setSelectedFile(null);
        setCompanyName('');
    };

    const cancelUploadForm = () => {
        setShowUploadForm(false);
        setSelectedFile(null);
        setCompanyName('');
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    return (
        <div className="chat-input-area">
            {/* Active session pill */}
            {uploadedFile && (
                <div className="upload-session-pill">
                    <div className="chat-input-inner">
                        <div className="upload-session-box">
                            <Paperclip size={12} />
                            <span>{uploadedFile} · Session active for this chat</span>
                            <button
                                className="upload-session-clear"
                                onClick={onClearUpload}
                                title="Remove uploaded PDF (revert to global corpus)"
                            >
                                <X size={12} />
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Upload form (inline, above input) */}
            {showUploadForm && (
                <div className="upload-form-panel">
                    <div className="chat-input-inner">
                        <div className="upload-form-box">
                            <div className="upload-form-header">
                                <span>Upload PDF for this chat</span>
                                <button className="upload-form-close" onClick={cancelUploadForm}>
                                    <X size={14} />
                                </button>
                            </div>
                            <form onSubmit={handleUploadSubmit} className="upload-form-body">
                                <div
                                    className={`upload-drop-zone ${selectedFile ? 'has-file' : ''}`}
                                    onClick={() => fileInputRef.current?.click()}
                                >
                                    <input
                                        ref={fileInputRef}
                                        type="file"
                                        accept=".pdf"
                                        style={{ display: 'none' }}
                                        onChange={handleFileChange}
                                    />
                                    {selectedFile ? (
                                        <>
                                            <Upload size={16} />
                                            <span className="upload-filename">{selectedFile.name}</span>
                                            <span className="upload-filesize">
                                                ({(selectedFile.size / 1024 / 1024).toFixed(1)} MB)
                                            </span>
                                        </>
                                    ) : (
                                        <>
                                            <Upload size={20} />
                                            <span>Click to select a PDF</span>
                                        </>
                                    )}
                                </div>
                                <div className="upload-fields">
                                    <input
                                        className="upload-field-input"
                                        type="text"
                                        placeholder="Company name *"
                                        value={companyName}
                                        onChange={e => setCompanyName(e.target.value)}
                                        required
                                    />
                                    <input
                                        className="upload-field-input upload-field-year"
                                        type="text"
                                        placeholder="Year"
                                        value={year}
                                        onChange={e => setYear(e.target.value)}
                                        maxLength={4}
                                    />
                                </div>
                                <button
                                    type="submit"
                                    className="upload-submit-btn"
                                    disabled={!selectedFile || !companyName.trim()}
                                >
                                    <Upload size={14} />
                                    Start Processing
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            )}

            {/* Main input row */}
            <form className="chat-input-form" onSubmit={handleSubmit}>
                <div className="chat-input-inner">
                    <div className="chat-input-wrapper">
                        {/* Upload trigger */}
                        <button
                            type="button"
                            className={`chat-upload-btn ${uploadedFile ? 'active' : ''}`}
                            onClick={() => setShowUploadForm(v => !v)}
                            title={uploadedFile ? 'PDF session active' : 'Upload a PDF for this chat'}
                            disabled={isLoading}
                        >
                            <Paperclip size={16} />
                        </button>

                        <textarea
                            ref={textareaRef}
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder={
                                uploadedFile
                                    ? `Ask about ${uploadedFile} or the global corpus...`
                                    : 'Ask about the financial document...'
                            }
                            disabled={isLoading}
                            rows={1}
                        />
                        <button
                            type="submit"
                            className="chat-send-btn"
                            disabled={isLoading || !question.trim()}
                            title="Send message"
                        >
                            <Send size={18} />
                        </button>
                    </div>
                    <p className="chat-input-hint">
                        Press Enter to send · Shift+Enter for new line
                        {uploadedFile && <span className="hint-session"> · PDF session active</span>}
                    </p>
                </div>
            </form>
        </div>
    );
}
