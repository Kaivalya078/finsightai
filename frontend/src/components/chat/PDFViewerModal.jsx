/**
 * PDFViewerModal.jsx
 * In-app PDF viewer with page jump + text highlight.
 * Uses pdfjs-dist to render the PDF page on a <canvas> element,
 * then draws yellow highlight boxes over words matching the chunk snippet.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { X, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, ExternalLink } from 'lucide-react';
import * as pdfjsLib from 'pdfjs-dist';
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

// Point pdfjs to the worker — imported from node_modules (Vite resolves via ?url)
pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorker;

const HIGHLIGHT_COLOR = 'rgba(212, 175, 55, 0.38)';
const MIN_SCALE = 0.5;
const MAX_SCALE = 3.0;
const DEFAULT_SCALE = 1.2;  // lower = faster render

export default function PDFViewerModal({ pdfUrl, pageNumber, chunkText, onClose }) {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const pdfRef = useRef(null);
    const renderTaskRef = useRef(null);

    const [currentPage, setCurrentPage] = useState(pageNumber || 1);
    const [totalPages, setTotalPages] = useState(0);
    const [scale, setScale] = useState(DEFAULT_SCALE);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Tokenise chunk text into meaningful words for highlight matching
    const highlightWords = chunkText
        ? chunkText
              .replace(/[^a-zA-Z0-9\s₹%.,]/g, ' ')
              .split(/\s+/)
              .map(w => w.trim().toLowerCase())
              .filter(w => w.length > 4)
              .slice(0, 30) // cap to avoid perf issues with huge snippets
        : [];

    // ── Load PDF document once ────────────────────────────────────
    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        setError(null);

        pdfjsLib.getDocument({
            url: pdfUrl,
            withCredentials: false,
            // Byte-range requests: downloads only the bytes for the
            // requested page, not the whole PDF — dramatically faster
            rangeChunkSize: 65536,       // 64 KB chunks
            disableFontFace: true,       // skip font downloads
            isEvalSupported: false,
            enableXfa: false,
        })
            .promise
            .then(pdf => {
                if (cancelled) return;
                pdfRef.current = pdf;
                setTotalPages(pdf.numPages);
                setCurrentPage(Math.min(Math.max(1, pageNumber || 1), pdf.numPages));
            })
            .catch(err => {
                if (!cancelled) setError(`Failed to load PDF: ${err.message}`);
            });

        return () => { cancelled = true; };
    }, [pdfUrl, pageNumber]);

    // ── Render page whenever page / scale / pdf changes ──────────
    const renderPage = useCallback(async () => {
        if (!pdfRef.current) return;
        const canvas = canvasRef.current;
        if (!canvas) return;

        // Cancel any in-progress render to avoid "Cannot use the same canvas
        // during multiple render() operations" error
        if (renderTaskRef.current) {
            try { renderTaskRef.current.cancel(); } catch (_) { /* already done */ }
            renderTaskRef.current = null;
        }

        setLoading(true);
        setError(null);
        try {
            const page = await pdfRef.current.getPage(currentPage);
            const viewport = page.getViewport({ scale });

            // Set canvas dimensions
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            const ctx = canvas.getContext('2d');

            // White background
            ctx.fillStyle = '#fff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Render PDF page — track the task so we can cancel it later
            const task = page.render({ canvasContext: ctx, viewport });
            renderTaskRef.current = task;
            await task.promise;
            renderTaskRef.current = null;

            // ── Highlight matching words ──────────────────────────
            if (highlightWords.length > 0) {
                const textContent = await page.getTextContent();

                ctx.save();
                textContent.items.forEach(item => {
                    if (!item.str || !item.str.trim()) return;

                    const itemWords = item.str.toLowerCase().split(/\s+/);
                    const hasMatch = itemWords.some(w =>
                        highlightWords.some(hw => w.includes(hw) || hw.includes(w))
                    );
                    if (!hasMatch) return;

                    // Convert PDF transform to canvas coords
                    // PDF transform: [scaleX, skewX, skewY, scaleY, tx, ty]
                    const [, , , itemScale, tx, ty] = item.transform;
                    const absItemScale = Math.abs(itemScale);

                    // Apply page viewport transform
                    const [vA, vB, vC, vD, vE, vF] = viewport.transform;
                    const canvasX = vA * tx + vC * ty + vE;
                    const canvasY = vB * tx + vD * ty + vF;

                    const w = (item.width || 0) * scale;
                    const h = absItemScale * scale * 1.2;

                    ctx.fillStyle = HIGHLIGHT_COLOR;
                    ctx.fillRect(
                        canvasX,
                        canvasY - h * 0.85,
                        w,
                        h
                    );
                });
                ctx.restore();
            }
        } catch (err) {
            // Ignore cancellation errors (expected when navigating quickly)
            if (err?.name === 'RenderingCancelledException') return;
            setError(`Render error: ${err.message}`);
        } finally {
            setLoading(false);
        }
    }, [currentPage, scale, highlightWords]);

    useEffect(() => {
        renderPage();
    }, [renderPage]);

    // ── Keyboard navigation ───────────────────────────────────────
    useEffect(() => {
        const handler = (e) => {
            if (e.key === 'Escape') onClose();
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown')
                setCurrentPage(p => Math.min(p + 1, totalPages));
            if (e.key === 'ArrowLeft' || e.key === 'ArrowUp')
                setCurrentPage(p => Math.max(p - 1, 1));
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [onClose, totalPages]);

    const goTo = (p) => setCurrentPage(Math.min(Math.max(1, p), totalPages));

    return (
        <div className="pdf-modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
            <div className="pdf-modal">
                {/* ── Header ── */}
                <div className="pdf-modal-header">
                    <div className="pdf-modal-title">
                        <span className="pdf-modal-doc">
                            {pdfUrl.split('/').pop()?.replace('.pdf', '')}
                        </span>
                        <span className="pdf-modal-page-badge">
                            Page {currentPage} / {totalPages || '…'}
                        </span>
                    </div>

                    <div className="pdf-modal-controls">
                        {/* Zoom */}
                        <button
                            className="pdf-ctrl-btn"
                            onClick={() => setScale(s => Math.max(MIN_SCALE, +(s - 0.2).toFixed(1)))}
                            title="Zoom out"
                        >
                            <ZoomOut size={15} />
                        </button>
                        <span className="pdf-scale-label">{Math.round(scale * 100)}%</span>
                        <button
                            className="pdf-ctrl-btn"
                            onClick={() => setScale(s => Math.min(MAX_SCALE, +(s + 0.2).toFixed(1)))}
                            title="Zoom in"
                        >
                            <ZoomIn size={15} />
                        </button>

                        {/* Open in new tab */}
                        <a
                            href={`${pdfUrl}#page=${currentPage}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="pdf-ctrl-btn"
                            title="Open in new tab"
                        >
                            <ExternalLink size={14} />
                        </a>

                        {/* Close */}
                        <button className="pdf-ctrl-btn pdf-close-btn" onClick={onClose} title="Close">
                            <X size={16} />
                        </button>
                    </div>
                </div>

                {/* ── Canvas area ── */}
                <div className="pdf-modal-canvas-wrap" ref={containerRef}>
                    {loading && (
                        <div className="pdf-loading">
                            <div className="pdf-spinner" />
                            <span>Rendering page {currentPage}…</span>
                        </div>
                    )}
                    {error && (
                        <div className="pdf-error">{error}</div>
                    )}
                    <canvas
                        ref={canvasRef}
                        className="pdf-canvas"
                    />
                </div>

                {/* ── Footer navigation ── */}
                <div className="pdf-modal-footer">
                    <button
                        className="pdf-nav-btn"
                        onClick={() => goTo(currentPage - 1)}
                        disabled={currentPage <= 1}
                    >
                        <ChevronLeft size={16} /> Prev
                    </button>

                    <div className="pdf-page-input-wrap">
                        <input
                            type="number"
                            className="pdf-page-input"
                            value={currentPage}
                            min={1}
                            max={totalPages}
                            onChange={e => goTo(Number(e.target.value))}
                        />
                        <span className="pdf-page-total">/ {totalPages}</span>
                    </div>

                    <button
                        className="pdf-nav-btn"
                        onClick={() => goTo(currentPage + 1)}
                        disabled={currentPage >= totalPages}
                    >
                        Next <ChevronRight size={16} />
                    </button>
                </div>

                {highlightWords.length > 0 && (
                    <div className="pdf-highlight-note">
                        ✦ Highlighted words from the cited passage
                    </div>
                )}
            </div>
        </div>
    );
}
