import { useState, useCallback } from 'react';

const API_BASE = '/api/vision';

export function useSessionReport(sessionId) {
  const [reportData, setReportData] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const downloadJSON = useCallback(async () => {
    if (!sessionId) return;
    setIsGenerating(true);
    try {
      const res = await fetch(`${API_BASE}/report/${sessionId}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setReportData(data);

      // Trigger browser download
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${sessionId}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download JSON report:', err);
    } finally {
      setIsGenerating(false);
    }
  }, [sessionId]);

  const downloadPDF = useCallback(async () => {
    if (!sessionId) return;
    setIsGenerating(true);
    try {
      const res = await fetch(`${API_BASE}/report/${sessionId}/pdf`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${sessionId}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download PDF report:', err);
    } finally {
      setIsGenerating(false);
    }
  }, [sessionId]);

  const markQuestion = useCallback(async (label) => {
    if (!sessionId || !label) return;
    try {
      await fetch(`${API_BASE}/session/${sessionId}/question`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label }),
      });
    } catch (err) {
      console.error('Failed to mark question:', err);
    }
  }, [sessionId]);

  return {
    reportData,
    isGenerating,
    downloadJSON,
    downloadPDF,
    markQuestion,
  };
}
