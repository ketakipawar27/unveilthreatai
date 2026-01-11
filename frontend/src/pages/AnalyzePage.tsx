import { useState } from "react";
import UploadCard from "../components/common/UploadCard";
import ConsequenceGraph from "../components/report/ConsequenceGraph";

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [caption, setCaption] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    if (caption.trim()) {
      formData.append("caption", caption);
    }

    try {
      const res = await fetch("http://127.0.0.1:8000/api/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      // 🔍 KEEP LOGS (VERY IMPORTANT FOR DEBUG)
      console.log("FULL BACKEND RESPONSE:", data);
      console.log("CONSEQUENCE GRAPH ONLY:", data.consequence_graph);

      if (!res.ok) {
        throw new Error(data.message || "Analysis failed");
      }

      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">

      {/* Upload */}
      <UploadCard onUpload={(f) => setFile(f)} />

      {/* Caption */}
      <textarea
        className="w-full rounded-lg border p-3"
        placeholder="Optional caption..."
        value={caption}
        onChange={(e) => setCaption(e.target.value)}
      />

      {/* Analyze Button */}
      <button
        onClick={handleAnalyze}
        disabled={!file || loading}
        className="px-6 py-2 rounded-lg bg-indigo-600 text-white disabled:opacity-50"
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {/* Error */}
      {error && (
        <p className="text-red-500 text-sm">{error}</p>
      )}

      {/* ✅ CONSEQUENCE GRAPH */}
      {result?.consequence_graph && (
        <ConsequenceGraph graph={result.consequence_graph} />
      )}

      {/* 🧪 OPTIONAL DEBUG (KEEP COMMENTED) */}
      {/*
      {result && (
        <pre className="bg-black text-green-400 p-4 rounded-lg overflow-auto max-h-[400px]">
          {JSON.stringify(result.consequence_graph, null, 2)}
        </pre>
      )}
      */}

    </div>
  );
}
