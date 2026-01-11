import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

import ConsequenceGraph from "../../components/report/ConsequenceGraph";
import RiskMeter from "../../components/report/RiskMeter";
import MetadataExposer from "../../components/report/MetadataExposer";
import PageMeta from "../../components/common/PageMeta";
import UploadCard from "../../components/common/UploadCard";
import ViewFullReport from "../../components/report/ViewFullReport";
import FullReportModal from "../../components/report/FullReportModal";
import FileInfoCard from "../../components/report/FileInfoCard";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showReport, setShowReport] = useState(false);

  // 🔥 THIS IS THE BACKEND CALL
  const analyzeFile = async (file: File) => {
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      console.log("BACKEND RESPONSE:", data);

      if (!res.ok) {
        throw new Error(data.message || "Analysis failed");
      }

      setAnalysis(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <PageMeta
        title="Security Dashboard"
        description="Responsive Security Dashboard with graphs and risk metrics"
      />

      <div className="grid grid-cols-12 gap-4 md:gap-6">

        <AnimatePresence mode="wait">
          {!analysis ? (
            /* ================= UPLOAD ================= */
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -30 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              className="col-span-12"
            >
              <UploadCard
                onUpload={(file: File) => {
                  setFile(file);
                  analyzeFile(file); // 🔥 START ANALYSIS HERE
                }}
              />

              {loading && (
                <p className="mt-4 text-sm text-gray-500">
                  Analyzing file, please wait…
                </p>
              )}

              {error && (
                <p className="mt-4 text-sm text-red-500">
                  {error}
                </p>
              )}
            </motion.div>
          ) : (
            <>
              {/* ================= LEFT ================= */}
              <motion.div
                key="left"
                initial={{ opacity: 0, x: -40 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6 }}
                className="col-span-12 lg:col-span-7 space-y-6"
              >
                <FileInfoCard
                  fileName={file?.name || ""}
                  fileType={file?.type || ""}
                />

                <ConsequenceGraph graph={analysis.consequence_graph} />
              </motion.div>

              {/* ================= RIGHT ================= */}
              <motion.div
                key="right"
                initial={{ opacity: 0, x: 40 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.15 }}
                className="col-span-12 lg:col-span-5 space-y-6"
              >
                <RiskMeter risk={analysis.ai_output?.risk_assessment} />
                <MetadataExposer aiOutput={analysis.ai_output} />
                <ViewFullReport onOpenReport={() => setShowReport(true)} />
              </motion.div>
            </>
          )}
        </AnimatePresence>
      </div>

      {/* ================= FULL REPORT MODAL ================= */}
      <AnimatePresence>
        {showReport && analysis && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
          >
            <FullReportModal
              data={analysis}
              onClose={() => setShowReport(false)}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
