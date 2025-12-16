import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import RippleGraph from "../../components/report/RippleGraph";
import RiskMeter from "../../components/report/RiskMeter";
import MetadataExposer from "../../components/report/MetadataExposer";
import PageMeta from "../../components/common/PageMeta";
import UploadCard from "../../components/common/UploadCard";
import ViewFullReport from "../../components/report/ViewFullReport";
import FullReportModal from "../../components/report/FullReportModal";
import FileInfoCard from "../../components/report/FileInfoCard";

export default function Home() {
  const [fileUploaded, setFileUploaded] = useState(false);
  const [fileName, setFileName] = useState("");
  const [fileType, setFileType] = useState("");
  const [showReport, setShowReport] = useState(false);

  return (
    <>
      <PageMeta
        title="Security Dashboard"
        description="Responsive Security Dashboard with graphs and risk metrics"
      />

      <div className="grid grid-cols-12 gap-4 md:gap-6">
        <AnimatePresence mode="wait">
          {!fileUploaded ? (
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
                  setFileUploaded(true);
                  setFileName(file.name);
                  setFileType(file.type);
                }}
              />
            </motion.div>
          ) : (
            <>
              {/* Left - File Info + Ripple Graph */}
              <motion.div
                key="left"
                initial={{ opacity: 0, x: -40 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, ease: "easeOut" }}
                className="col-span-12 lg:col-span-7 space-y-6"
              >
                <FileInfoCard fileName={fileName} fileType={fileType} />
                <RippleGraph />
              </motion.div>

              {/* Right - stacked cards */}
              <motion.div
                key="right"
                initial={{ opacity: 0, x: 40 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
                className="col-span-12 lg:col-span-5 space-y-6"
              >
                <RiskMeter />
                <MetadataExposer />
                <ViewFullReport onOpenReport={() => setShowReport(true)} />
              </motion.div>
            </>
          )}
        </AnimatePresence>
      </div>

      {/* Full Report Modal */}
      <AnimatePresence>
        {showReport && (
          <motion.div
            key="modal"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
          >
            <FullReportModal onClose={() => setShowReport(false)} />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
