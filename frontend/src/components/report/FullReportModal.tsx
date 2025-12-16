import { X } from "lucide-react";

interface FullReportModalProps {
  onClose: () => void;
}

export default function FullReportModal({ onClose }: FullReportModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="relative w-full max-w-4xl bg-gray-900 text-white rounded-2xl shadow-lg p-6 overflow-y-auto max-h-[90vh]">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-white"
        >
          <X size={24} />
        </button>

        {/* Report Title */}
        <h2 className="text-2xl font-bold mb-4">Full Security Report</h2>
        <p className="text-gray-400 mb-6">
          This is the detailed analysis generated from the uploaded file. Review
          all insights, risks, and metadata exposures below.
        </p>

        {/* Example Sections */}
        <div className="space-y-6">
          {/* Risk Overview */}
          <section className="p-4 bg-gray-800 rounded-xl">
            <h3 className="text-xl font-semibold mb-2">📊 Risk Overview</h3>
            <p>
              Overall risk level detected: <span className="font-bold text-orange-400">Moderate (68%)</span>.
              It is recommended to monitor closely and take preventive actions.
            </p>
          </section>

          {/* Metadata Exposer */}
          <section className="p-4 bg-gray-800 rounded-xl">
            <h3 className="text-xl font-semibold mb-2">🗂 Metadata Exposer</h3>
            <p>
              Sensitive metadata exposure: <span className="font-bold text-blue-400">62%</span>.
              Review file headers and embedded metadata to reduce leakage risk.
            </p>
          </section>

          {/* Recommendations */}
          <section className="p-4 bg-gray-800 rounded-xl">
            <h3 className="text-xl font-semibold mb-2">✅ Recommendations</h3>
            <ul className="list-disc pl-6 space-y-1 text-gray-300">
              <li>Encrypt sensitive files before sharing externally.</li>
              <li>Remove unnecessary metadata using sanitization tools.</li>
              <li>Regularly monitor files for anomalies.</li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}
