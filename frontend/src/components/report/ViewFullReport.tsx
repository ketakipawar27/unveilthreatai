import { FileText } from "lucide-react";

interface ViewFullReportProps {
  onOpenReport: () => void;
}

export default function ViewFullReport({ onOpenReport }: ViewFullReportProps) {
  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm 
                    dark:border-gray-800 dark:bg-white/[0.03] flex flex-col items-center justify-center">
      <FileText className="w-10 h-10 text-indigo-500 mb-3" />
      <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90">
        View Full Report
      </h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4 text-center">
        Open the detailed security analysis report with full insights.
      </p>
      <button
        className="px-4 py-2 bg-indigo-500 text-white text-sm font-medium rounded-lg hover:bg-indigo-600"
        onClick={onOpenReport}
      >
        Open Report
      </button>
    </div>
  );
}
