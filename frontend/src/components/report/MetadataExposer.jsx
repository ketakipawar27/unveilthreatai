import { useState } from "react";

export default function MetadataExposer() {
  // Example percentage (you can make this dynamic later from API / props)
  const [exposure, setExposure] = useState(62);

  return (
    <div className="mt-6 overflow-hidden rounded-2xl border border-gray-200 bg-white p-6 
                    dark:border-gray-800 dark:bg-white/[0.03]">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90">
          Metadata Exposer
        </h3>
        <span className="text-xl font-bold text-indigo-600 dark:text-indigo-400">
          {exposure}%
        </span>
      </div>
      <div className="mt-4 w-full bg-gray-200 rounded-full h-3 dark:bg-gray-700">
        <div
          className="bg-indigo-500 h-3 rounded-full transition-all"
          style={{ width: `${exposure}%` }}
        ></div>
      </div>
      <p className="mt-3 text-sm text-gray-500 dark:text-gray-400">
        Indicates how much sensitive metadata is being exposed.
      </p>
    </div>
  );
}
