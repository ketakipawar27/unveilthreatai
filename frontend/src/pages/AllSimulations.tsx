import { useState } from "react";
import FullReportModal from "../components/report/FullReportModal";

const AllSimulations = () => {
  const [isReportOpen, setIsReportOpen] = useState(false);

  const simulations = [
    { id: 1, name: "Simulation A", description: "Description of simulation A" },
    { id: 2, name: "Simulation B", description: "Description of simulation B" },
  ];

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
        All Simulations
      </h1>
      <p className="mb-6 text-gray-600 dark:text-gray-400">
        This page shows all available simulations.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {simulations.map((sim) => (
          <div
            key={sim.id}
            className="p-4 border rounded-lg shadow flex flex-col justify-between 
                       bg-white dark:bg-gray-900 
                       border-gray-200 dark:border-gray-700"
          >
            <div>
              <h2 className="font-semibold text-gray-900 dark:text-white">
                {sim.name}
              </h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {sim.description}
              </p>
            </div>
            <button
              className="mt-4 px-4 py-2 bg-indigo-500 text-white rounded-lg 
                         hover:bg-indigo-600 transition-colors"
              onClick={() => setIsReportOpen(true)}
            >
              View Report
            </button>
          </div>
        ))}
      </div>

      {/* Show modal when report is open */}
      {isReportOpen && <FullReportModal onClose={() => setIsReportOpen(false)} />}
    </div>
  );
};

export default AllSimulations;
