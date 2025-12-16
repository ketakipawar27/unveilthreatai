import Chart from "react-apexcharts";
import { ApexOptions } from "apexcharts";
import { useState } from "react";
import { Dropdown } from "../ui/dropdown/Dropdown";
import { DropdownItem } from "../ui/dropdown/DropdownItem";
import { MoreDotIcon } from "../../icons";

export default function RiskMeter() {
  const riskScore = 68; // Example: risk % (0–100)

  const options: ApexOptions = {
    colors: [riskScore > 75 ? "#D92D20" : riskScore > 40 ? "#F79009" : "#039855"],
    chart: {
      fontFamily: "Outfit, sans-serif",
      type: "radialBar",
      height: 330,
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      radialBar: {
        startAngle: -90,
        endAngle: 90,
        hollow: {
          size: "75%",
        },
        track: {
          background: "#E4E7EC",
          strokeWidth: "100%",
          margin: 5,
        },
        dataLabels: {
          name: {
            show: false,
          },
          value: {
            fontSize: "32px",
            fontWeight: "600",
            offsetY: -35, // moved higher
            color: "#1D2939",
            formatter: (val) => val + "%",
          },
        },

      },
    },
    fill: {
      type: "solid",
      colors: [riskScore > 75 ? "#D92D20" : riskScore > 40 ? "#F79009" : "#039855"],
    },
    stroke: {
      lineCap: "round",
    },
    labels: ["Risk Score"],
  };

  const [isOpen, setIsOpen] = useState(false);

  function toggleDropdown() {
    setIsOpen(!isOpen);
  }
  function closeDropdown() {
    setIsOpen(false);
  }

  const riskLevel =
    riskScore > 75 ? "High Risk" : riskScore > 40 ? "Moderate Risk" : "Low Risk";
  const riskColor =
    riskScore > 75 ? "text-red-600 bg-red-100 dark:bg-red-500/15 dark:text-red-400" :
    riskScore > 40 ? "text-yellow-600 bg-yellow-100 dark:bg-yellow-500/15 dark:text-yellow-400" :
    "text-green-600 bg-green-100 dark:bg-green-500/15 dark:text-green-400";

  return (
    <div className="rounded-2xl border border-gray-200 bg-gray-100 dark:border-gray-800 dark:bg-white/[0.03]">
      <div className="px-5 pt-5 bg-white shadow-default rounded-2xl pb-11 dark:bg-gray-900 sm:px-6 sm:pt-6">
        <div className="flex justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90">
              Risk Meter
            </h3>
            <p className="mt-1 text-gray-500 text-theme-sm dark:text-gray-400">
              Current risk score based on analysis
            </p>
          </div>
          <div className="relative inline-block">
            <button className="dropdown-toggle" onClick={toggleDropdown}>
              <MoreDotIcon className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 size-6" />
            </button>
            <Dropdown
              isOpen={isOpen}
              onClose={closeDropdown}
              className="w-40 p-2"
            >
              <DropdownItem
                onItemClick={closeDropdown}
                className="flex w-full font-normal text-left text-gray-500 rounded-lg hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-white/5 dark:hover:text-gray-300"
              >
                View Details
              </DropdownItem>
              <DropdownItem
                onItemClick={closeDropdown}
                className="flex w-full font-normal text-left text-gray-500 rounded-lg hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-white/5 dark:hover:text-gray-300"
              >
                Export Report
              </DropdownItem>
            </Dropdown>
          </div>
        </div>

        <div className="relative">
          <div className="max-h-[330px]">
            <Chart
              options={options}
              series={[riskScore]}
              type="radialBar"
              height={330}
            />
          </div>

          <span
            className={`absolute left-1/2 top-full -translate-x-1/2 -translate-y-[95%] rounded-full px-3 py-1 text-xs font-medium ${riskColor}`}
          >
            {riskLevel}
          </span>
        </div>

        <p className="mx-auto mt-10 w-full max-w-[380px] text-center text-sm text-gray-500 sm:text-base">
          Risk is currently at <span className="font-semibold">{riskScore}%</span>.  
          Recommended to {riskScore > 75 ? "mitigate immediately" : riskScore > 40 ? "monitor closely" : "maintain current defenses"}.
        </p>
      </div>
    </div>
  );
}
