import Chart from "react-apexcharts";
import { ApexOptions } from "apexcharts";
import { Dropdown } from "../ui/dropdown/Dropdown";
import { DropdownItem } from "../ui/dropdown/DropdownItem";
import { MoreDotIcon } from "../../icons";
import { useState } from "react";

export default function RippleGraph() {
  const options: ApexOptions = {
    colors: ["#465FFF", "#7B8CFF", "#AEB8FF"],
    chart: { fontFamily: "Outfit, sans-serif", type: "radialBar", sparkline: { enabled: true } },
    plotOptions: {
      radialBar: {
        hollow: { size: "35%" },
        track: { background: "#E4E7EC", strokeWidth: "100%", margin: 8 },
        dataLabels: {
          name: { show: false },
          value: {
            show: true,
            fontSize: "20px",
            fontWeight: "600",
            offsetY: -15,
            color: "#1D2939",
            formatter: (val) => `${val}%`,
          },
        },
      },
    },
    legend: {
      show: true,
      floating: true,
      fontSize: "12px",
      position: "bottom",
      labels: { colors: "#6B7280" },
    },
    stroke: { lineCap: "round" },
    responsive: [
      {
        breakpoint: 1024,
        options: {
          plotOptions: { radialBar: { dataLabels: { value: { fontSize: "18px" } } } },
          legend: { fontSize: "11px" },
        },
      },
      {
        breakpoint: 640,
        options: {
          plotOptions: {
            radialBar: { hollow: { size: "30%" }, dataLabels: { value: { fontSize: "14px" } } },
          },
        },
      },
    ],
  };

  const series = [75, 55, 35];
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white px-4 pt-4 sm:px-6 sm:pt-6 
                    dark:border-gray-800 dark:bg-white/[0.03]">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-base sm:text-lg font-semibold text-gray-800 dark:text-white/90">
          Ripple Graph
        </h3>
        <div className="relative inline-block">
          <button className="dropdown-toggle" onClick={() => setIsOpen(!isOpen)}>
            <MoreDotIcon className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 size-5 sm:size-6" />
          </button>
          <Dropdown isOpen={isOpen} onClose={() => setIsOpen(false)} className="w-36 sm:w-40 p-2">
            <DropdownItem onItemClick={() => setIsOpen(false)}>View More</DropdownItem>
            <DropdownItem onItemClick={() => setIsOpen(false)}>Delete</DropdownItem>
          </Dropdown>
        </div>
      </div>

      <div className="flex justify-center items-center w-full h-[250px] sm:h-[400px] lg:h-[500px]">
        <Chart options={options} series={series} type="radialBar" height="100%" width="100%" />
      </div>
    </div>
  );
}
