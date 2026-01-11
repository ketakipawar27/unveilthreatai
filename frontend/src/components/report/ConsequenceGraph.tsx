import { useEffect, useRef } from "react";
import * as d3 from "d3";

type NodeType = {
  id: string;
  label: string;
  type: "root" | "signal" | "risk" | "consequence";
  level: number;
};

type EdgeType = {
  from: string;
  to: string;
};

type Props = {
  graph: {
    nodes: NodeType[];
    edges: EdgeType[];
  };
};

export default function ConsequenceGraph({ graph }: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!graph?.nodes?.length || !svgRef.current) return;

    const width = 1000;
    const height = 520;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // ---------------------------------------
    // Layout config
    // ---------------------------------------
    const levelX: Record<number, number> = {
      0: width - 120, // root (right)
      1: width - 350, // signals
      2: width - 600, // risks
      3: width - 850, // consequences (left)
    };

    const nodesByLevel = d3.group(graph.nodes, (d) => d.level);

    // Assign fixed positions
    const positionedNodes = graph.nodes.map((node) => {
      const siblings = nodesByLevel.get(node.level) || [];
      const index = siblings.findIndex((n) => n.id === node.id);

      return {
        ...node,
        x: levelX[node.level],
        y:
          height / 2 -
          (siblings.length - 1) * 40 +
          index * 80,
      };
    });

    const nodeMap = new Map(positionedNodes.map((n) => [n.id, n]));

    const links = graph.edges
      .map((e) => ({
        source: nodeMap.get(e.from),
        target: nodeMap.get(e.to),
      }))
      .filter((l) => l.source && l.target);

    // ---------------------------------------
    // Styling
    // ---------------------------------------
    const colorMap: Record<string, string> = {
      root: "#465FFF",
      signal: "#F59E0B",
      risk: "#EF4444",
      consequence: "#7C3AED",
    };

    // ---------------------------------------
    // Links
    // ---------------------------------------
    svg
      .append("g")
      .attr("stroke", "#CBD5E1")
      .attr("stroke-width", 2)
      .selectAll("path")
      .data(links)
      .enter()
      .append("path")
      .attr("fill", "none")
      .attr("d", (d: any) => {
        const midX = (d.source.x + d.target.x) / 2;
        return `
          M ${d.source.x},${d.source.y}
          C ${midX},${d.source.y}
            ${midX},${d.target.y}
            ${d.target.x},${d.target.y}
        `;
      });

    // ---------------------------------------
    // Nodes
    // ---------------------------------------
    const node = svg
      .append("g")
      .selectAll("g")
      .data(positionedNodes)
      .enter()
      .append("g")
      .attr("transform", (d) => `translate(${d.x},${d.y})`);

    node
      .append("circle")
      .attr("r", (d) => (d.level === 0 ? 22 : 14))
      .attr("fill", (d) => colorMap[d.type]);

    node
      .append("text")
      .text((d) => d.label)
      .attr("x", (d) => (d.level === 0 ? 30 : -30))
      .attr("text-anchor", (d) => (d.level === 0 ? "start" : "end"))
      .attr("alignment-baseline", "middle")
      .attr("font-size", "12px")
      .attr("fill", "#111827");

  }, [graph]);

  return (
    <div className="rounded-2xl border bg-white p-4 dark:bg-white/[0.03]">
      <h3 className="mb-3 text-lg font-semibold">
        Consequence Chain (Right → Left)
      </h3>

      <svg
        ref={svgRef}
        width="100%"
        height="520"
        viewBox="0 0 1000 520"
      />
    </div>
  );
}
