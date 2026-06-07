from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import networkx as nx

TYPE_COLORS = {"raster": "#4caf50", "vector": "#2196f3", "table": "#ff9800", "report": "#9c27b0", "json": "#607d8b"}
TYPE_SHAPES = {"raster": "s", "vector": "d", "table": "o", "report": "p", "json": "h"}


def load_artifact_graph(metadata_path: Path) -> dict:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_nx_graph(metadata: dict) -> nx.DiGraph:
    artifacts = metadata["artifacts"]
    graph = nx.DiGraph()
    for artifact_id, artifact in artifacts.items():
        graph.add_node(artifact_id, label=artifact_id, atype=artifact["type"])
        for parent in artifact.get("parents", []):
            graph.add_edge(parent, artifact_id)
    return graph


def draw_artifact_graph(graph: nx.DiGraph, output_path: Path) -> None:
    pos = _layout(graph)
    node_colors = [TYPE_COLORS.get(graph.nodes[n]["atype"], "#999") for n in graph.nodes]
    labels = {n: f"{n}\n({graph.nodes[n]['atype']})" for n in graph.nodes}

    fig, ax = plt.subplots(figsize=(10, 4))
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#888", arrows=True, arrowsize=14, arrowstyle="-|>")
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=2200, node_shape="s", edgecolors="#333", linewidths=0.8)
    nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=8, font_weight="bold")
    ax.set_title("GeoHarness Artifact Graph", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_mermaid_graph(graph: nx.DiGraph, output_path: Path) -> None:
    lines = ["```mermaid", "graph TD"]
    safe = lambda s: s.replace(" ", "_")
    for node in graph.nodes:
        atype = graph.nodes[node]["atype"]
        lines.append(f'  {safe(node)}["{node}<br/>({atype})"]')
    for u, v in graph.edges:
        lines.append(f"  {safe(u)} --> {safe(v)}")
    lines.append("```")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _layout(graph: nx.DiGraph) -> dict:
    roots = [n for n, d in graph.in_degree() if d == 0]
    if len(roots) == 1:
        return nx.spring_layout(graph, seed=42, k=1.5, iterations=100)
    # multi-root: place left and right roots then spring-relax
    pos = {}
    for idx, root in enumerate(roots):
        pos[root] = (-1.5 + idx * 3.0, 0)
    pos = nx.spring_layout(graph, pos=pos, fixed=roots, seed=42, k=1.2, iterations=80)
    return pos


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the artifact graph from metadata.json.")
    parser.add_argument("--metadata", default="runs/measure_mvp/store/metadata.json", help="Path to metadata.json")
    parser.add_argument("--output-dir", default=None, help="Output directory (defaults to next to metadata)")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir) if args.output_dir else metadata_path.parent.parent

    metadata = load_artifact_graph(metadata_path)
    graph = build_nx_graph(metadata)
    draw_artifact_graph(graph, output_dir / "artifact_graph.png")
    write_mermaid_graph(graph, output_dir / "artifact_graph.md")
    print(f"wrote {output_dir / 'artifact_graph.png'}")
    print(f"wrote {output_dir / 'artifact_graph.md'}")


if __name__ == "__main__":
    main()
