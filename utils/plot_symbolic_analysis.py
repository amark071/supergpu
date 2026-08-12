#!/usr/bin/env python3
"""Plot a CHOLMOD symbolic factor and its elimination trees from CSV."""

import argparse
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_integer_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return [
            {key: int(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def compute_depths(parents):
    depths = [-1] * len(parents)
    for start in range(len(parents)):
        if depths[start] >= 0:
            continue
        path = []
        node = start
        seen = set()
        while node >= 0 and depths[node] < 0:
            if node in seen:
                raise ValueError("the exported parent array contains a cycle")
            seen.add(node)
            path.append(node)
            node = parents[node]
        depth = -1 if node < 0 else depths[node]
        while path:
            current = path.pop()
            depth += 1
            depths[current] = depth
    return depths


def save_symbolic_factor(supernodes, row_records, output_dir):
    rows_by_supernode = [[] for _ in supernodes]
    for record in row_records:
        rows_by_supernode[record["supernode"]].append(record["row"])

    even_x, even_y, odd_x, odd_y = [], [], [], []
    for supernode in supernodes:
        sid = supernode["supernode"]
        first_col = supernode["first_col"]
        end_col = supernode["end_col"]
        target_x, target_y = (even_x, even_y) if sid % 2 == 0 else (odd_x, odd_y)
        for col in range(first_col, end_col):
            for row in rows_by_supernode[sid]:
                if row >= col:
                    target_x.append(col)
                    target_y.append(row)

    n = supernodes[-1]["end_col"]
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    marker_size = max(0.08, min(1.2, 700.0 / max(n, 1)))
    ax.scatter(even_x, even_y, s=marker_size, c="#224f78", marker="s",
               linewidths=0, rasterized=True, label="Even supernode")
    ax.scatter(odd_x, odd_y, s=marker_size, c="#d47a28", marker="s",
               linewidths=0, rasterized=True, label="Odd supernode")
    for supernode in supernodes:
        boundary = supernode["first_col"] - 0.5
        ax.axvline(boundary, color="#9b2c2c", linewidth=0.25, alpha=0.35)
    ax.axvline(n - 0.5, color="#9b2c2c", linewidth=0.25, alpha=0.35)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title("Symbolic factor L with basic-supernode boundaries")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.legend(loc="upper right", markerscale=8, frameon=False)
    fig.savefig(os.path.join(output_dir, "symbolic_factor_supernodes.svg"), dpi=180)
    plt.close(fig)


def save_column_tree(columns, output_dir):
    parents = [record["parent"] for record in columns]
    depths = compute_depths(parents)
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for child, parent in enumerate(parents):
        if parent >= 0:
            ax.plot([child, parent], [depths[child], depths[parent]],
                    color="#7b8794", linewidth=0.35, alpha=0.55)
    colors = ["#224f78" if record["supernode"] % 2 == 0 else "#d47a28"
              for record in columns]
    ax.scatter(range(len(columns)), depths, s=5, c=colors, linewidths=0)
    ax.set_title("Column elimination tree (x preserves ordered-column index)")
    ax.set_xlabel("Ordered column")
    ax.set_ylabel("Depth from root")
    ax.invert_yaxis()
    ax.grid(axis="y", linewidth=0.3, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "elimination_tree.svg"))
    plt.close(fig)


def save_supernode_tree(supernodes, output_dir):
    parents = [record["parent"] for record in supernodes]
    depths = compute_depths(parents)
    centers = [(record["first_col"] + record["end_col"] - 1) / 2.0
               for record in supernodes]
    work = [max(1, record["width"] * record["row_count"])
            for record in supernodes]
    max_work = max(work)
    sizes = [8.0 + 90.0 * math.sqrt(value / max_work) for value in work]

    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for child, parent in enumerate(parents):
        if parent >= 0:
            ax.plot([centers[child], centers[parent]],
                    [depths[child], depths[parent]],
                    color="#7b8794", linewidth=0.5, alpha=0.6)
    points = ax.scatter(centers, depths, s=sizes, c=work, cmap="viridis",
                        linewidths=0.2, edgecolors="#263238")
    largest = sorted(range(len(supernodes)), key=lambda sid: work[sid], reverse=True)[:15]
    for sid in largest:
        record = supernodes[sid]
        ax.annotate(
            "S{} ({}x{})".format(sid, record["row_count"], record["width"]),
            (centers[sid], depths[sid]), xytext=(3, 3),
            textcoords="offset points", fontsize=7)
    colorbar = fig.colorbar(points, ax=ax, pad=0.01)
    colorbar.set_label("Stored block entries (row count x width)")
    ax.set_title("Compressed supernode elimination tree")
    ax.set_xlabel("Ordered-column position")
    ax.set_ylabel("Depth from root")
    ax.invert_yaxis()
    ax.grid(axis="y", linewidth=0.3, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "supernode_tree.svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or input_dir)
    os.makedirs(output_dir, exist_ok=True)

    columns = read_integer_rows(os.path.join(input_dir, "column_tree.csv"))
    supernodes = read_integer_rows(os.path.join(input_dir, "supernodes.csv"))
    rows = read_integer_rows(os.path.join(input_dir, "supernode_rows.csv"))
    if not columns or not supernodes:
        raise ValueError("symbolic-analysis CSV files are empty")

    save_symbolic_factor(supernodes, rows, output_dir)
    save_column_tree(columns, output_dir)
    save_supernode_tree(supernodes, output_dir)
    print("Symbolic-analysis figures written to {}".format(output_dir))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("visualization failed: {}".format(error), file=sys.stderr)
        raise
