import yaml
import numpy as np
from pathlib import Path
from jinja2 import Template
from polis_core import OpinionAnalyzer


def load_from_yaml(filepath):
    with open(filepath) as f:
        data = yaml.safe_load(f)
    vote_map = {"agree": 1, "disagree": -1}
    votes = [
        [vote_map.get(v, 0) for v in user_votes]
        for user_votes in data["votes"].values()
    ]
    return data["statements"], np.array(votes)


def generate_report(template_path, analysis_results):
    template = Template(Path(template_path).read_text())
    return template.render(
        consensus_data=analysis_results["consensus_data"],
        divisive_data=analysis_results["divisive_data"],
        group_data=analysis_results["group_data"],
    )


def main(yaml_file):
    # Load and prepare data
    statements, votes = load_from_yaml(yaml_file)

    # Analyze opinions
    analyzer = OpinionAnalyzer()
    results = analyzer.analyze(votes, statements)

    # Generate and print report
    template_path = Path(__file__).parent / "report_template.j2"
    report = generate_report(template_path, results)
    print(report)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python polislite.py input.yaml")
        sys.exit(1)
    main(sys.argv[1])
