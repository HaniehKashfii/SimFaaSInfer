# simfaasinfer/reports/report_generator.py
"""
Generate JSON/HTML/PDF reports for search results.
"""
from typing import Dict, Any
import json
import os
from pathlib import Path


def generate_report(search_result: Dict[str, Any], out_dir: str, format: str = 'json') -> str:
    """
    Write a JSON report and simple HTML summary and return path to JSON file.
    
    Args:
        search_result: Result from vidur_search
        out_dir: Output directory
        format: Output format ('json', 'html', or 'both')
    
    Returns:
        Path to main report file
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate JSON report
    json_path = os.path.join(out_dir, "search_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(search_result, f, indent=2)
    
    print(f"Generated JSON report: {json_path}")
    
    # Generate HTML report if requested
    if format in ['html', 'both']:
        html_path = os.path.join(out_dir, "search_report.html")
        _generate_html_report(search_result, html_path)
        print(f"Generated HTML report: {html_path}")
    
    # Generate Pareto plot if we have matplotlib
    try:
        import matplotlib.pyplot as plt
        plot_path = os.path.join(out_dir, "pareto_frontier.png")
        _generate_pareto_plot(search_result, plot_path)
        print(f"Generated Pareto plot: {plot_path}")
    except ImportError:
        pass
    
    return json_path


def _generate_html_report(search_result: Dict[str, Any], output_path: str):
    """Generate HTML report."""
    best_config = search_result.get('best_config', {})
    pareto = search_result.get('pareto', {})
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SimFaaSInfer Search Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #0066cc; }}
        .config {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>SimFaaSInfer Capacity Planning Report</h1>
    
    <h2>Best Configuration</h2>
    <div class="config">
        <p><strong>GPU Type:</strong> {best_config.get('config', {}).get('gpu_type', 'N/A')}</p>
        <p><strong>GPUs per Replica:</strong> {best_config.get('config', {}).get('num_gpus_per_replica', 'N/A')}</p>
        <p><strong>TP Size:</strong> {best_config.get('config', {}).get('tp', 'N/A')}</p>
        <p><strong>Replicas:</strong> {best_config.get('config', {}).get('num_replicas', 'N/A')}</p>
        <p class="metric">Max QPS: {best_config.get('max_qps', 0):.2f}</p>
        <p class="metric">Cost: ${best_config.get('cost_per_hour', 0):.2f}/hour</p>
        <p class="metric">QPS per Dollar: {best_config.get('qps_per_dollar', 0):.4f}</p>
    </div>
    
    <h2>Pareto Frontier</h2>
    <table>
        <tr>
            <th>Configuration</th>
            <th>Max QPS</th>
            <th>Cost ($/hr)</th>
            <th>QPS/$</th>
        </tr>
"""
    
    for point in pareto.get('points', [])[:10]:  # Top 10
        config = point['config']
        html += f"""
        <tr>
            <td>{config.get('gpu_type')} x{config.get('total_gpus')} (TP={config.get('tp')})</td>
            <td>{point['max_qps']:.2f}</td>
            <td>${point['cost_per_hour']:.2f}</td>
            <td>{point['qps_per_dollar']:.4f}</td>
        </tr>
"""
    
    html += """
    </table>
    
    <h2>Summary</h2>
    <p>Total configurations evaluated: {}</p>
    <p>Pareto-optimal configurations: {}</p>
    
</body>
</html>
""".format(
        search_result.get('raw', {}).get('num_configs_evaluated', 0),
        pareto.get('count', 0)
    )
    
    with open(output_path, 'w') as f:
        f.write(html)


def _generate_pareto_plot(search_result: Dict[str, Any], output_path: str):
    """Generate Pareto frontier plot."""
    import matplotlib.pyplot as plt
    
    candidates = search_result.get('candidates', [])
    pareto_points = search_result.get('pareto', {}).get('points', [])
    
    if not candidates:
        return
    
    # Extract data
    all_costs = [c['cost_per_hour'] for c in candidates]
    all_qps = [c['max_qps'] for c in candidates]
    
    pareto_costs = [p['cost_per_hour'] for p in pareto_points]
    pareto_qps = [p['max_qps'] for p in pareto_points]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot all candidates
    plt.scatter(all_costs, all_qps, alpha=0.5, s=50, label='All Configurations')
    
    # Plot Pareto frontier
    plt.scatter(pareto_costs, pareto_qps, c='red', s=100, marker='*', 
                label='Pareto Frontier', zorder=5)
    
    # Connect Pareto points
    if len(pareto_costs) > 1:
        sorted_pareto = sorted(zip(pareto_costs, pareto_qps))
        plt.plot([p[0] for p in sorted_pareto], [p[1] for p in sorted_pareto], 
                'r--', alpha=0.5, linewidth=2)
    
    plt.xlabel('Cost ($/hour)', fontsize=12)
    plt.ylabel('Max QPS', fontsize=12)
    plt.title('Cost vs Performance - Pareto Frontier', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()