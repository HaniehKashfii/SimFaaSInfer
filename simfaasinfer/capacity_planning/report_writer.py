"""
Report writer for capacity planning - generates human-readable summaries.
"""
from typing import Dict, Any


class ReportWriter:
    """Generates human-readable capacity planning reports."""

    def generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary from results."""
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("   CAPACITY PLANNING REPORT")
        model_name = results.get('model_name', 'Unknown Model')
        hw_info = results.get('profiling_hardware', {})
        gpu_name = hw_info.get('gpu', 'Unknown GPU')
        lines.append(f"   Model: {model_name} on {gpu_name}")
        timestamp = results.get('timestamp', 'N/A')
        lines.append(f"   Date: {timestamp}")
        lines.append("=" * 60)
        lines.append("")

        best_config = results.get('best_configuration')

        if not best_config:
            lines.append("NO CONFIGURATION FOUND MEETING CONSTRAINTS")
            lines.append("")
            lines.append("Suggestions:")
            lines.append("  - Relax SLO constraints")
            lines.append("  - Increase max cost budget")
            lines.append("  - Add more powerful GPUs to search space")
            return "\n".join(lines)

        # Recommended Configuration
        lines.append("RECOMMENDED CONFIGURATION")
        lines.append("━" * 60)
        lines.append("Hardware:")
        lines.append(f"  GPU Type:         {best_config['gpu_type']}")
        lines.append(f"  Total GPUs:       {best_config['total_gpus']} "
                    f"({best_config['num_replicas']} replicas × "
                    f"{best_config['num_gpus_per_replica']} GPUs each)")
        lines.append(f"  Parallelism:      TP={best_config['tensor_parallel']}, "
                    f"PP={best_config['pipeline_parallel']}")
        lines.append("")
        lines.append("Software:")
        lines.append(f"  Scheduler:        {best_config['scheduler']}")
        lines.append(f"  Max Batch Size:   {best_config['max_batch_size']}")
        lines.append("")

        # Performance Estimates
        perf = best_config['performance']
        lines.append("PERFORMANCE ESTIMATES")
        lines.append("━" * 60)
        lines.append(f"Throughput:         {perf['max_sustainable_qps']:.1f} QPS (requests/second)")
        lines.append(f"Latency (P95):      {perf['p95_latency_ms']:.1f} ms")
        lines.append(f"Latency (P99):      {perf['p99_latency_ms']:.1f} ms")

        # Add SLO check marks
        slo = best_config.get('slo_compliance', {})
        ttft_check = "✓" if slo.get('ttft_slo_met', True) else "✗"
        tbt_check = "✓" if slo.get('tbt_slo_met', True) else "✗"

        if perf.get('p95_ttft_ms', 0) > 0:
            lines.append(f"TTFT (P95):         {perf['p95_ttft_ms']:.1f} ms {ttft_check}")
        if perf.get('p99_tbt_ms', 0) > 0:
            lines.append(f"TBT (P99):          {perf['p99_tbt_ms']:.1f} ms {tbt_check}")

        lines.append(f"Resource Util:      {perf['mean_mfu']*100:.0f}% MFU, "
                    f"{perf['mean_memory_util']*100:.0f}% Memory")
        lines.append("")

        # Cost Analysis
        cost = best_config['cost']
        lines.append("COST ANALYSIS")
        lines.append("━" * 60)
        lines.append(f"Hourly Cost:        ${cost['cost_per_hour']:.2f}/hour")
        lines.append(f"Cost per 1K Req:    ${cost['cost_per_1k_requests']:.2f}")
        lines.append(f"QPS per Dollar:     {cost['qps_per_dollar']:.2f}")
        lines.append("")

        # SLO Compliance
        all_met = slo.get('meets_all_slos', True)
        status = "✓ ALL SLOS MET" if all_met else "✗ SOME SLOS NOT MET"
        lines.append(f"SLO COMPLIANCE:     {status}")
        lines.append("")

        # Alternative Configurations (Pareto Frontier)
        pareto = results.get('pareto', {}).get('points', [])
        if len(pareto) > 1:
            lines.append("ALTERNATIVE CONFIGURATIONS")
            lines.append("━" * 60)

            # Find budget and premium options
            pareto_sorted = sorted(pareto, key=lambda x: x['cost_per_hour'])

            if len(pareto_sorted) > 0:
                budget_opt = pareto_sorted[0]
                lines.append("Budget Option:")
                lines.append(f"  {budget_opt['gpu_type']} (x{budget_opt['config'].get('total_gpus', 'N/A')}): "
                           f"{budget_opt['max_qps']:.0f} QPS @ "
                           f"${budget_opt['cost_per_hour']:.2f}/hr "
                           f"({budget_opt['qps_per_dollar']:.2f} QPS/$)")

            if len(pareto_sorted) > 2:
                premium_opt = pareto_sorted[-1]
                if premium_opt['cost_per_hour'] > best_config['cost']['cost_per_hour']:
                    lines.append("")
                    lines.append("Premium Option:")
                    lines.append(f"  {premium_opt['gpu_type']} (x{premium_opt['config'].get('total_gpus', 'N/A')}): "
                               f"{premium_opt['max_qps']:.0f} QPS @ "
                               f"${premium_opt['cost_per_hour']:.2f}/hr "
                               f"({premium_opt['qps_per_dollar']:.2f} QPS/$)")

            lines.append("")

        # Recommendation
        lines.append("RECOMMENDATION")
        lines.append("━" * 60)
        lines.append(f"Deploy with {best_config['gpu_type']} "
                    f"(x{best_config['total_gpus']}) configuration using "
                    f"{best_config['scheduler']} scheduler.")
        lines.append(f"This provides optimal cost-efficiency "
                    f"({cost['qps_per_dollar']:.2f} QPS/$) while meeting")
        lines.append("all SLO requirements." if all_met else "most requirements.")

        # Add budget recommendation if applicable
        if len(pareto) > 1:
            budget_opt = sorted(pareto, key=lambda x: x['cost_per_hour'])[0]
            max_qps = perf['max_sustainable_qps']
            budget_qps = budget_opt['max_qps']
            if budget_qps < max_qps * 0.6:
                threshold = int(budget_qps * 0.9)
                lines.append("")
                lines.append(f"For lower traffic (<{threshold} QPS), consider "
                           f"{budget_opt['gpu_type']} "
                           f"(x{budget_opt['config'].get('total_gpus', 'N/A')}) "
                           f"for better cost efficiency "
                           f"({budget_opt['qps_per_dollar']:.2f} QPS/$).")

        lines.append("")

        # Summary Stats
        summary = results.get('summary', {})
        if summary:
            lines.append("SEARCH SUMMARY")
            lines.append("━" * 60)
            lines.append(f"Configurations Evaluated:    {summary.get('num_configs_evaluated', 0)}")
            lines.append(f"Configurations Meeting SLOs: {summary.get('num_configs_meeting_slos', 0)}")
            if 'total_simulation_time_s' in summary:
                lines.append(f"Total Simulation Time:       {summary['total_simulation_time_s']:.1f}s")
            lines.append("")

        # Calibration info
        if results.get('calibrated', False):
            lines.append("Note: Results calibrated with production telemetry data")
            lines.append("")

        return "\n".join(lines)

    def generate_pareto_table(self, pareto_points: list) -> str:
        """Generate ASCII table of Pareto frontier."""
        if not pareto_points:
            return "No Pareto frontier points available"

        lines = []
        lines.append("\nPareto Frontier (Cost vs QPS Trade-offs)")
        lines.append("─" * 80)
        lines.append(f"{'GPU Type':<12} {'GPUs':<6} {'Max QPS':<10} {'Cost/hr':<12} {'QPS/$':<10} {'Config':<20}")
        lines.append("─" * 80)

        for point in sorted(pareto_points, key=lambda x: x['cost_per_hour']):
            config = point.get('config', {})
            gpu_type = point['gpu_type']
            gpus = point.get('total_gpus', config.get('total_gpus', 'N/A'))
            qps = point['max_qps']
            cost = point['cost_per_hour']
            qps_dollar = point['qps_per_dollar']

            config_str = f"TP={config.get('tp', 1)} PP={config.get('pp', 1)} R={config.get('num_replicas', 1)}"

            lines.append(f"{gpu_type:<12} {gpus:<6} {qps:<10.1f} ${cost:<11.2f} {qps_dollar:<10.2f} {config_str:<20}")

        lines.append("─" * 80)
        return "\n".join(lines)
