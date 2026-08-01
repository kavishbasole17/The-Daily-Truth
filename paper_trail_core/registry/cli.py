"""
Model registry CLI — query trained models from the command line.

Usage examples:
    python -m paper_trail_core.registry.cli list
    python -m paper_trail_core.registry.cli top --n 5 --metric f1
    python -m paper_trail_core.registry.cli top --n 3 --metric auc --category "Classical ML"
    python -m paper_trail_core.registry.cli top --n 5 --metric f1 --feature-type tfidf
"""
from __future__ import annotations

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import click
from paper_trail_core.registry.store import get_all_models, get_top_models


def _row(e) -> str:
    acc = f"{e.accuracy:.4f}" if e.accuracy is not None else "  N/A "
    f1  = f"{e.f1:.4f}"      if e.f1       is not None else "  N/A "
    auc = f"{e.auc:.4f}"     if e.auc      is not None else "  N/A "
    tt  = f"{e.train_time:.1f}s" if e.train_time is not None else "   N/A"
    ok  = "✓" if e.loaded else "✗"
    return (
        f"  {ok} {e.name:<28}  {e.category:<14}  {e.feature_type:<8}"
        f"  acc={acc}  f1={f1}  auc={auc}  t={tt}"
    )


@click.group()
def cli():
    """Paper Trail — model registry CLI."""


@cli.command("list")
def list_models():
    """List all models in the registry."""
    entries = get_all_models()
    if not entries:
        click.echo("No models found.  Run `python export_models.py` first.")
        return
    click.echo(f"\n{'Model':<30}  {'Category':<14}  {'Features':<8}  Metrics")
    click.echo("─" * 85)
    for e in entries:
        click.echo(_row(e))
    click.echo(f"\n  Total: {len(entries)} models")


@cli.command("top")
@click.option("--n",            default=5,           show_default=True, help="Number of models")
@click.option("--metric",       default="f1",        show_default=True,
              type=click.Choice(["accuracy", "f1", "precision", "recall", "auc"]),
              help="Sort metric")
@click.option("--category",     default=None,        help="Filter by category")
@click.option("--feature-type", default=None,        help="Filter by feature type")
def top_models(n, metric, category, feature_type):
    """Show the top-N models sorted by a chosen metric."""
    entries = get_top_models(n=n, metric=metric, category=category, feature_type=feature_type)
    if not entries:
        click.echo("No models match the given filters.")
        return
    label = f"Top {n} by {metric}"
    if category:    label += f"  (category={category})"
    if feature_type: label += f"  (feature_type={feature_type})"
    click.echo(f"\n{label}")
    click.echo("─" * 85)
    for rank, e in enumerate(entries, 1):
        click.echo(f"  #{rank}  {_row(e)}")


if __name__ == "__main__":
    cli()
