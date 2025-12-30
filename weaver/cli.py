# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line interface for Weaver SDK."""

import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import click
from rich import box
from rich.console import Console
from rich.table import Table

from ._http import WeaverAPIError
from .service_client import ServiceClient

console = Console()


def format_date(date_str: Any) -> str:
    """Format ISO date string to readable format."""
    if not date_str:
        return "N/A"
    try:
        if isinstance(date_str, str):
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
            dt = date_str
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return str(date_str)


def format_json_output(data: Any) -> None:
    """Pretty-print JSON data."""
    console.print_json(json.dumps(data, default=str, ensure_ascii=False))


def handle_error(e: Exception) -> None:
    """Handle and display errors gracefully."""
    if isinstance(e, WeaverAPIError):
        console.print(f"[red]API Error ({e.status_code}):[/red] {e.message}")
        if e.status_code == 401:
            console.print("[yellow]Tip:[/yellow] Check your API key configuration")
    else:
        console.print(f"[red]Error:[/red] {str(e)}")
    sys.exit(1)


def create_training_runs_table(items: List[Dict[str, Any]]) -> Table:
    """Create a rich table for training runs."""
    table = Table(title="Training Runs", box=box.ROUNDED)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Base Model", style="green")
    table.add_column("LoRA Rank", justify="right")
    table.add_column("Last Request Time", style="magenta")

    for item in items:
        lora_rank = str(item.get("lora_rank", "N/A")) if item.get("lora_rank") else "N/A"
        table.add_row(
            str(item.get("id", ""))[:8],
            item.get("base_model", ""),
            lora_rank,
            format_date(item.get("last_request_at")),
        )

    return table


def create_models_table(items: List[Dict[str, Any]]) -> Table:
    """Create a rich table for models."""
    table = Table(title="Models", box=box.ROUNDED)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Session ID", style="blue", no_wrap=True)
    table.add_column("Base Model", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Last Seq", justify="right")
    table.add_column("Created At", style="magenta")

    for item in items:
        table.add_row(
            str(item.get("id", ""))[:8],
            str(item.get("session_id", ""))[:8],
            item.get("base_model", ""),
            item.get("status", ""),
            str(item.get("last_seq_id", 0)),
            format_date(item.get("created_at")),
        )

    return table


def display_training_run_detail(data: Dict[str, Any]) -> None:
    """Display detailed training run information."""
    console.print("\n[bold cyan]Training Run Details[/bold cyan]\n")
    console.print(f"[bold]ID:[/bold] {data.get('id')}")
    console.print(f"[bold]Session ID:[/bold] {data.get('session_id')}")
    console.print(f"[bold]Base Model:[/bold] {data.get('base_model')}")
    console.print(f"[bold]Status:[/bold] {data.get('status')}")
    console.print(f"[bold]Model Seq ID:[/bold] {data.get('model_seq_id')}")
    console.print(f"[bold]Last Seq ID:[/bold] {data.get('last_seq_id')}")

    if data.get("lora_rank"):
        console.print(f"[bold]LoRA Rank:[/bold] {data.get('lora_rank')}")

    console.print(f"[bold]Owner User ID:[/bold] {data.get('owner_user_id', 'N/A')}")
    console.print(f"[bold]Owner Tenant ID:[/bold] {data.get('owner_tenant_id', 'N/A')}")
    console.print(f"[bold]Created At:[/bold] {format_date(data.get('created_at'))}")
    console.print(f"[bold]Last Request At:[/bold] {format_date(data.get('last_request_at'))}")

    checkpoints = data.get("checkpoints", [])
    if checkpoints:
        console.print(f"\n[bold cyan]Checkpoints ({len(checkpoints)}):[/bold cyan]")
        checkpoint_table = Table(box=box.SIMPLE)
        checkpoint_table.add_column("ID", style="cyan", no_wrap=True)
        checkpoint_table.add_column("Created At", style="magenta")
        checkpoint_table.add_column("Full Path", style="green")

        for cp in checkpoints:
            checkpoint_table.add_row(
                str(cp.get("id", ""))[:8],
                format_date(cp.get("created_at")),
                cp.get("path", "N/A"),
            )
        console.print(checkpoint_table)
        console.print("\n[dim]Tip: Copy the full path to use with sampling clients[/dim]")


def display_model_detail(data: Dict[str, Any]) -> None:
    """Display detailed model information."""
    console.print("\n[bold cyan]Model Details[/bold cyan]\n")
    console.print(f"[bold]ID:[/bold] {data.get('id')}")
    console.print(f"[bold]Session ID:[/bold] {data.get('session_id')}")
    console.print(f"[bold]Base Model:[/bold] {data.get('base_model')}")
    console.print(f"[bold]Status:[/bold] {data.get('status')}")
    console.print(f"[bold]Model Seq ID:[/bold] {data.get('model_seq_id')}")
    console.print(f"[bold]Last Seq ID:[/bold] {data.get('last_seq_id')}")
    console.print(f"[bold]Created At:[/bold] {format_date(data.get('created_at'))}")
    console.print(f"[bold]Updated At:[/bold] {format_date(data.get('updated_at'))}")

    lora_config = data.get("lora_config")
    if lora_config:
        console.print("\n[bold cyan]LoRA Configuration:[/bold cyan]")
        console.print_json(json.dumps(lora_config, indent=2))

    user_metadata = data.get("user_metadata")
    if user_metadata:
        console.print("\n[bold cyan]User Metadata:[/bold cyan]")
        console.print_json(json.dumps(user_metadata, indent=2))


@click.group()
def cli():
    """Weaver SDK command-line interface.

    Manage and view training runs, models, and more.
    """


@cli.group()
def list():  # pylint: disable=redefined-builtin
    """List resources (training runs, models, etc.)."""


@cli.group()
def show():
    """Show detailed information about a specific resource."""


@list.command("training-runs")
@click.option("--limit", "-l", default=25, help="Number of items to return")
@click.option("--offset", "-o", default=0, help="Number of items to skip")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.option("--base-url", envvar="WEAVER_BASE_URL", help="Weaver server base URL")
@click.option("--api-key", envvar="WEAVER_API_KEY", help="Weaver API key")
def list_training_runs_cmd(
    limit: int, offset: int, output_format: str, base_url: Optional[str], api_key: Optional[str]
):
    """List training runs."""
    try:
        with ServiceClient(base_url=base_url, api_key=api_key) as client:
            result = client.list_training_runs(limit=limit, offset=offset)

            items = result.get("items", [])
            pagination = result.get("pagination", {})

            if output_format == "json":
                format_json_output(result)
            else:
                table = create_training_runs_table(items)
                console.print(table)
                total = pagination.get("total_count", len(items))
                console.print(f"\nShowing {len(items)} of {total} training runs (offset: {offset})")
    except Exception as e:
        handle_error(e)


@list.command("models")
@click.option("--limit", "-l", default=25, help="Number of items to return")
@click.option("--offset", "-o", default=0, help="Number of items to skip")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.option("--base-url", envvar="WEAVER_BASE_URL", help="Weaver server base URL")
@click.option("--api-key", envvar="WEAVER_API_KEY", help="Weaver API key")
def list_models_cmd(
    limit: int, offset: int, output_format: str, base_url: Optional[str], api_key: Optional[str]
):
    """List models."""
    try:
        with ServiceClient(base_url=base_url, api_key=api_key) as client:
            result = client.list_models(limit=limit, offset=offset)

            items = result.get("items", [])
            pagination = result.get("pagination", {})

            if output_format == "json":
                format_json_output(result)
            else:
                table = create_models_table(items)
                console.print(table)
                total = pagination.get("total_count", len(items))
                console.print(f"\nShowing {len(items)} of {total} models (offset: {offset})")
    except Exception as e:
        handle_error(e)


@show.command("training-run")
@click.argument("run_id")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["detail", "json"]),
    default="detail",
    help="Output format",
)
@click.option("--base-url", envvar="WEAVER_BASE_URL", help="Weaver server base URL")
@click.option("--api-key", envvar="WEAVER_API_KEY", help="Weaver API key")
def show_training_run_cmd(
    run_id: str, output_format: str, base_url: Optional[str], api_key: Optional[str]
):
    """Show detailed information about a training run."""
    try:
        with ServiceClient(base_url=base_url, api_key=api_key) as client:
            result = client.get_training_run(run_id)

            if output_format == "json":
                format_json_output(result)
            else:
                display_training_run_detail(result)
    except Exception as e:
        handle_error(e)


@show.command("model")
@click.argument("model_id")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["detail", "json"]),
    default="detail",
    help="Output format",
)
@click.option("--base-url", envvar="WEAVER_BASE_URL", help="Weaver server base URL")
@click.option("--api-key", envvar="WEAVER_API_KEY", help="Weaver API key")
def show_model_cmd(
    model_id: str, output_format: str, base_url: Optional[str], api_key: Optional[str]
):
    """Show detailed information about a model."""
    try:
        with ServiceClient(base_url=base_url, api_key=api_key) as client:
            result = client.get_model(model_id)

            if output_format == "json":
                format_json_output(result)
            else:
                display_model_detail(result)
    except Exception as e:
        handle_error(e)


if __name__ == "__main__":
    cli()
