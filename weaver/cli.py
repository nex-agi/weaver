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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import click
from rich import box
from rich.console import Console
from rich.table import Table

from ._http import WeaverAPIError
from .service_client import ServiceClient

console = Console()


def format_date(date_str: Any) -> str:
    """Format ISO date string to readable format in local timezone."""
    if not date_str:
        return "N/A"
    try:
        if isinstance(date_str, str):
            # Handle various ISO formats: with Z, with timezone, or without
            date_str_clean = date_str.replace("Z", "+00:00")
            # Try parsing with timezone info
            try:
                dt = datetime.fromisoformat(date_str_clean)
            except ValueError:
                # Fallback: try without timezone (assume UTC)
                dt = datetime.strptime(date_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                # Assume UTC if no timezone info
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = date_str

        # Convert to local timezone if it has timezone info
        if dt.tzinfo is not None:
            # Use astimezone() without arguments to convert to system local timezone
            # This automatically uses the system's timezone configuration
            dt = dt.astimezone()

        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        # Fallback: return the first 19 chars (YYYY-MM-DD HH:MM:SS)
        date_str_str = str(date_str)
        if len(date_str_str) >= 19 and "T" in date_str_str:
            return date_str_str[:10] + " " + date_str_str[11:19]
        return date_str_str


def format_json_output(data: Any) -> None:
    """Pretty-print JSON data."""
    console.print_json(json.dumps(data, default=str, ensure_ascii=False))


def format_training_mode(
    training_mode: Optional[str], lora_config: Optional[Dict[str, Any]] = None
) -> str:
    """Format training mode with LoRA rank if applicable."""
    if not training_mode or training_mode == "N/A":
        return "N/A"

    # Check if it's a LoRA training mode
    if training_mode.lower().startswith("lora"):
        if lora_config and "rank" in lora_config:
            rank = lora_config["rank"]
            return f"{training_mode} (rank={rank})"

    return training_mode


def handle_error(e: Exception) -> None:
    """Handle and display errors gracefully."""
    if isinstance(e, WeaverAPIError):
        if e.status_code == 402:
            console.print(f"[red]Quota exceeded:[/red] {e.message}")
            if e.required_usd is not None or e.available_usd is not None:
                required = f"${e.required_usd}" if e.required_usd is not None else "unknown"
                available = f"${e.available_usd}" if e.available_usd is not None else "unknown"
                console.print(f"Required: {required}; available: {available}")
        elif e.status_code == 429:
            console.print(f"[red]Rate limited:[/red] {e.message}")
            if e.retry_after:
                console.print(f"Retry after: {e.retry_after} seconds")
        elif e.status_code == 503:
            console.print(f"[red]Service temporarily unavailable:[/red] {e.message}")
        else:
            console.print(f"[red]API Error ({e.status_code}):[/red] {e.message}")
        if e.status_code == 401:
            console.print("[yellow]Tip:[/yellow] Check your API key configuration")
        if e.request_id:
            console.print(f"Request ID: {e.request_id}")
    else:
        console.print(f"[red]Error:[/red] {str(e)}")
    sys.exit(1)


def create_training_runs_table(items: List[Dict[str, Any]]) -> Table:
    """Create a rich table for training runs."""
    table = Table(title="Training Runs", box=box.ROUNDED)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Base Model", style="green")
    table.add_column("Training Mode", style="blue")
    table.add_column("Last Request Time", style="magenta")

    for item in items:
        training_mode = format_training_mode(
            item.get("training_mode", "N/A"), item.get("lora_config")
        )
        table.add_row(
            str(item.get("id", ""))[:8],
            item.get("base_model", ""),
            training_mode,
            format_date(item.get("last_request_at")),
        )

    return table


def create_models_table(items: List[Dict[str, Any]]) -> Table:
    """Create a rich table for models."""
    table = Table(title="Models", box=box.ROUNDED)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Session ID", style="blue", no_wrap=True)
    table.add_column("Base Model", style="green")
    table.add_column("Training Mode", style="blue")
    table.add_column("Status", style="yellow")
    table.add_column("Last Seq", justify="right")
    table.add_column("Created At", style="magenta")

    for item in items:
        training_mode = format_training_mode(
            item.get("training_mode", "N/A"), item.get("lora_config")
        )
        table.add_row(
            str(item.get("id", ""))[:8],
            str(item.get("session_id", ""))[:8],
            item.get("base_model", ""),
            training_mode,
            item.get("status", ""),
            str(item.get("last_seq_id", 0)),
            format_date(item.get("created_at")),
        )

    return table


def create_organizations_table(items: List[Dict[str, Any]]) -> Table:
    """Create a copy-friendly organization table."""

    table = Table(title="Organizations", box=box.ROUNDED)
    table.add_column("Organization ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Role", style="blue")
    for item in items:
        table.add_row(
            str(item.get("id", "")),
            str(item.get("name", "")),
            str(item.get("role") or item.get("current_user_role", "")),
        )
    return table


def create_projects_table(items: List[Dict[str, Any]]) -> Table:
    """Create a copy-friendly project table."""

    table = Table(title="Projects", box=box.ROUNDED)
    table.add_column("Project ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Role", style="blue")
    for item in items:
        table.add_row(
            str(item.get("id", "")),
            str(item.get("name", "")),
            str(item.get("role") or item.get("current_user_role", "")),
        )
    return table


def scope_selection_rows(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Return the stable, public fields needed to select an org/project."""

    return [
        {
            "id": str(item.get("id", "")),
            "name": str(item.get("name", "")),
            "role": str(item.get("role") or item.get("current_user_role", "")),
        }
        for item in items
    ]


def display_training_run_detail(data: Dict[str, Any]) -> None:
    """Display detailed training run information."""
    console.print("\n[bold cyan]Training Run Details[/bold cyan]\n")
    console.print(f"[bold]ID:[/bold] {data.get('id')}")
    console.print(f"[bold]Session ID:[/bold] {data.get('session_id')}")
    console.print(f"[bold]Base Model:[/bold] {data.get('base_model')}")
    console.print(f"[bold]Status:[/bold] {data.get('status')}")
    console.print(f"[bold]Model Seq ID:[/bold] {data.get('model_seq_id')}")
    console.print(f"[bold]Last Seq ID:[/bold] {data.get('last_seq_id')}")

    training_mode = format_training_mode(data.get("training_mode", "N/A"), data.get("lora_config"))
    console.print(f"[bold]Training Mode:[/bold] {training_mode}")

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
        checkpoint_table.add_column("TTL", style="yellow")
        checkpoint_table.add_column("Expires At", style="red")

        for cp in checkpoints:
            ttl = cp.get("ttl_seconds")
            ttl_display = f"{ttl}s" if ttl is not None else "permanent"
            expires = format_date(cp.get("expires_at")) if cp.get("expires_at") else "never"
            checkpoint_table.add_row(
                str(cp.get("id", ""))[:8],
                format_date(cp.get("created_at")),
                cp.get("path", "N/A"),
                ttl_display,
                expires,
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

    training_mode = format_training_mode(data.get("training_mode", "N/A"), data.get("lora_config"))
    console.print(f"[bold]Training Mode:[/bold] {training_mode}")

    console.print(f"[bold]Created At:[/bold] {format_date(data.get('created_at'))}")
    console.print(f"[bold]Updated At:[/bold] {format_date(data.get('updated_at'))}")

    # Check if training mode starts with "lora" (includes "lora-r8", "lora", etc.)
    is_lora = (
        training_mode.lower().startswith("lora")
        if training_mode and training_mode != "N/A"
        else False
    )
    lora_config = data.get("lora_config")
    if is_lora and lora_config:
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


@cli.group()
def checkpoint():
    """Manage checkpoints."""


@cli.group("organizations")
def organizations_group():
    """Discover organizations available to the current user."""


@cli.group("projects")
def projects_group():
    """Discover projects available to the current user."""


@cli.group("scope")
def scope_group():
    """Resolve organization and project references."""


def _run_list_organizations(
    output_format: str, base_url: Optional[str], api_key: Optional[str]
) -> None:
    client = ServiceClient(base_url=base_url, api_key=api_key)
    try:
        client.connect(ensure_session=False)
        items = scope_selection_rows(client.list_organizations())
        if output_format == "json":
            format_json_output(items)
        else:
            console.print(create_organizations_table(items))
            console.print(f"\n{len(items)} organizations")
    except Exception as e:
        handle_error(e)
    finally:
        client.close()


def _run_list_projects(
    org_id: Optional[str],
    org_reference: Optional[str],
    output_format: str,
    base_url: Optional[str],
    api_key: Optional[str],
) -> None:
    if org_id and org_reference:
        raise click.UsageError("Use either --organization-id or --organization, not both")
    client = ServiceClient(base_url=base_url, api_key=api_key, organization_id=org_id)
    try:
        client.connect(ensure_session=False)
        resolved_org_id = org_id
        if org_reference:
            scope = client.resolve_scope(org_reference, None)
            organization = scope.get("organization")
            if not isinstance(organization, dict) or not organization.get("id"):
                raise ValueError("Scope response missing organization id")
            resolved_org_id = str(organization["id"])
        items = scope_selection_rows(client.list_projects(resolved_org_id))
        if output_format == "json":
            format_json_output(items)
        else:
            console.print(create_projects_table(items))
            console.print(f"\n{len(items)} projects")
    except Exception as e:
        handle_error(e)
    finally:
        client.close()


def _run_resolve_scope(
    organization: Optional[str],
    project: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> None:
    client = ServiceClient(base_url=base_url, api_key=api_key)
    try:
        client.connect(ensure_session=False)
        format_json_output(client.resolve_scope(organization, project))
    except Exception as e:
        handle_error(e)
    finally:
        client.close()


@list.command("organizations")
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
def list_organizations_cmd(output_format: str, base_url: Optional[str], api_key: Optional[str]):
    """List organizations available to the current user."""

    _run_list_organizations(output_format, base_url, api_key)


@organizations_group.command("list")
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
def organizations_list_cmd(
    output_format: str, base_url: Optional[str], api_key: Optional[str]
) -> None:
    """List organizations available to the current user."""

    _run_list_organizations(output_format, base_url, api_key)


@list.command("projects")
@click.option(
    "--organization-id",
    "--org-id",
    "org_id",
    envvar="WEAVER_ORGANIZATION_ID",
    help="Organization ID; defaults to the user's stable default organization",
)
@click.option(
    "--organization",
    "org_reference",
    envvar="WEAVER_ORGANIZATION",
    help="Organization UUID, globally unique slug, or display name",
)
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
def list_projects_cmd(
    org_id: Optional[str],
    org_reference: Optional[str],
    output_format: str,
    base_url: Optional[str],
    api_key: Optional[str],
):
    """List projects in an organization."""

    _run_list_projects(org_id, org_reference, output_format, base_url, api_key)


@projects_group.command("list")
@click.option(
    "--organization-id",
    "--org-id",
    "org_id",
    envvar="WEAVER_ORGANIZATION_ID",
    help="Organization ID; defaults to the user's default organization",
)
@click.option(
    "--organization",
    "org_reference",
    envvar="WEAVER_ORGANIZATION",
    help="Organization UUID, globally unique slug, or display name",
)
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
def projects_list_cmd(
    org_id: Optional[str],
    org_reference: Optional[str],
    output_format: str,
    base_url: Optional[str],
    api_key: Optional[str],
) -> None:
    """List projects in an organization."""

    _run_list_projects(org_id, org_reference, output_format, base_url, api_key)


@scope_group.command("resolve")
@click.option(
    "--organization",
    envvar="WEAVER_ORGANIZATION",
    help="Organization UUID, globally unique slug, or display name",
)
@click.option(
    "--project",
    envvar="WEAVER_PROJECT",
    help="Project UUID, organization-local slug, or display name",
)
@click.option("--base-url", envvar="WEAVER_BASE_URL", help="Weaver server base URL")
@click.option("--api-key", envvar="WEAVER_API_KEY", help="Weaver API key")
def resolve_scope_cmd(
    organization: Optional[str],
    project: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> None:
    """Resolve references to canonical organization and project IDs."""

    _run_resolve_scope(organization, project, base_url, api_key)


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


@checkpoint.command("set-ttl")
@click.argument("model_id")
@click.argument("path")
@click.argument("seconds", type=int, required=False, default=None)
@click.option("--remove", is_flag=True, help="Cancel expiration (make permanent)")
@click.option("--base-url", envvar="WEAVER_BASE_URL", help="Weaver server base URL")
@click.option("--api-key", envvar="WEAVER_API_KEY", help="Weaver API key")
def checkpoint_set_ttl_cmd(  # pylint: disable=too-many-positional-arguments
    model_id: str,
    path: str,
    seconds: Optional[int],
    remove: bool,
    base_url: Optional[str],
    api_key: Optional[str],
):
    """Set or remove TTL for a checkpoint.

    MODEL_ID is the model that owns the checkpoint.
    PATH is the checkpoint storage path (weaver://...).
    SECONDS is the TTL in seconds (omit when using --remove).
    """
    try:
        if remove and seconds is not None:
            raise click.UsageError("Cannot specify both SECONDS and --remove")
        if not remove and seconds is None:
            raise click.UsageError("Provide SECONDS or use --remove")

        ttl_value: Optional[int] = None if remove else seconds

        with ServiceClient(base_url=base_url, api_key=api_key) as client:
            client.http.patch(
                f"/api/v1/models/{model_id}/checkpoints/ttl",
                json={"path": path, "ttl_seconds": ttl_value},
            )
            if remove:
                console.print(f"[green]Expiration removed for checkpoint:[/green] {path}")
            else:
                console.print(f"[green]TTL set to {seconds}s for checkpoint:[/green] {path}")
    except Exception as e:
        handle_error(e)


if __name__ == "__main__":
    cli()
