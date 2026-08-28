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
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import click
from rich import box
from rich.console import Console
from rich.table import Table

from ._artifacts import (
    DEFAULT_EXPORT_TTL_SECONDS,
    is_artifact_payload,
    parse_download_target,
    resolve_checkpoint_id_from_listing,
    validate_resource_id,
)
from ._deployments import build_create_deployment_body, translate_deployment_error
from ._http import WeaverAPIError
from .operations import build_operation_handle
from .service_client import ServiceClient
from .types.deployment import Deployment
from .types.supported_model import SupportedModel, SupportedModelPrice

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


def create_deployments_table(items: List[Deployment]) -> Table:
    """Create a rich table for deployments."""
    table = Table(title="Deployments", box=box.ROUNDED)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Endpoint", style="blue")
    table.add_column("GPU", style="magenta")
    table.add_column("Replicas", justify="right")
    table.add_column("Created At", style="magenta")

    for item in items:
        table.add_row(
            str(item.id or "")[:8],
            item.name or "",
            item.status or "",
            item.endpoint or "-",
            item.gpu_type or "default",
            str(item.replicas if item.replicas is not None else ""),
            format_date(item.created_at),
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


def format_supported_model_price(price: SupportedModelPrice | None) -> str:
    """Format an exact catalog price for narrow terminal output."""

    if price is None:
        return "[dim]Not priced[/dim]"
    if price.unit == "million_tokens":
        return f"${price.unit_price_usd}"
    return f"${price.unit_price_usd} / {price.unit}"


def create_supported_models_table(items: List[SupportedModel], mode: str = "all") -> Table:
    """Render one compact row for each explicitly supported model mode."""

    normalized_mode = mode.replace("-", "_")
    table = Table(title="Supported Models · USD per 1M tokens", box=box.ROUNDED)
    table.add_column("Model", style="green")
    table.add_column("Mode", style="cyan", no_wrap=True)
    table.add_column("Train", justify="right", no_wrap=True)
    table.add_column("Input", justify="right", no_wrap=True)
    table.add_column("Cached input", justify="right", no_wrap=True)
    table.add_column("Output", justify="right", no_wrap=True)
    for model in items:
        modes = [
            item
            for item in model.training_modes
            if normalized_mode == "all" or item.mode == normalized_mode
        ]
        for index, training_mode in enumerate(modes):
            table.add_row(
                model.name if index == 0 else "",
                training_mode.display_name,
                format_supported_model_price(training_mode.price_for("training_tokens")),
                format_supported_model_price(training_mode.price_for("sampling_prefill_tokens")),
                format_supported_model_price(
                    training_mode.price_for("sampling_cached_prefill_tokens")
                ),
                format_supported_model_price(training_mode.price_for("sampling_output_tokens")),
                end_section=index == len(modes) - 1,
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


@cli.group()
def deployment():
    """Publish checkpoints as public endpoints and manage them."""


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


@list.command("supported-models")
@click.option(
    "--mode",
    type=click.Choice(["all", "lora", "full-ft"]),
    default="all",
    show_default=True,
    help="Show both training modes or only one mode",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json", "names"]),
    default="table",
    show_default=True,
    help="Table for humans, JSON for scripts, or one model name per line",
)
@click.option(
    "--organization-id",
    "--org-id",
    envvar="WEAVER_ORGANIZATION_ID",
    help="Organization whose effective catalog prices should be shown",
)
@click.option("--base-url", envvar="WEAVER_BASE_URL", help="Weaver server base URL")
@click.option("--api-key", envvar="WEAVER_API_KEY", help="Weaver API key")
def list_supported_models_cmd(
    mode: str,
    output_format: str,
    organization_id: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
) -> None:
    """List usable models with separate LoRA and Full-FT choices."""

    client = ServiceClient(
        base_url=base_url,
        api_key=api_key,
        organization_id=organization_id,
    )
    try:
        client.connect(ensure_session=False)
        models = client.list_supported_model_details()
        normalized_mode = mode.replace("-", "_")
        selected_models = (
            models
            if normalized_mode == "all"
            else [model for model in models if model.training_mode(normalized_mode) is not None]
        )
        if output_format == "names":
            for model in selected_models:
                click.echo(model.name)
        elif output_format == "json":
            selected_mode = None if mode == "all" else mode
            format_json_output([model.to_dict(mode=selected_mode) for model in selected_models])
        else:
            console.print(create_supported_models_table(selected_models, mode))
            console.print(f"\n{len(selected_models)} supported models")
    except Exception as e:
        handle_error(e)
    finally:
        client.close()


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


def _resolve_cli_checkpoint_id(client: ServiceClient, target: str) -> str:
    """Resolve a ``weaver://`` checkpoint URI (or raw id) to a checkpoint id."""
    if not target.startswith("weaver://"):
        # Raw ids are interpolated into API paths by the commands below; the
        # same UUID guard the client methods apply must hold here, or
        # `deployment create ../models/<id>` reroutes the request.
        return validate_resource_id(target, kind="checkpoint")
    parsed = parse_download_target(target)
    listing = client.http.get(f"/api/v1/models/{parsed.model_id}/checkpoints")
    items = (listing or {}).get("items", []) if isinstance(listing, dict) else []
    checkpoint_id = resolve_checkpoint_id_from_listing(items, parsed.checkpoint_path or "")
    if checkpoint_id is None:
        raise ValueError(
            f"No checkpoint with path {parsed.checkpoint_path!r} found for "
            f"model {parsed.model_id}"
        )
    return checkpoint_id


@checkpoint.command("export")
@click.argument("target")
@click.option(
    "--merge-adapter",
    is_flag=True,
    help="Merge a LoRA adapter into the base model (exports a full HF model)",
)
@click.option(
    "--ttl",
    "ttl_seconds",
    type=int,
    default=DEFAULT_EXPORT_TTL_SECONDS,
    show_default=True,
    help="Artifact TTL in seconds",
)
@click.option("--force", is_flag=True, help="Re-export even if a completed artifact exists")
@click.option("--no-wait", is_flag=True, help="Enqueue the export and exit without waiting")
@click.option("--base-url", envvar="WEAVER_BASE_URL", help="Weaver server base URL")
@click.option("--api-key", envvar="WEAVER_API_KEY", help="Weaver API key")
def checkpoint_export_cmd(  # pylint: disable=too-many-positional-arguments
    target: str,
    merge_adapter: bool,
    ttl_seconds: int,
    force: bool,
    no_wait: bool,
    base_url: Optional[str],
    api_key: Optional[str],
):
    """Export a checkpoint to HuggingFace format.

    TARGET is a checkpoint weaver:// URI or a checkpoint id. The export
    produces a downloadable artifact (full HF model for full fine-tuning,
    HF PEFT adapter for LoRA); fetch it with `weaver checkpoint download`.
    """
    client = ServiceClient(base_url=base_url, api_key=api_key)
    try:
        client.connect(ensure_session=False)
        checkpoint_id = _resolve_cli_checkpoint_id(client, target)
        body = {
            "format": "huggingface",
            "merge_adapter": merge_adapter,
            "ttl_seconds": ttl_seconds,
            "force": force,
        }
        response = client.http.post(
            f"/api/v1/checkpoints/{checkpoint_id}/export", json=body, max_retries=1
        )
        # A completed idempotent hit answers with the artifact itself instead
        # of an operation envelope.
        if is_artifact_payload(response):
            console.print("[green]Export already completed:[/green]")
            format_json_output(response)
            return
        handle = build_operation_handle(client.http, response if isinstance(response, dict) else {})
        if no_wait:
            console.print(f"[green]Export enqueued.[/green] Operation ID: {handle.operation_id}")
            return
        console.print(f"Waiting for export operation {handle.operation_id}...")
        result = handle.result()
        console.print("[green]Export completed:[/green]")
        format_json_output(result)
    except Exception as e:
        handle_error(e)
    finally:
        client.close()


@checkpoint.command("download")
@click.argument("uri")
@click.option(
    "--output",
    "-o",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to download the files into",
)
@click.option(
    "--kind",
    type=click.Choice(["hf_model", "hf_adapter"]),
    default=None,
    help="Artifact kind to select when the checkpoint has several",
)
@click.option("--base-url", envvar="WEAVER_BASE_URL", help="Weaver server base URL")
@click.option("--api-key", envvar="WEAVER_API_KEY", help="Weaver API key")
def checkpoint_download_cmd(  # pylint: disable=too-many-positional-arguments
    uri: str,
    output_dir: str,
    kind: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
):
    """Download exported HF weights to a local directory.

    URI is an artifact weaver:// URI (…/artifacts/{kind}), a checkpoint
    weaver:// URI, or an artifact id. Requires a completed export — run
    `weaver checkpoint export` first.
    """
    client = ServiceClient(base_url=base_url, api_key=api_key)
    try:
        client.connect(ensure_session=False)
        dest = client.download_weights(uri, output_dir, kind=kind)
        console.print(f"[green]Downloaded weights to:[/green] {dest}")
    except Exception as e:
        handle_error(e)
    finally:
        client.close()


@deployment.command("create")
@click.argument("checkpoint_uri")
@click.option("--name", required=True, help="Public model name to publish under")
@click.option("--gpu-type", default=None, help="GPU type to serve on (default: server's choice)")
@click.option("--replicas", type=int, default=1, show_default=True, help="Serving replicas (1-8)")
@click.option(
    "--gpus-per-replica",
    type=int,
    default=None,
    help="GPUs per replica (1-16, default: sized by the launcher)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace an existing gateway registration with this name",
)
@click.option("--no-wait", is_flag=True, help="Start the deployment and exit without waiting")
@click.option("--base-url", envvar="WEAVER_BASE_URL", help="Weaver server base URL")
@click.option("--api-key", envvar="WEAVER_API_KEY", help="Weaver API key")
def deployment_create_cmd(  # pylint: disable=too-many-positional-arguments,too-many-arguments
    checkpoint_uri: str,
    name: str,
    gpu_type: Optional[str],
    replicas: int,
    gpus_per_replica: Optional[int],
    overwrite: bool,
    no_wait: bool,
    base_url: Optional[str],
    api_key: Optional[str],
):
    """Publish a checkpoint as a public OpenAI-compatible endpoint.

    CHECKPOINT_URI is a checkpoint weaver:// URI or a checkpoint id. The
    checkpoint is converted to HuggingFace format, launched as a standalone
    inference workload, and registered on the NorthGate gateway under --name.
    This takes tens of minutes, mostly conversion.

    Publishing is permission-gated: the server grants it by principal origin
    (SSO always; an API key only when minted under an allowlisted biz_code).
    """
    client = ServiceClient(base_url=base_url, api_key=api_key)
    try:
        client.connect(ensure_session=False)
        body = build_create_deployment_body(
            name=name,
            gpu_type=gpu_type,
            replicas=replicas,
            gpus_per_replica=gpus_per_replica,
            overwrite=overwrite,
        )
        checkpoint_id = _resolve_cli_checkpoint_id(client, checkpoint_uri)
        try:
            # max_retries=1: this POST launches GPUs and claims a global name.
            response = client.http.post(
                f"/api/v1/checkpoints/{checkpoint_id}/deployments", json=body, max_retries=1
            )
        except WeaverAPIError as exc:
            raise translate_deployment_error(exc) from exc
        handle = build_operation_handle(client.http, response if isinstance(response, dict) else {})
        if no_wait:
            console.print(f"[green]Deployment started.[/green] Operation ID: {handle.operation_id}")
            return
        console.print(f"Waiting for deployment operation {handle.operation_id}...")
        result = handle.result()
        console.print("[green]Deployment ready:[/green]")
        format_json_output(result)
    except Exception as e:
        handle_error(e)
    finally:
        client.close()


@deployment.command("list")
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
def deployment_list_cmd(output_format: str, base_url: Optional[str], api_key: Optional[str]):
    """List the deployments you published."""
    client = ServiceClient(base_url=base_url, api_key=api_key)
    try:
        client.connect(ensure_session=False)
        deployments = client.list_deployments()
        if output_format == "json":
            format_json_output([asdict(item) for item in deployments])
        else:
            console.print(create_deployments_table(deployments))
            console.print(f"\nShowing {len(deployments)} deployment(s)")
    except Exception as e:
        handle_error(e)
    finally:
        client.close()


@deployment.command("get")
@click.argument("deployment_id")
@click.option("--base-url", envvar="WEAVER_BASE_URL", help="Weaver server base URL")
@click.option("--api-key", envvar="WEAVER_API_KEY", help="Weaver API key")
def deployment_get_cmd(deployment_id: str, base_url: Optional[str], api_key: Optional[str]):
    """Show one deployment, including its endpoint URL."""
    client = ServiceClient(base_url=base_url, api_key=api_key)
    try:
        client.connect(ensure_session=False)
        format_json_output(asdict(client.get_deployment(deployment_id)))
    except Exception as e:
        handle_error(e)
    finally:
        client.close()


@deployment.command("delete")
@click.argument("deployment_id")
@click.option("--no-wait", is_flag=True, help="Start the teardown and exit without waiting")
@click.option("--base-url", envvar="WEAVER_BASE_URL", help="Weaver server base URL")
@click.option("--api-key", envvar="WEAVER_API_KEY", help="Weaver API key")
def deployment_delete_cmd(
    deployment_id: str,
    no_wait: bool,
    base_url: Optional[str],
    api_key: Optional[str],
):
    """Take a deployment down and release its name."""
    client = ServiceClient(base_url=base_url, api_key=api_key)
    try:
        client.connect(ensure_session=False)
        if no_wait:
            handle = client.delete_deployment(deployment_id, wait=False)
            console.print(f"[green]Teardown started.[/green] Operation ID: {handle.operation_id}")
            return
        console.print(f"Tearing down deployment {deployment_id}...")
        stopped = client.delete_deployment(deployment_id)
        console.print("[green]Deployment stopped:[/green]")
        format_json_output(asdict(stopped))
    except Exception as e:
        handle_error(e)
    finally:
        client.close()


if __name__ == "__main__":
    cli()
