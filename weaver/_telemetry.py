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

"""Lightweight telemetry support for Weaver SDK.

This module provides distributed tracing capabilities by generating trace context
and propagating it to the Weaver server. Spans are NOT exported from the SDK -
only the trace context (trace_id, span_id) is passed along via HTTP headers.

The Weaver server extracts this context and continues the trace, exporting all
spans to the configured observability backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv.resource import ResourceAttributes

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer: Tracer | None = None
_INITIALIZED = False


def init_tracer(service_name: str = "weaver-sdk") -> Tracer:
    """
    Initialize a basic OpenTelemetry tracer for the Weaver SDK.

    This creates an in-memory tracer that generates trace context (trace_id, span_id)
    but does NOT export spans to any backend. The trace context is propagated to the
    Weaver server via HTTP headers (W3C Trace Context standard).

    Args:
        service_name: Name of the service for resource attributes

    Returns:
        Tracer instance for creating spans

    Note:
        This function is called automatically by the SDK. Users do not need to call it.
    """
    global _tracer, _INITIALIZED  # pylint: disable=global-statement

    if _INITIALIZED and _tracer is not None:
        return _tracer

    # Create resource with service information
    resource = Resource(
        attributes={
            ResourceAttributes.SERVICE_NAME: service_name,
        }
    )

    # Create a basic tracer provider (no exporters - spans stay in memory)
    provider = TracerProvider(resource=resource)

    # Set as global provider
    trace.set_tracer_provider(provider)

    # Get tracer instance
    _tracer = trace.get_tracer(__name__)
    _INITIALIZED = True

    logger.debug("Weaver SDK tracer initialized (trace context generation enabled)")

    return _tracer


def get_tracer() -> Tracer:
    """
    Get the Weaver SDK tracer instance.

    Returns:
        Tracer instance (initializes if not already done)
    """
    if _tracer is None:
        return init_tracer()
    return _tracer
