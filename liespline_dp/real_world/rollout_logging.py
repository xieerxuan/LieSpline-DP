"""Structured rollout events for console output and optional JSONL recordings."""

import json

import click


def emit_event(event, output_stream=None):
    line = json.dumps(event, sort_keys=True)
    click.echo(line)
    if output_stream is not None:
        output_stream.write(line + "\n")
        output_stream.flush()
