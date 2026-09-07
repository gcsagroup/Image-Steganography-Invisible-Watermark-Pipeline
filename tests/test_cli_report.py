from dataclasses import replace

from PipeLine.cli import build_parser
from PipeLine.configuration import load_config
from PipeLine.schemas import TestImageResult as ImageResult
from PipeLine.testing.report import render_report, write_report


def test_cli_parser_has_all_commands():
    parser = build_parser()
    for command in [
        "preprocess", "stability", "convert-stability", "embed",
        "extract", "evaluate", "test",
    ]:
        if command == "test":
            args = parser.parse_args([command])
        elif command == "convert-stability":
            args = parser.parse_args([command, "--input", "x.pt", "--profile", "1024x1024"])
        elif command == "stability":
            args = parser.parse_args([command, "--original-dir", "a", "--reconstructed-dir", "b"])
        elif command == "extract":
            args = parser.parse_args([command, "--image", "a.png"])
        elif command == "evaluate":
            args = parser.parse_args([command, "--cover", "a.png", "--stego", "b.png", "--text", "x"])
        else:
            args = parser.parse_args([command, "--image", "a.png", "--text", "x"])
        assert args.command == command


def test_markdown_report_contains_rows_and_aggregate(tmp_path):
    config = load_config()
    config = replace(config, testing=replace(config.testing, report_dir=tmp_path))
    results = [ImageResult(filename="bad.png", error="unsupported")]
    rendered = render_report(config, results)
    assert "bad.png" in rendered
    assert "## Aggregate" in rendered
    assert "Frame header:" in rendered
    assert "Capacity ratio: 0.005 bit/pixel" in rendered
    assert "End marker:" not in rendered
    path = write_report(config, results)
    assert path.is_file()
