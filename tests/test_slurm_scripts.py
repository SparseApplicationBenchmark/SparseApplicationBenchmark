from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_refresh_logs_use_invocation_directory(tmp_path):
    scripts = tmp_path / "repo" / "scripts"
    scripts.mkdir(parents=True)
    submission = tmp_path / "submitted from 50%"
    submission.mkdir()
    shutil.copy(ROOT / "scripts/submit-refresh-jobs.sh", scripts)
    setup = scripts / "ensure-poetry-env.sh"
    setup.write_text("#!/usr/bin/env bash\nexit 0\n")
    setup.chmod(0o755)
    record = tmp_path / "submissions.jsonl"
    capture = (
        "import json, os, sys; "
        'f=open(os.environ["SAPS_TEST_SUBMISSIONS"], "a"); '
        'f.write(json.dumps(sys.argv[1:])+"\\n"); f.close(); print("12345")'
    )
    shell_env = tmp_path / "shell-env"
    shell_env.write_text(
        'aws() { if [[ "$1 $2" == "configure list-profiles" ]]; then '
        'printf "dataset-upload\\n"; fi; }\n'
        "poetry() { return 0; }\n"
        "sbatch() { "
        + shlex.quote(sys.executable)
        + " -c "
        + shlex.quote(capture)
        + ' "$@"; }\n'
    )
    env = {
        **os.environ,
        "BASH_ENV": str(shell_env),
        "SAPS_TEST_SUBMISSIONS": str(record),
    }
    subprocess.run(
        ["bash", str(scripts / "submit-refresh-jobs.sh")],
        cwd=submission,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    submissions = [json.loads(line) for line in record.read_text().splitlines()]
    names = ["upload-%j.log", "trace-%A_%a.log", "finalize-metadata-%j.log"]
    assert len(submissions) == len(names)
    for args, name in zip(submissions, names, strict=True):
        expected = str(submission.resolve()).replace("%", "%%") + "/" + name
        assert args[args.index("--output") + 1] == expected
        assert args[args.index("--chdir") + 1] == str(scripts.parent.resolve())


def test_competition_resume_uses_original_task_directory(tmp_path):
    run_root = tmp_path / "old run" / "run_12345"
    run_root.mkdir(parents=True)
    record = tmp_path / "commands.jsonl"
    capture = (
        "import json, os, sys; "
        'f=open(os.environ["SAPS_TEST_COMMANDS"], "a"); '
        'f.write(json.dumps(sys.argv[1:])+"\\n"); f.close()'
    )
    shell_env = tmp_path / "shell-env"
    shell_env.write_text(
        "poetry() { "
        + shlex.quote(sys.executable)
        + " -c "
        + shlex.quote(capture)
        + ' "$@"; }\n'
    )
    env = {
        **os.environ,
        "BASH_ENV": str(shell_env),
        "SAPS_TEST_COMMANDS": str(record),
        "SAPS_REPO_DIRECTORY": str(ROOT),
        "SLURM_ARRAY_JOB_ID": "99999",
        "SLURM_ARRAY_TASK_ID": "2",
        "SLURM_ARRAY_TASK_COUNT": "5",
    }
    env.pop("SAPS_COMPETITION_ARGS", None)
    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run-competition.slurm"),
            "--resume",
            str(run_root),
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    run, combine = [json.loads(line) for line in record.read_text().splitlines()]
    task_directory = str(run_root.resolve() / "task_2")
    assert "--resume" in run
    assert run[run.index("--saps-dir") + 1] == task_directory
    assert run[run.index("--results-dir") + 1] == task_directory + "/results"
    assert run[run.index("--machine") + 1] == "run_12345-task-2"
    assert combine[combine.index("--run-directory") + 1] == task_directory
    assert combine[combine.index("--output") + 1] == str(
        run_root.resolve().parent / "results_12345_task_2.json"
    )
    for script in (ROOT / "scripts").glob("*.slurm"):
        text = script.read_text()
        assert "#SBATCH --output=" in text
        assert "#SBATCH --output=/dev/null" not in text
        assert "exec >" not in text
