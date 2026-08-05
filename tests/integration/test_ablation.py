import json
from pathlib import Path

import pytest

from evoshift.ablation import run_memory_ablation_audit
from evoshift.audit import load_evolved_state
from evoshift.benchmarks import create_benchmark
from evoshift.config import load_config
from evoshift.runner import EvoShiftRunner


@pytest.mark.asyncio
@pytest.mark.integration
async def test_heldout_memory_ablation_is_paired_and_frozen(tmp_path: Path) -> None:
    source_config = load_config(Path("configs/experiments/offline_demo.yaml"))
    source_config = source_config.model_copy(
        update={
            "storage": source_config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "source"), "cache_enabled": False}
            )
        }
    )
    source_adapter = create_benchmark(
        source_config.benchmark, root=Path.cwd(), seed=source_config.evaluation.seed
    )
    source = await EvoShiftRunner(source_config, source_adapter, workdir=Path.cwd()).run()
    state = load_evolved_state(source.run_dir)

    heldout_config = source_config.model_copy(
        update={
            "evaluation": source_config.evaluation.model_copy(update={"seed": 99}),
            "storage": source_config.storage.model_copy(
                update={"runs_dir": str(tmp_path / "ablation"), "cache_enabled": False}
            ),
        }
    )
    heldout_adapter = create_benchmark(
        heldout_config.benchmark, root=Path.cwd(), seed=heldout_config.evaluation.seed
    )
    audit = await run_memory_ablation_audit(
        heldout_config,
        heldout_adapter,
        state,
        workdir=Path.cwd(),
    )

    report = audit.report
    assert report["sample_ids_aligned"] is True
    assert report["all_frozen_states_unchanged"] is True
    assert report["n_cards_tested"] == report["n_cards_source"]
    assert audit.report_path.exists()
    assert audit.markdown_path.exists()
    for card in report["cards"]:
        manifest = json.loads((Path(card["run_dir"]) / "manifest.json").read_text())
        assert manifest["audit_variant"] == "leave_one_memory_out"
        assert manifest["source_state_hash"] == state.fingerprint
        assert manifest["excluded_memory_id"] == card["memory_id"]
        assert card["state_unchanged"] is True
