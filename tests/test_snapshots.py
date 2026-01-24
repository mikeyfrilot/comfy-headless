"""Tests for workflow snapshot system."""

import tempfile
import time

import pytest


class TestWorkflowSnapshot:
    """Test WorkflowSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test basic snapshot creation."""
        from comfy_headless.workflows import WorkflowSnapshot, WorkflowVersion

        snapshot = WorkflowSnapshot(
            id="test_123_abc",
            version=WorkflowVersion(1, 0, 0),
            workflow={"1": {"class_type": "KSampler"}},
            parameters={"steps": 20},
            workflow_hash="abc123",
            created_at=time.time(),
        )

        assert snapshot.id == "test_123_abc"
        assert str(snapshot.version) == "1.0.0"
        assert "KSampler" in str(snapshot.workflow)

    def test_snapshot_to_dict(self):
        """Test snapshot serialization."""
        from comfy_headless.workflows import WorkflowSnapshot, WorkflowVersion

        snapshot = WorkflowSnapshot(
            id="test_snapshot",
            version=WorkflowVersion(2, 1, 0),
            workflow={"node": "data"},
            parameters={"key": "value"},
            workflow_hash="hash123",
            created_at=1000.0,
            metadata={"author": "test"},
        )

        data = snapshot.to_dict()
        assert data["id"] == "test_snapshot"
        assert data["version"] == "2.1.0"
        assert data["metadata"]["author"] == "test"

    def test_snapshot_from_dict(self):
        """Test snapshot deserialization."""
        from comfy_headless.workflows import WorkflowSnapshot

        data = {
            "id": "restored_snapshot",
            "version": "1.2.3",
            "workflow": {"nodes": []},
            "parameters": {"cfg": 7.0},
            "workflow_hash": "xyz789",
            "created_at": 2000.0,
            "metadata": {},
        }

        snapshot = WorkflowSnapshot.from_dict(data)
        assert snapshot.id == "restored_snapshot"
        assert snapshot.version.major == 1
        assert snapshot.version.minor == 2
        assert snapshot.version.patch == 3

    def test_snapshot_diff(self):
        """Test snapshot comparison."""
        from comfy_headless.workflows import WorkflowSnapshot, WorkflowVersion

        snapshot_a = WorkflowSnapshot(
            id="a",
            version=WorkflowVersion(1, 0, 0),
            workflow={"1": {"type": "A"}, "2": {"type": "B"}},
            parameters={"steps": 20, "cfg": 7.0},
            workflow_hash="hash_a",
            created_at=1000.0,
        )

        snapshot_b = WorkflowSnapshot(
            id="b",
            version=WorkflowVersion(1, 0, 1),
            workflow={"1": {"type": "A_modified"}, "3": {"type": "C"}},
            parameters={"steps": 30, "cfg": 7.0},
            workflow_hash="hash_b",
            created_at=2000.0,
        )

        diff = snapshot_a.diff(snapshot_b)

        assert diff["version_changed"] is True
        assert diff["hash_changed"] is True
        assert "steps" in diff["parameter_changes"]
        assert diff["parameter_changes"]["steps"]["old"] == 20
        assert diff["parameter_changes"]["steps"]["new"] == 30
        assert "1" in diff["node_changes"]["modified"]
        assert "2" in diff["node_changes"]["removed"]
        assert "3" in diff["node_changes"]["added"]


class TestSnapshotManager:
    """Test SnapshotManager functionality."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_storage):
        """Create a snapshot manager with temp storage."""
        from comfy_headless.workflows import SnapshotManager

        return SnapshotManager(storage_path=temp_storage, max_snapshots_per_workflow=5)

    @pytest.fixture
    def sample_compiled(self):
        """Create a sample compiled workflow."""
        from comfy_headless.workflows import CompiledWorkflow

        return CompiledWorkflow(
            template_id="txt2img_standard",
            template_name="Text to Image",
            workflow={"1": {"class_type": "KSampler", "inputs": {"steps": 20}}},
            parameters={"prompt": "test", "steps": 20},
            is_valid=True,
            workflow_hash="sample_hash_123",
        )

    def test_create_snapshot(self, manager, sample_compiled):
        """Test creating a snapshot."""
        snapshot = manager.create_snapshot(sample_compiled)

        assert snapshot is not None
        assert "txt2img_standard" in snapshot.id
        assert str(snapshot.version) == "1.0.0"
        assert snapshot.workflow_hash == sample_compiled.workflow_hash

    def test_list_snapshots(self, manager, sample_compiled):
        """Test listing snapshots for a workflow."""
        # Create multiple snapshots
        for i in range(3):
            sample_compiled.workflow_hash = f"hash_{i}"
            manager.create_snapshot(sample_compiled)
            time.sleep(0.01)  # Small delay for unique timestamps

        snapshots = manager.list_snapshots("txt2img_standard")
        assert len(snapshots) == 3

    def test_get_latest(self, manager, sample_compiled):
        """Test getting the latest snapshot."""
        manager.create_snapshot(sample_compiled)
        time.sleep(0.01)
        sample_compiled.workflow_hash = "newer_hash"
        manager.create_snapshot(sample_compiled)

        latest = manager.get_latest("txt2img_standard")
        assert latest is not None
        assert latest.workflow_hash == "newer_hash"

    def test_rollback(self, manager, sample_compiled):
        """Test rolling back to a snapshot."""
        snapshot = manager.create_snapshot(sample_compiled)

        restored = manager.rollback(snapshot.id)
        assert restored is not None
        assert restored.workflow == sample_compiled.workflow
        assert restored.parameters == sample_compiled.parameters

    def test_compare_snapshots(self, manager, sample_compiled):
        """Test comparing two snapshots."""
        snapshot_a = manager.create_snapshot(sample_compiled)

        # Modify and create another
        sample_compiled.parameters["steps"] = 30
        sample_compiled.workflow_hash = "different_hash"
        snapshot_b = manager.create_snapshot(sample_compiled)

        diff = manager.compare(snapshot_a.id, snapshot_b.id)
        assert diff is not None
        assert diff["hash_changed"] is True
        assert "steps" in diff["parameter_changes"]

    def test_retention_policy(self, manager, sample_compiled):
        """Test that old snapshots are removed when limit is reached."""
        # Create more than max_snapshots (5)
        for i in range(7):
            sample_compiled.workflow_hash = f"hash_{i}"
            manager.create_snapshot(sample_compiled)

        snapshots = manager.list_snapshots("txt2img_standard")
        assert len(snapshots) == 5  # Max is 5

    def test_delete_snapshot(self, manager, sample_compiled):
        """Test deleting a snapshot."""
        snapshot = manager.create_snapshot(sample_compiled)
        snapshot_id = snapshot.id

        assert manager.delete_snapshot(snapshot_id) is True
        assert manager.get_snapshot(snapshot_id) is None

    def test_stats(self, manager, sample_compiled):
        """Test manager statistics."""
        manager.create_snapshot(sample_compiled)

        stats = manager.stats()
        assert stats["workflow_count"] == 1
        assert stats["total_snapshots"] == 1
        assert stats["max_per_workflow"] == 5

    def test_persistence(self, temp_storage, sample_compiled):
        """Test that snapshots persist across manager instances."""
        from comfy_headless.workflows import SnapshotManager

        # Create snapshot with first manager
        manager1 = SnapshotManager(storage_path=temp_storage)
        snapshot = manager1.create_snapshot(sample_compiled)
        snapshot_id = snapshot.id

        # Create new manager instance
        manager2 = SnapshotManager(storage_path=temp_storage)
        restored = manager2.get_snapshot(snapshot_id)

        assert restored is not None
        assert restored.workflow_hash == sample_compiled.workflow_hash


class TestWorkflowVersion:
    """Test WorkflowVersion class."""

    def test_version_string(self):
        """Test version string formatting."""
        from comfy_headless.workflows import WorkflowVersion

        v = WorkflowVersion(2, 3, 4)
        assert str(v) == "2.3.4"

    def test_version_with_label(self):
        """Test version with label."""
        from comfy_headless.workflows import WorkflowVersion

        v = WorkflowVersion(1, 0, 0, label="beta")
        assert str(v) == "1.0.0-beta"

    def test_version_parse(self):
        """Test parsing version string."""
        from comfy_headless.workflows import WorkflowVersion

        v = WorkflowVersion.parse("3.2.1")
        assert v.major == 3
        assert v.minor == 2
        assert v.patch == 1

    def test_version_parse_with_label(self):
        """Test parsing version with label."""
        from comfy_headless.workflows import WorkflowVersion

        v = WorkflowVersion.parse("1.0.0-rc1")
        assert v.major == 1
        assert v.label == "rc1"

    def test_version_bump(self):
        """Test version bumping."""
        from comfy_headless.workflows import WorkflowVersion

        v = WorkflowVersion(1, 2, 3)

        assert str(v.bump_major()) == "2.0.0"
        assert str(v.bump_minor()) == "1.3.0"
        assert str(v.bump_patch()) == "1.2.4"

    def test_version_comparison(self):
        """Test version comparison."""
        from comfy_headless.workflows import WorkflowVersion

        v1 = WorkflowVersion(1, 0, 0)
        v2 = WorkflowVersion(1, 1, 0)
        v3 = WorkflowVersion(2, 0, 0)

        assert v1 < v2
        assert v2 < v3
        assert not v3 < v1
