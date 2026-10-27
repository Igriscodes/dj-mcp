import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from ai_dj.config import DJConfig, get_default_config
from ai_dj.errors import DJError, PrerequisiteError, ResourceNotFoundError
from ai_dj.jobs import JobManager
from ai_dj.library import AudioLibrary
from ai_dj.project import Project
from ai_dj.recovery import StateSnapshotManager
from ai_dj.rendering import ProjectRenderer


class WorkflowStep:
    def __init__(
        self,
        step_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        condition_field: Optional[str] = None,
        condition_operator: Optional[str] = None,
        condition_value: Optional[Any] = None,
        on_failure: str = "stop",  # "stop", "retry", "rollback", "continue"
    ):
        self.step_id = step_id
        self.tool_name = tool_name
        self.arguments = arguments
        self.condition_field = condition_field
        self.condition_operator = condition_operator
        self.condition_value = condition_value
        self.on_failure = on_failure
        self.status = "pending"
        self.result = None
        self.error = None

    def evaluate_condition(self, context: Dict[str, Any]) -> bool:
        if not self.condition_field:
            return True
        val = context.get(self.condition_field)
        op = self.condition_operator or "equals"
        if op in ("equals", "=="):
            return val == self.condition_value
        elif op in ("not_equals", "!="):
            return val != self.condition_value
        elif op == "is_none":
            return val is None
        elif op == "is_not_none":
            return val is not None
        elif op == "contains":
            return self.condition_value in (val or [])
        return True


class Workflow:
    def __init__(self, workflow_id: str, name: str):
        self.workflow_id = workflow_id
        self.name = name
        self.steps: List[WorkflowStep] = []
        self.status = "created"  # "created", "running", "completed", "failed", "paused", "cancelled"
        self.current_step_idx = 0
        self.context: Dict[str, Any] = {}

    def add_step(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        condition_field: Optional[str] = None,
        condition_operator: Optional[str] = None,
        condition_value: Optional[Any] = None,
        on_failure: str = "stop",
    ) -> WorkflowStep:
        step_id = f"step_{len(self.steps) + 1}"
        step = WorkflowStep(
            step_id=step_id,
            tool_name=tool_name,
            arguments=arguments,
            condition_field=condition_field,
            condition_operator=condition_operator,
            condition_value=condition_value,
            on_failure=on_failure,
        )
        self.steps.append(step)
        return step


class Session:
    def __init__(self, config: Optional[DJConfig] = None):
        self.config = config or get_default_config()
        self.library = AudioLibrary(self.config)
        self.job_manager = JobManager(max_workers=self.config.max_workers)
        self.renderer = ProjectRenderer(self.config)
        self.snapshot_manager = StateSnapshotManager(self.config.cache_dir / "snapshots")
        self.current_project: Optional[Project] = None
        self.workflows: Dict[str, Workflow] = {}
        self._tool_dispatcher: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None

    def require_project(self) -> Project:
        if self.current_project is None:
            raise PrerequisiteError(
                code="NO_ACTIVE_PROJECT",
                message="No project is currently open in the session.",
                required_action="project_create",
                suggested_tool="project_create",
                suggested_arguments={"project_id": "proj_1", "name": "New Project"},
            )
        return self.current_project

    def create_project(self, project_id: str, name: str, sample_rate: int = 44100, num_decks: int = 2) -> Project:
        proj = Project.create_new(project_id=project_id, name=name, sample_rate=sample_rate, num_decks=num_decks)
        self.current_project = proj
        self.take_snapshot(f"init_{project_id}")
        return proj

    def take_snapshot(self, label: Optional[str] = None) -> str:
        proj = self.require_project()
        snap_id = f"snap_{label or uuid.uuid4().hex[:8]}"
        self.snapshot_manager.save_snapshot(snap_id, proj.model_dump(mode="json"))
        return snap_id

    def restore_snapshot(self, snapshot_id: str) -> Project:
        data = self.snapshot_manager.load_snapshot(snapshot_id)
        proj = Project.model_validate(data)
        self.current_project = proj
        return proj

    def undo(self) -> Optional[Project]:
        data = self.snapshot_manager.undo()
        if data:
            self.current_project = Project.model_validate(data)
            return self.current_project
        return None

    def redo(self) -> Optional[Project]:
        data = self.snapshot_manager.redo()
        if data:
            self.current_project = Project.model_validate(data)
            return self.current_project
        return None

    def register_tool_dispatcher(self, dispatcher: Callable[[str, Dict[str, Any]], Dict[str, Any]]) -> None:
        self._tool_dispatcher = dispatcher

    def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ResourceNotFoundError("workflow", workflow_id)
        if not self._tool_dispatcher:
            raise DJError("NO_DISPATCHER", "Session tool dispatcher is not registered.")

        workflow.status = "running"
        executed_steps = []

        for idx, step in enumerate(workflow.steps):
            workflow.current_step_idx = idx
            if not step.evaluate_condition(workflow.context):
                step.status = "skipped"
                executed_steps.append({"step_id": step.step_id, "status": "skipped"})
                continue

            step.status = "running"
            try:
                # Merge dynamic context into arguments if placeholders exist
                call_args = {}
                for k, v in step.arguments.items():
                    if isinstance(v, str) and v.startswith("$"):
                        call_args[k] = workflow.context.get(v[1:], v)
                    else:
                        call_args[k] = v

                res = self._tool_dispatcher(step.tool_name, call_args)
                step.status = "completed"
                step.result = res
                # Merge result into workflow context
                if isinstance(res, dict):
                    workflow.context.update(res)
                executed_steps.append({"step_id": step.step_id, "status": "completed", "result": res})
            except Exception as e:
                step.status = "failed"
                step.error = str(e)
                executed_steps.append({"step_id": step.step_id, "status": "failed", "error": str(e)})

                if step.on_failure == "rollback":
                    self.undo()
                    workflow.status = "failed"
                    return {"workflow_id": workflow_id, "status": "rolled_back", "failed_step": step.step_id, "steps": executed_steps}
                elif step.on_failure == "continue":
                    continue
                else:  # stop
                    workflow.status = "failed"
                    return {"workflow_id": workflow_id, "status": "failed", "failed_step": step.step_id, "steps": executed_steps}

        workflow.status = "completed"
        return {"workflow_id": workflow_id, "status": "completed", "steps": executed_steps, "context": workflow.context}


_ACTIVE_SESSION: Optional[Session] = None


def get_active_session() -> Session:
    global _ACTIVE_SESSION
    if _ACTIVE_SESSION is None:
        _ACTIVE_SESSION = Session()
    return _ACTIVE_SESSION


def set_active_session(sess: Session) -> None:
    global _ACTIVE_SESSION
    _ACTIVE_SESSION = sess
