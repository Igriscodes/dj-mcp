import time
from ai_dj.jobs import JobManager
from ai_dj.session import Session, Workflow


def test_job_manager_execution():
    manager = JobManager()

    def sample_work(x, y):
        return {"sum": x + y}

    job_id = manager.submit_job("Addition", sample_work, 10, 20)
    assert job_id.startswith("job_")

    time.sleep(0.1)
    status = manager.get_job(job_id)
    assert status.status == "completed"
    assert status.result == {"sum": 30}
    assert len(status.logs) >= 2


def test_sequential_workflow_execution():
    sess = Session()
    records = []

    def test_tool(tool_name, args):
        records.append((tool_name, args))
        if tool_name == "fail_step":
            raise ValueError("Intentional step failure")
        return {"output": "ok", "status_flag": "ready"}

    sess.register_tool_dispatcher(test_tool)

    wf = Workflow("wf_1", "Test Pipeline")
    wf.add_step("step_init", {"param": 1})
    wf.add_step(
        "step_conditional",
        {"param": 2},
        condition_field="status_flag",
        condition_operator="equals",
        condition_value="ready",
    )
    sess.workflows["wf_1"] = wf

    res = sess.execute_workflow("wf_1")
    assert res["status"] == "completed"
    assert len(records) == 2
