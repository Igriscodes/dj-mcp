from ai_dj.capabilities import get_system_capabilities
from ai_dj.cli import run_doctor


def test_system_capabilities_structure():
    caps = get_system_capabilities()
    assert "os" in caps
    assert "gpu" in caps
    assert "engines" in caps
    assert "codecs" in caps
    assert "features" in caps
    assert caps["engines"]["soundfile"]["available"] is True
    assert caps["engines"]["pedalboard"]["available"] is True
    assert caps["codecs"]["wav"] is True


def test_doctor_execution():
    assert run_doctor(as_json=True) == 0
    assert run_doctor(as_json=False) == 0
