
def test_imports():
    import train
    from models import vmdn

    assert hasattr(train, "load_config")
    assert hasattr(vmdn, "init_vmdn")
