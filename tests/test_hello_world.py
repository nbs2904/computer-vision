from src.hello_world import hello_world


def test_hello_world() -> None:
    """Dummy test"""
    assert hello_world() == "hello world!"
    