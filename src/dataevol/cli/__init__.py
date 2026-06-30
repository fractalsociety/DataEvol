__all__ = ["app", "main"]


def main() -> None:
    from .main import main as cli_main

    cli_main()


def __getattr__(name: str):
    if name == "app":
        from .main import app, main as cli_main

        globals()["app"] = app
        globals()["main"] = cli_main
        return app
    raise AttributeError(name)
