from commands.manual_test_cmd import TestCommands
import fire

class Commands:
    def __init__(self) -> None:
        self.test = TestCommands()

if __name__ == '__main__':
    fire.Fire(Commands)
