import torch

class TestCommands:
    """
    Commands used to run tests.
    """

    def gpu(self) -> None:
        """
        Checks if GPUs are available and working correctly.
        """
        cuda_av = torch.cuda.is_available()

        print(f'CUDA Available: {cuda_av}')

        if cuda_av:
            for i in range(torch.cuda.device_count()):
                print(f'Found GPU device: {torch.cuda.get_device_name(i)}')

                a = torch.rand(10, 10).to(f'cuda:{i}')
                b = torch.rand(10, 20).to(f'cuda:{i}')
                r = a @ b
            
            print('GPU Test successful')
