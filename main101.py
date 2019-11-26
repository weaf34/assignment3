from extra_files import (showrandom_batch,show_random_data)
from trainer.trainer101 import Food101smallTrainer

if __name__ == '__main__':
    """
    Visualize random data (samples of each class)
    """
    #show_random_data()

    """
    Visualize Random Dataloader 
    """
    #showrandom_batch()

    fdt = Food101smallTrainer()
    
    #fdt.run()
