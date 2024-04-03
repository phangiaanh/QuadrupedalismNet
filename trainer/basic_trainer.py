

class Trainer():
    def __init__(self, cfg):
        pass

    def load_network(self, network, label, epoch, network_directory=None):
    save_filename = f"{label}_{epoch}.pth"
    if network_directory is None:
        network_directory = self.save_directory

    save_path = os.path.join(network_directory, save_filename)
    return network.load_state_dict(torch.load(save_path))

    def save_network(self, network, label, epoch, network_directory= None):
      save_filename = f"{label}_{epoch}.pth"
      if network_directory is None:
          network_directory = self.save_directory
    
      save_path = os.path.join(network_directory, save_filename)
      torch.save(network.cpu().state_dict(), save_path)
      if torch.cuda.is_available():
        network.cuda()
      return