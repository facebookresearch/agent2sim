import os 

def get_local_rank():
    try:
        return int(os.environ["LOCAL_RANK"])
    except:
        # print("LOCAL_RANK not found, set to 0")
        return 0

def remove_ddp_prefix(state_dict):
    """Remove distributed data parallel prefix from model checkpoint

    Args:
        state_dict (Dict): Model checkpoint
    Returns:
        new_state_dict (Dict): New model checkpoint
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict