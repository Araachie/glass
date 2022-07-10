import yaml

from utils.dict_wrapper import DictWrapper


class Configuration(DictWrapper):
    """
    Represents the configuration parameters for running the process
    """

    def __init__(self, path: str):
        # Loads configuration file
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        super(Configuration, self).__init__(config)

        self.check_config()

    def check_config(self):
        assert self["training"]["batching"]["observation_stacking"] == \
            self["evaluation"]["batching"]["observation_stacking"]

        if "max_num_videos" not in self["data"]:
            self["data"]["max_num_videos"] = None

        if "offset" not in self["data"]:
            self["data"]["offset"] = None

        if "use_plain_direction" not in self["model"]["action_separator"]["action_network"]:
            self["model"]["action_separator"]["action_network"]["use_plain_direction"] = False

        if "use_gumbel" not in self["model"]["action_separator"]["action_network"]:
            self["model"]["action_separator"]["action_network"]["use_gumbel"] = False

        if "modulated_conv" not in self["model"]["action_separator"]["dynamics_network"]:
            self["model"]["action_separator"]["dynamics_network"]["modulated_conv"] = True

        if "joint_input" not in self["model"]["action_separator"]["dynamics_network"]:
            self["model"]["action_separator"]["dynamics_network"]["joint_input"] = False
