from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_file", None, "location of dataset file")
flags.mark_flag_as_required("data_file")

from .episodic import EpisodicDataset
from .crp import CRPDataset
