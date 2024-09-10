import os

from absl import flags

opts = flags.FLAGS


class GDMDMConfig:
    # for compatibility reasons
    flags.DEFINE_string("load_logname", "", "")
    flags.DEFINE_string("logname_gd", "", "")
    flags.DEFINE_integer("sample_idx", 0, "")
    flags.DEFINE_integer("eval_batch_size", 1, "")
    flags.DEFINE_bool("use_test_data", False, "")