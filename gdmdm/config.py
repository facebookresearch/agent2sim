import argparse


def get_config():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument("--load_logname", type=str, default="home-2023-curated3")
    parser.add_argument("--logname_gd", type=str, default="base")
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--condition_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument(
        "--time_embedding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable", "linear", "zero"],
    )
    parser.add_argument(
        "--input_embedding",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "learnable", "linear", "identity"],
    )
    parser.add_argument("--save_images_step", type=int, default=1)
    parser.add_argument("--save_model_epoch", type=int, default=100)
    parser.add_argument(
        "--pred_type", type=str, default="diffuse", choices=["diffuse", "regress"]
    )
    parser.add_argument(
        "--use_world", action="store_true", default=False, help="Use world coordinate"
    )
    parser.add_argument(
        "--swap_cam_root", action="store_true", default=False, help="Train observer model"
    )
    parser.add_argument(
        "--norot_aug",
        action="store_true",
        default=False,
        help="Do not use rotation augmentation",
    )
    parser.add_argument("--use_test_data", action="store_true")
    parser.add_argument("--fill_to_size", type=int, default=0)

    # test
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--drop_cam", action="store_true")
    parser.add_argument("--drop_past", action="store_true")
    parser.add_argument("--drop_goal", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_ntrial", type=int, default=12)
    parser.add_argument("--suffix", type=str, default="latest")
    parser.add_argument("--use_two_agents", action="store_true")

    # for compatibility with lab4d
    parser.add_argument("--flagfile", type=str, default="")
    parser.add_argument("--load_suffix", type=str, default="")
    config = parser.parse_args()
    return config
