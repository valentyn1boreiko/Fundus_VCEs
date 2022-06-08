import argparse


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def get_arguments(parser=None) -> argparse.Namespace:
    if parser is None:
        parser = argparse.ArgumentParser()

    # Norm
    parser.add_argument(
        "--norm", type=str, help="The desired norm for lp-VCEs (Lp for p > 1)", default='1.5'
    )
    # Radius
    parser.add_argument("--eps_project", type=float, help="The desired radius", default=30.)

    parser.add_argument(
        "--verbose",
        help="Indicator for enforcing the cone projection",
        action="store_true",
    )

    # Misc
    parser.add_argument("--seed", type=int, help="The random seed", default=1)
    parser.add_argument("--gpu_id", type=int, help="The GPU ID", default=0)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The number number if images to sample each diffusion process",
        default=128#4,
    )
    parser.add_argument('--gpu', '--list', nargs='+', default=[0],
                        help='GPU indices, if more than 1 parallel modules will be called')

    parser.add_argument(
        "--method", type=str, help='Method to use (dvces/svces)', default='svces'
    )

    defaults = dict(
        dataset='funduskaggle',
        data_folder='',
        config='fundus.yml',
        project_folder='.',
        consistent=False,
        step_lr=-1,
        nsigma=1,
        model_types=None,
        ODI_steps=-1,
        fid_num_samples=1,
        begin_ckpt=1,
        end_ckpt=1,
        adam=False,
        D_adam=False,
        D_steps=0,
        model_epoch_num=0,
        device_ids=None,
        num_imgs=2048,
        script_type='sampling',
        classifier_type=6,
        second_classifier_type=-1,
        plot_freq=5,
        world_size=1,
        world_id=0,
    )

    add_dict_to_argparser(parser, defaults)

    args = parser.parse_args()

    return args
