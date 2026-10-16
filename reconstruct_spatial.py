"""Entrypoint for RKHS FTM split inference."""

from physftm.inference.config import _parse_args as parse_rkhs_inference_args
from physftm.inference.runner import run_rkhs_split_inference

__all__ = ["main", "parse_rkhs_inference_args", "run_rkhs_split_inference"]


def main(argv=None):
    args = parse_rkhs_inference_args(argv)
    return run_rkhs_split_inference(args)


if __name__ == "__main__":
    main()
