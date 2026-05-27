import os
import warnings

from vggt.utils.inference_utils import create_arg_parser, extract_frames
from vggt.utils.inference_utils import run_feed_forward_inference, run_g3t_long_inference

def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    # If input is a video file, extract frames to cache_dir; if it's a directory, use it directly
    if os.path.isfile(args.input):
        image_dir = extract_frames(args.input, args.cache_dir, args.nth_frame)
    elif os.path.isdir(args.input):
        image_dir = args.input
    else:
        parser.error(f"--input is not a valid file or directory: {args.input}")

    # Run inference based on the selected backend
    if args.backend == "feed_forward":
        run_feed_forward_inference(args, image_dir)
    elif args.backend == "g3t_long":
        run_g3t_long_inference(args, image_dir)


if __name__ == "__main__":

    warnings.filterwarnings("ignore", message="xFormers is available")
    main()
