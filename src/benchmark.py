from argparse import ArgumentParser
import json
from pathlib import Path
import os
import time
from timeit import repeat


def benchmark(config_file, routines_file):
    with open(config_file, "r") as j_file:
        config = json.load(j_file)

    with open(routines_file, "r") as j_file:
        routines = json.load(j_file)

    # Decide platform to use.
    if config["platform"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif config["platform"] == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{config['gpu_id']}"
        print(
            "Warning: setting hardware to `gpu`. "
            f"Essentia has no way to verify that the divice with id {config['gpu_id']} "
            "exists and is available, and that the required versions of the CUDA and CuDNN "
            "libraries are insateled. It is the user responsability to check the TF logs "
            "to verify the the inference happens in the GPU. "
            "Note that if any of those conditions fails, inference will happend on the cpu."
        )
    else:
        raise NotImplementedError("Options are: `gpu` or `cpu`")

    # Import after seting the platform
    from tensorflowpredictmusicnn import algoTensorflowPredictMusiCNN
    from tensorflowpredictvggish import algoTensorflowPredictVGGish
    from tensorflowpredicteffnetdiscogs import algoTensorflowPredictEffnetDiscogs

    results = dict()
    for routine in routines:
        print(f"Running test for: {routine['name']}")

        inference_class = eval(routine["class"])(
            routine["kwargs"],
            audio_duration=config["audio_duration"],
            models_base=config["models_base"],
        )

        """
        According to timeit's documentation, min() is more reliable than mean() for 
        benchmarking. Full note from https://docs.python.org/3/library/timeit.html :

        It's tempting to calculate mean and standard deviation from the result vector and
        report these. However, this is not very useful. In a typical case, the lowest value
        gives a lower bound for how fast your machine can run the given code snippet; higher
        values in the result vector are typically not caused by variability in Python's
        speed, but by other processes interfering with your timing accuracy. So the min()
        of the result is probably the only number you should be interested in. After that,
        you should look at the entire vector and apply common sense rather than statistics.
        """

        # initialization model
        instantiate_time = min(
            repeat(
                lambda: inference_class.instantiate(),
                repeat=config["repetitions"],
                number=1,
            )
        )

        # benchmark model
        inference_time = min(
            repeat(
                lambda: inference_class.inference(),
                repeat=config["repetitions"],
                number=1,
            )
        )

        results[routine["name"]] = {
            "instantiate": instantiate_time,
            "inference": inference_time,
        }

    timestamp = time.strftime("%y%m%d-%H%M%S")
    r_name = f"{timestamp}.results.{config['benchmark_name']}.{config['audio_duration']}s.{config['platform']}.json"
    results_file = Path("..", "out", r_name)
    results_file.parent.mkdir(exist_ok=True)

    results["config"] = config
    with open(results_file, "w") as jfile:
        json.dump(
            results,
            jfile,
            indent=4,
        )


if __name__ == "__main__":
    parser = ArgumentParser("A script to benchmark the Essentia Models.")
    parser.add_argument("--config-file", default="../cfg/config.json")
    parser.add_argument("--routines-file", default="../cfg/routines.json")
    args = parser.parse_args()

    benchmark(
        args.config_file,
        args.routines_file,
    )
